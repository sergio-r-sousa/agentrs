use std::{collections::HashMap, sync::Arc};

use async_stream::try_stream;
use async_trait::async_trait;
use futures::{stream::BoxStream, StreamExt};

use agentrs_core::{
    Agent as AgentTrait, AgentError, AgentEvent, AgentOutput, CompletionRequest,
    CompletionResponse, LlmProvider, Memory, Message, Result, ToolOutput,
};
use agentrs_tools::ToolRegistry;

/// Runtime agent configuration.
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// Default model name.
    pub model: String,
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Maximum output tokens.
    pub max_tokens: Option<u32>,
    /// Loop strategy.
    pub loop_strategy: LoopStrategy,
    /// Maximum ReAct iterations.
    pub max_steps: usize,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            model: String::new(),
            temperature: Some(0.2),
            max_tokens: Some(4096),
            loop_strategy: LoopStrategy::ReAct { max_steps: 8 },
            max_steps: 8,
        }
    }
}

/// Built-in loop strategies.
#[derive(Debug, Clone)]
pub enum LoopStrategy {
    /// Standard reason-and-act loop.
    ReAct {
        /// Maximum number of reasoning/tool iterations.
        max_steps: usize,
    },
    /// Single-pass answer generation.
    CoT,
    /// Planner + executor loop using the same LLM.
    PlanAndExecute {
        /// Maximum number of execution steps to budget for planning.
        max_steps: usize,
    },
    /// Custom instruction prepended before execution.
    Custom(String),
}

impl LoopStrategy {
    pub(crate) fn max_steps_hint(&self, fallback: usize) -> usize {
        match self {
            Self::ReAct { max_steps } | Self::PlanAndExecute { max_steps } => *max_steps,
            Self::CoT | Self::Custom(_) => fallback,
        }
    }
}

/// Runnable agent implementation.
pub struct AgentRunner<M> {
    llm: Arc<dyn LlmProvider>,
    memory: M,
    tools: ToolRegistry,
    system_prompt: Option<String>,
    config: AgentConfig,
}

impl<M> AgentRunner<M>
where
    M: Memory,
{
    /// Creates a new agent runner.
    pub fn new(
        llm: Arc<dyn LlmProvider>,
        memory: M,
        tools: ToolRegistry,
        system_prompt: Option<String>,
        config: AgentConfig,
    ) -> Self {
        Self {
            llm,
            memory,
            tools,
            system_prompt,
            config,
        }
    }

    /// Runs the agent to completion.
    pub async fn run(&mut self, input: &str) -> Result<AgentOutput> {
        AgentTrait::run(self, input).await
    }

    /// Runs the agent as a stream of events.
    pub async fn stream_run(&mut self, input: &str) -> Result<BoxStream<'_, Result<AgentEvent>>> {
        AgentTrait::stream_run(self, input).await
    }

    async fn run_react(&mut self, input: &str) -> Result<AgentOutput> {
        self.memory.store("user", Message::user(input)).await?;

        let max_steps = match self.config.loop_strategy {
            LoopStrategy::ReAct { max_steps } => max_steps,
            _ => self.config.max_steps,
        };

        for step in 1..=max_steps {
            let history = self.memory.history().await?;
            let request = self.build_request(history, !self.tools.is_empty());
            let response = self.llm.complete(request).await?;
            let assistant_message = response.message.clone();
            self.memory
                .store("assistant", assistant_message.clone())
                .await?;

            if let Some(tool_calls) = assistant_message
                .tool_calls
                .clone()
                .filter(|calls| !calls.is_empty())
            {
                for message in self.execute_tool_calls(tool_calls).await? {
                    self.memory.store("tool", message).await?;
                }
                continue;
            }

            return self.finish_output(response, step).await;
        }

        Err(AgentError::MaxStepsReached { steps: max_steps })
    }

    async fn run_cot(&mut self, input: &str) -> Result<AgentOutput> {
        self.memory.store("user", Message::user(input)).await?;
        let history = self.memory.history().await?;
        let response = self
            .llm
            .complete(self.build_request(history, false))
            .await?;
        self.memory
            .store("assistant", response.message.clone())
            .await?;
        self.finish_output(response, 1).await
    }

    async fn run_plan_execute(&mut self, input: &str, max_steps: usize) -> Result<AgentOutput> {
        let planner_prompt =
            format!("Create a concise numbered plan to solve the user task. Task: {input}");
        let plan_response = self
            .llm
            .complete(CompletionRequest {
                messages: vec![Message::user(planner_prompt)],
                tools: None,
                model: self.config.model.clone(),
                temperature: Some(0.1),
                max_tokens: self.config.max_tokens,
                stream: false,
                system: self.system_prompt.clone(),
                extra: HashMap::new(),
            })
            .await?;

        self.memory
            .store(
                "plan",
                Message::assistant(plan_response.message.text_content()),
            )
            .await?;
        let execution_prompt = format!(
            "Use this plan to solve the task.\nPlan:\n{}\n\nTask: {input}",
            plan_response.message.text_content()
        );
        self.memory
            .store("user", Message::user(execution_prompt))
            .await?;

        let mut output = self.run_react(input).await?;
        output.steps = output.steps.max(max_steps.min(output.steps.max(1)));
        Ok(output)
    }

    fn build_request(&self, history: Vec<Message>, include_tools: bool) -> CompletionRequest {
        CompletionRequest {
            messages: history,
            tools: include_tools.then(|| self.tools.to_definitions()),
            model: self.config.model.clone(),
            temperature: self.config.temperature,
            max_tokens: self.config.max_tokens,
            stream: false,
            system: self.system_prompt.clone(),
            extra: HashMap::new(),
        }
    }

    async fn execute_tool_calls(
        &self,
        tool_calls: Vec<agentrs_core::ToolCall>,
    ) -> Result<Vec<Message>> {
        let futures = tool_calls.into_iter().map(|tool_call| {
            let tools = self.tools.clone();
            async move {
                let output = match tools
                    .call(&tool_call.name, tool_call.arguments.clone())
                    .await
                {
                    Ok(output) => output,
                    Err(error) => ToolOutput::error(error.to_string()),
                };
                Ok::<_, AgentError>(Message::tool_result(tool_call.id, tool_call.name, output))
            }
        });
        futures::future::try_join_all(futures).await
    }

    async fn finish_output(
        &self,
        response: CompletionResponse,
        steps: usize,
    ) -> Result<AgentOutput> {
        let history = self.memory.history().await?;
        Ok(AgentOutput {
            text: response.message.text_content(),
            steps,
            usage: response.usage,
            messages: history,
            metadata: HashMap::new(),
        })
    }
}

#[async_trait]
impl<M> AgentTrait for AgentRunner<M>
where
    M: Memory,
{
    async fn run(&mut self, input: &str) -> Result<AgentOutput> {
        match self.config.loop_strategy.clone() {
            LoopStrategy::ReAct { .. } => self.run_react(input).await,
            LoopStrategy::CoT => self.run_cot(input).await,
            LoopStrategy::PlanAndExecute { max_steps } => {
                self.run_plan_execute(input, max_steps).await
            }
            LoopStrategy::Custom(instruction) => {
                let input = format!("{instruction}\n\nUser task: {input}");
                self.run_cot(&input).await
            }
        }
    }

    async fn stream_run(&mut self, input: &str) -> Result<BoxStream<'_, Result<AgentEvent>>> {
        let output = self.run(input).await?;
        Ok(try_stream! {
            yield AgentEvent::Thinking("completed".to_string());
            for token in output.text.split_whitespace() {
                yield AgentEvent::Token(format!("{token} "));
            }
            yield AgentEvent::Done(output);
        }
        .boxed())
    }
}
