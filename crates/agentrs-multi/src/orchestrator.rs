use std::{collections::HashMap, sync::Arc};

use tokio::sync::Mutex;

use agentrs_core::{
    Agent as AgentTrait, AgentError, AgentOutput, CompletionRequest, LlmProvider, Message, Result,
};

use crate::{AgentGraph, EventBus, OrchestratorEvent, SharedConversation};

type SharedAgent = Arc<Mutex<Box<dyn AgentTrait>>>;

/// Routing strategy for multi-agent runs.
#[derive(Clone)]
pub enum RoutingStrategy {
    /// Run agents in the given order.
    Sequential(Vec<String>),
    /// Run agents concurrently and merge their outputs.
    Parallel(Vec<String>),
    /// Use a routing LLM to decide which agent handles the task.
    Supervisor {
        /// Provider used for routing decisions.
        llm: Arc<dyn LlmProvider>,
        /// Eligible agent names.
        agents: Vec<String>,
        /// Maximum supervisor turns.
        max_turns: usize,
    },
    /// Traverse a graph of agents.
    Graph(AgentGraph),
}

impl Default for RoutingStrategy {
    fn default() -> Self {
        Self::Sequential(Vec::new())
    }
}

/// Builder for [`MultiAgentOrchestrator`].
pub struct MultiAgentOrchestratorBuilder {
    agents: HashMap<String, SharedAgent>,
    order: Vec<String>,
    routing: Option<RoutingStrategy>,
    shared_memory: Option<SharedConversation>,
    event_bus: Option<Arc<dyn EventBus>>,
}

/// Multi-agent orchestrator.
pub struct MultiAgentOrchestrator {
    agents: HashMap<String, SharedAgent>,
    routing: RoutingStrategy,
    shared_memory: Option<SharedConversation>,
    event_bus: Option<Arc<dyn EventBus>>,
}

impl MultiAgentOrchestrator {
    /// Starts building an orchestrator.
    pub fn builder() -> MultiAgentOrchestratorBuilder {
        MultiAgentOrchestratorBuilder {
            agents: HashMap::new(),
            order: Vec::new(),
            routing: None,
            shared_memory: None,
            event_bus: None,
        }
    }

    /// Runs the configured workflow.
    pub async fn run(&mut self, input: &str) -> Result<AgentOutput> {
        match self.routing.clone() {
            RoutingStrategy::Sequential(names) => self.run_sequential(input, names).await,
            RoutingStrategy::Parallel(names) => self.run_parallel(input, names).await,
            RoutingStrategy::Supervisor {
                llm,
                agents,
                max_turns,
            } => self.run_supervisor(input, llm, agents, max_turns).await,
            RoutingStrategy::Graph(graph) => self.run_graph(input, &graph).await,
        }
    }

    /// Registers or replaces a named agent after construction.
    pub fn add_agent_boxed(&mut self, name: impl Into<String>, agent: Box<dyn AgentTrait>) {
        self.agents.insert(name.into(), Arc::new(Mutex::new(agent)));
    }

    async fn run_sequential(&mut self, input: &str, names: Vec<String>) -> Result<AgentOutput> {
        let mut current_input = input.to_string();
        let mut final_output = None;

        for name in names {
            let output = self.run_named_agent(&name, &current_input).await?;
            current_input = output.text.clone();
            final_output = Some(output);
        }

        final_output
            .ok_or_else(|| AgentError::InvalidConfiguration("no agents configured".to_string()))
    }

    async fn run_parallel(&mut self, input: &str, names: Vec<String>) -> Result<AgentOutput> {
        let futures = names.into_iter().map(|name| {
            let agent = self
                .agents
                .get(&name)
                .cloned()
                .ok_or_else(|| AgentError::AgentNotFound(name.clone()));
            let event_bus = self.event_bus.clone();
            let shared_memory = self.shared_memory.clone();
            let input = input.to_string();
            async move {
                let agent = agent?;
                let mut agent = agent.lock().await;
                let output = agent.run(&input).await?;
                if let Some(shared_memory) = shared_memory {
                    if let Some(last_message) = output.messages.last().cloned() {
                        shared_memory.add(&name, last_message).await?;
                    }
                }
                if let Some(event_bus) = event_bus {
                    event_bus
                        .publish(OrchestratorEvent::AgentCompleted {
                            agent: name.clone(),
                            output: output.clone(),
                        })
                        .await?;
                }
                Ok::<_, AgentError>((name, output))
            }
        });

        let results = futures::future::try_join_all(futures).await?;
        let mut messages = Vec::new();
        let mut text = String::new();
        for (name, output) in results {
            if !text.is_empty() {
                text.push_str("\n\n");
            }
            text.push_str(&format!("[{name}]\n{}", output.text));
            messages.extend(output.messages);
        }

        Ok(AgentOutput {
            text,
            steps: 1,
            usage: Default::default(),
            messages,
            metadata: HashMap::new(),
        })
    }

    async fn run_supervisor(
        &mut self,
        input: &str,
        llm: Arc<dyn LlmProvider>,
        agents: Vec<String>,
        max_turns: usize,
    ) -> Result<AgentOutput> {
        let agent_lines = agents
            .iter()
            .map(|name| format!("- {name}"))
            .collect::<Vec<_>>()
            .join("\n");
        let mut context = format!(
            "You are a supervisor. Available agents:\n{agent_lines}\n\nReturn JSON {{\"agent\": \"name\"}} for the best agent to handle the task. Task: {input}"
        );

        for _ in 0..max_turns {
            let response = llm
                .complete(CompletionRequest {
                    messages: vec![Message::user(context.clone())],
                    tools: None,
                    model: String::new(),
                    temperature: Some(0.0),
                    max_tokens: Some(256),
                    stream: false,
                    system: None,
                    extra: HashMap::new(),
                })
                .await?;
            let choice: serde_json::Value = serde_json::from_str(&response.message.text_content())?;
            if let Some(agent_name) = choice.get("agent").and_then(serde_json::Value::as_str) {
                return self.run_named_agent(agent_name, input).await;
            }
            context = format!(
                "{context}\nPrevious response was invalid JSON: {}",
                response.message.text_content()
            );
        }

        Err(AgentError::MaxStepsReached { steps: max_turns })
    }

    async fn run_graph(&mut self, input: &str, graph: &AgentGraph) -> Result<AgentOutput> {
        let mut current = graph.entry()?.to_string();
        let mut current_input = input.to_string();

        loop {
            let output = self.run_named_agent(&current, &current_input).await?;
            if let Some(next) = graph.next(&current, &output) {
                current = next;
                current_input = output.text.clone();
                continue;
            }
            return Ok(output);
        }
    }

    async fn run_named_agent(&mut self, name: &str, input: &str) -> Result<AgentOutput> {
        let agent = self
            .agents
            .get(name)
            .cloned()
            .ok_or_else(|| AgentError::AgentNotFound(name.to_string()))?;

        let mut agent = agent.lock().await;
        let output = agent.run(input).await?;

        if let Some(shared_memory) = &self.shared_memory {
            if let Some(last_message) = output.messages.last().cloned() {
                shared_memory.add(name, last_message).await?;
            }
        }
        if let Some(event_bus) = &self.event_bus {
            event_bus
                .publish(OrchestratorEvent::AgentCompleted {
                    agent: name.to_string(),
                    output: output.clone(),
                })
                .await?;
        }

        Ok(output)
    }
}

impl MultiAgentOrchestratorBuilder {
    /// Registers an agent by name.
    pub fn add_agent(mut self, name: impl Into<String>, agent: impl AgentTrait + 'static) -> Self {
        let name = name.into();
        self.order.push(name.clone());
        self.agents
            .insert(name, Arc::new(Mutex::new(Box::new(agent))));
        self
    }

    /// Registers a boxed agent by name.
    pub fn add_agent_boxed(mut self, name: impl Into<String>, agent: Box<dyn AgentTrait>) -> Self {
        let name = name.into();
        self.order.push(name.clone());
        self.agents.insert(name, Arc::new(Mutex::new(agent)));
        self
    }

    /// Sets the routing strategy.
    pub fn routing(mut self, routing: RoutingStrategy) -> Self {
        self.routing = Some(routing);
        self
    }

    /// Enables shared conversation memory.
    pub fn shared_memory(mut self, shared_memory: SharedConversation) -> Self {
        self.shared_memory = Some(shared_memory);
        self
    }

    /// Enables orchestration events.
    pub fn event_bus(mut self, event_bus: Arc<dyn EventBus>) -> Self {
        self.event_bus = Some(event_bus);
        self
    }

    /// Builds the orchestrator.
    pub fn build(self) -> Result<MultiAgentOrchestrator> {
        let routing = self
            .routing
            .unwrap_or_else(|| RoutingStrategy::Sequential(self.order.clone()));
        Ok(MultiAgentOrchestrator {
            agents: self.agents,
            routing,
            shared_memory: self.shared_memory,
            event_bus: self.event_bus,
        })
    }
}
