use std::{marker::PhantomData, sync::Arc};

use agentrs_core::{AgentError, LlmProvider, Memory, Result, Tool};
use agentrs_memory::InMemoryMemory;
use agentrs_tools::ToolRegistry;

use crate::runner::{AgentConfig, AgentRunner, LoopStrategy};

/// Typestate marker indicating no LLM has been configured.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoLlm;

/// Typestate marker indicating an LLM has been configured.
#[derive(Debug, Clone, Copy, Default)]
pub struct WithLlm;

/// User-facing entrypoint for agent construction.
pub struct Agent;

impl Agent {
    /// Starts building an agent.
    pub fn builder() -> AgentBuilder<NoLlm, InMemoryMemory> {
        AgentBuilder::new()
    }
}

/// Typestate builder for [`AgentRunner`].
pub struct AgentBuilder<State, M = InMemoryMemory> {
    state: PhantomData<State>,
    llm: Option<Arc<dyn LlmProvider>>,
    memory: M,
    tools: ToolRegistry,
    system_prompt: Option<String>,
    config: AgentConfig,
}

impl AgentBuilder<NoLlm, InMemoryMemory> {
    /// Creates a builder with in-memory defaults.
    pub fn new() -> Self {
        Self {
            state: PhantomData,
            llm: None,
            memory: InMemoryMemory::new(),
            tools: ToolRegistry::new(),
            system_prompt: None,
            config: AgentConfig::default(),
        }
    }
}

impl<State, M> AgentBuilder<State, M> {
    /// Sets the system prompt.
    pub fn system(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Replaces the tool registry.
    pub fn tools(mut self, tools: ToolRegistry) -> Self {
        self.tools = tools;
        self
    }

    /// Registers a single tool.
    pub fn tool(mut self, tool: impl Tool + 'static) -> Self {
        self.tools = self.tools.register(tool);
        self
    }

    /// Replaces the memory backend.
    pub fn memory<NewM>(self, memory: NewM) -> AgentBuilder<State, NewM> {
        AgentBuilder {
            state: PhantomData,
            llm: self.llm,
            memory,
            tools: self.tools,
            system_prompt: self.system_prompt,
            config: self.config,
        }
    }

    /// Sets the default model.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.config.model = model.into();
        self
    }

    /// Sets the sampling temperature.
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.config.temperature = Some(temperature);
        self
    }

    /// Sets the maximum output tokens.
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.config.max_tokens = Some(max_tokens);
        self
    }

    /// Sets the loop strategy.
    pub fn loop_strategy(mut self, loop_strategy: LoopStrategy) -> Self {
        self.config.max_steps = loop_strategy.max_steps_hint(self.config.max_steps);
        self.config.loop_strategy = loop_strategy;
        self
    }

    /// Sets the maximum tool/action steps.
    pub fn max_steps(mut self, max_steps: usize) -> Self {
        self.config.max_steps = max_steps;
        self
    }
}

impl<M> AgentBuilder<NoLlm, M> {
    /// Configures the LLM provider and advances the typestate.
    pub fn llm(self, llm: impl LlmProvider + 'static) -> AgentBuilder<WithLlm, M> {
        AgentBuilder {
            state: PhantomData,
            llm: Some(Arc::new(llm)),
            memory: self.memory,
            tools: self.tools,
            system_prompt: self.system_prompt,
            config: self.config,
        }
    }

    /// Configures a shared LLM provider and advances the typestate.
    pub fn llm_arc(self, llm: Arc<dyn LlmProvider>) -> AgentBuilder<WithLlm, M> {
        AgentBuilder {
            state: PhantomData,
            llm: Some(llm),
            memory: self.memory,
            tools: self.tools,
            system_prompt: self.system_prompt,
            config: self.config,
        }
    }
}

impl<M> AgentBuilder<WithLlm, M>
where
    M: Memory,
{
    /// Builds the configured agent.
    pub fn build(self) -> Result<AgentRunner<M>> {
        let llm = self.llm.ok_or(AgentError::NoLlmProvider)?;
        Ok(AgentRunner::new(
            llm,
            self.memory,
            self.tools,
            self.system_prompt,
            self.config,
        ))
    }
}
