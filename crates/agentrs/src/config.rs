//! YAML-driven agent and orchestrator configuration.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
};

use agentrs_agents::{Agent, LoopStrategy};
use agentrs_core::{Agent as AgentTrait, AgentError, AgentOutput, LlmProvider, Result};
use agentrs_memory::{SlidingWindowMemory, TokenAwareMemory, VectorMemory};
use agentrs_multi::{MultiAgentOrchestrator, RoutingStrategy};
use agentrs_tools::{
    CalculatorTool, FileReadTool, FileWriteTool, ToolRegistry, WebFetchTool, WebSearchTool,
};
use serde::{Deserialize, Serialize};

#[cfg(feature = "anthropic")]
use agentrs_llm::AnthropicProvider;
#[cfg(feature = "azureopenai")]
use agentrs_llm::AzureOpenAiProvider;
#[cfg(feature = "gemini")]
use agentrs_llm::GeminiProvider;
#[cfg(feature = "ollama")]
use agentrs_llm::OllamaProvider;
#[cfg(feature = "openai")]
use agentrs_llm::OpenAiProvider;

/// Runtime loaded from YAML.
pub enum ConfiguredRuntime {
    /// A single runnable agent.
    Agent(Box<dyn AgentTrait>),
    /// A multi-agent orchestrator.
    Multi(MultiAgentOrchestrator),
}

impl ConfiguredRuntime {
    /// Runs the configured runtime with a user input.
    pub async fn run(&mut self, input: &str) -> Result<AgentOutput> {
        match self {
            Self::Agent(agent) => agent.run(input).await,
            Self::Multi(orchestrator) => orchestrator.run(input).await,
        }
    }
}

/// Loads a YAML runtime from disk.
pub async fn load_runtime_from_yaml(path: impl AsRef<Path>) -> Result<ConfiguredRuntime> {
    let content = tokio::fs::read_to_string(path).await?;
    load_runtime_from_yaml_str(&content).await
}

/// Loads a YAML runtime from a string.
pub async fn load_runtime_from_yaml_str(yaml: &str) -> Result<ConfiguredRuntime> {
    let config: RuntimeConfig = serde_yaml::from_str(yaml)
        .map_err(|error| AgentError::InvalidConfiguration(error.to_string()))?;
    config.build().await
}

/// Loads a single agent from disk.
pub async fn load_agent_from_yaml(path: impl AsRef<Path>) -> Result<Box<dyn AgentTrait>> {
    let content = tokio::fs::read_to_string(path).await?;
    load_agent_from_yaml_str(&content).await
}

/// Loads a single agent from a string.
pub async fn load_agent_from_yaml_str(yaml: &str) -> Result<Box<dyn AgentTrait>> {
    let config: AgentYamlConfig = serde_yaml::from_str(yaml)
        .map_err(|error| AgentError::InvalidConfiguration(error.to_string()))?;
    config.build_boxed().await
}

/// Loads a multi-agent orchestrator from disk.
pub async fn load_multi_agent_from_yaml(path: impl AsRef<Path>) -> Result<MultiAgentOrchestrator> {
    let content = tokio::fs::read_to_string(path).await?;
    load_multi_agent_from_yaml_str(&content).await
}

/// Loads a multi-agent orchestrator from a string.
pub async fn load_multi_agent_from_yaml_str(yaml: &str) -> Result<MultiAgentOrchestrator> {
    let config: MultiAgentYamlConfig = serde_yaml::from_str(yaml)
        .map_err(|error| AgentError::InvalidConfiguration(error.to_string()))?;
    config.build().await
}

/// Top-level YAML runtime config.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RuntimeConfig {
    /// Single-agent runtime.
    Agent(AgentYamlConfig),
    /// Multi-agent runtime.
    MultiAgent(MultiAgentYamlConfig),
}

impl RuntimeConfig {
    async fn build(self) -> Result<ConfiguredRuntime> {
        match self {
            Self::Agent(config) => Ok(ConfiguredRuntime::Agent(config.build_boxed().await?)),
            Self::MultiAgent(config) => Ok(ConfiguredRuntime::Multi(config.build().await?)),
        }
    }
}

/// YAML config for a single agent.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AgentYamlConfig {
    /// Optional friendly agent name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// LLM provider configuration.
    pub llm: LlmYamlConfig,
    /// Optional system prompt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    /// Optional memory backend.
    #[serde(default, skip_serializing_if = "MemoryYamlConfig::is_default")]
    pub memory: MemoryYamlConfig,
    /// Optional tools.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolYamlConfig>,
    /// Loop strategy.
    #[serde(default, skip_serializing_if = "LoopStrategyYamlConfig::is_default")]
    pub loop_strategy: LoopStrategyYamlConfig,
    /// Optional model override at agent level.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Sampling temperature.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Max output tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Max steps.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_steps: Option<usize>,
}

impl AgentYamlConfig {
    async fn build_boxed(self) -> Result<Box<dyn AgentTrait>> {
        let llm = build_llm(self.llm)?;
        let tools = build_tools(self.tools).await?;
        let system = self.system.clone();
        let model = self.model.clone();
        let temperature = self.temperature;
        let max_tokens = self.max_tokens;
        let max_steps = self.max_steps;
        let loop_strategy = self.loop_strategy.clone().into_loop_strategy();

        match self.memory {
            MemoryYamlConfig::InMemory => {
                let mut builder = Agent::builder().llm_arc(Arc::clone(&llm)).tools(tools);
                builder = apply_agent_options(
                    builder,
                    system,
                    model,
                    temperature,
                    max_tokens,
                    max_steps,
                    loop_strategy,
                );
                Ok(Box::new(builder.build()?))
            }
            MemoryYamlConfig::SlidingWindow { window_size } => {
                let mut builder = Agent::builder()
                    .llm_arc(Arc::clone(&llm))
                    .tools(tools)
                    .memory(SlidingWindowMemory::new(window_size));
                builder = apply_agent_options(
                    builder,
                    system,
                    model,
                    temperature,
                    max_tokens,
                    max_steps,
                    loop_strategy,
                );
                Ok(Box::new(builder.build()?))
            }
            MemoryYamlConfig::TokenAware {
                max_tokens: memory_max_tokens,
            } => {
                let mut builder = Agent::builder()
                    .llm_arc(Arc::clone(&llm))
                    .tools(tools)
                    .memory(TokenAwareMemory::new(memory_max_tokens));
                builder = apply_agent_options(
                    builder,
                    system,
                    model,
                    temperature,
                    max_tokens,
                    max_steps,
                    loop_strategy,
                );
                Ok(Box::new(builder.build()?))
            }
            MemoryYamlConfig::Vector => {
                let mut builder = Agent::builder()
                    .llm_arc(llm)
                    .tools(tools)
                    .memory(VectorMemory::new());
                builder = apply_agent_options(
                    builder,
                    system,
                    model,
                    temperature,
                    max_tokens,
                    max_steps,
                    loop_strategy,
                );
                Ok(Box::new(builder.build()?))
            }
        }
    }
}

/// YAML config for multi-agent orchestration.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MultiAgentYamlConfig {
    /// Named agents available to the orchestrator.
    pub agents: Vec<NamedAgentYamlConfig>,
    /// Routing strategy.
    #[serde(default, skip_serializing_if = "RoutingYamlConfig::is_default")]
    pub routing: RoutingYamlConfig,
}

impl MultiAgentYamlConfig {
    async fn build(self) -> Result<MultiAgentOrchestrator> {
        let mut builder = MultiAgentOrchestrator::builder();
        let default_order = self
            .agents
            .iter()
            .map(|agent| agent.name.clone())
            .collect::<Vec<_>>();

        for agent in self.agents {
            builder = builder.add_agent_boxed(agent.name, agent.agent.build_boxed().await?);
        }

        builder
            .routing(self.routing.into_routing_strategy(default_order)?)
            .build()
    }
}

/// A named agent entry inside a multi-agent YAML config.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct NamedAgentYamlConfig {
    /// Public agent name.
    pub name: String,
    /// Agent configuration.
    #[serde(flatten)]
    pub agent: AgentYamlConfig,
}

/// YAML config for memory backends.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MemoryYamlConfig {
    /// In-memory history.
    InMemory,
    /// Sliding window history.
    SlidingWindow {
        /// Number of recent messages to keep.
        window_size: usize,
    },
    /// Token-aware history.
    TokenAware {
        /// Maximum token budget retained in memory.
        max_tokens: usize,
    },
    /// Vector memory using the built-in embedder/store.
    Vector,
}

impl Default for MemoryYamlConfig {
    fn default() -> Self {
        Self::InMemory
    }
}

impl MemoryYamlConfig {
    fn is_default(&self) -> bool {
        matches!(self, Self::InMemory)
    }
}

/// YAML config for loop strategies.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum LoopStrategyYamlConfig {
    /// Standard ReAct loop.
    ReAct {
        /// Optional max steps override.
        #[serde(default)]
        max_steps: Option<usize>,
    },
    /// Single completion pass.
    ChainOfThought,
    /// Planner + executor loop.
    PlanAndExecute {
        /// Optional max steps override.
        #[serde(default)]
        max_steps: Option<usize>,
    },
    /// Custom instruction.
    Custom {
        /// Instruction prepended to the user input.
        instruction: String,
    },
}

impl Default for LoopStrategyYamlConfig {
    fn default() -> Self {
        Self::ReAct { max_steps: None }
    }
}

impl LoopStrategyYamlConfig {
    fn is_default(&self) -> bool {
        matches!(self, Self::ReAct { max_steps: None })
    }

    fn into_loop_strategy(self) -> LoopStrategy {
        match self {
            Self::ReAct { max_steps } => LoopStrategy::ReAct {
                max_steps: max_steps.unwrap_or(8),
            },
            Self::ChainOfThought => LoopStrategy::CoT,
            Self::PlanAndExecute { max_steps } => LoopStrategy::PlanAndExecute {
                max_steps: max_steps.unwrap_or(8),
            },
            Self::Custom { instruction } => LoopStrategy::Custom(instruction),
        }
    }
}

/// YAML config for routing strategies.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RoutingYamlConfig {
    /// Sequential routing.
    Sequential {
        /// Optional explicit execution order.
        #[serde(default)]
        order: Option<Vec<String>>,
    },
    /// Parallel routing.
    Parallel {
        /// Agents executed concurrently.
        agents: Vec<String>,
    },
}

impl Default for RoutingYamlConfig {
    fn default() -> Self {
        Self::Sequential { order: None }
    }
}

impl RoutingYamlConfig {
    fn is_default(&self) -> bool {
        matches!(self, Self::Sequential { order: None })
    }

    fn into_routing_strategy(self, default_order: Vec<String>) -> Result<RoutingStrategy> {
        Ok(match self {
            Self::Sequential { order } => {
                RoutingStrategy::Sequential(order.unwrap_or(default_order))
            }
            Self::Parallel { agents } => RoutingStrategy::Parallel(agents),
        })
    }
}

/// YAML config for provider creation.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "provider", rename_all = "snake_case")]
pub enum LlmYamlConfig {
    /// OpenAI-compatible config.
    OpenAi {
        /// Optional API key override.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        api_key: Option<String>,
        /// Optional base URL override.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        base_url: Option<String>,
        /// Optional model override.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        model: Option<String>,
    },
    /// Azure OpenAI config.
    AzureOpenAi {
        /// Optional API key override.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        api_key: Option<String>,
        /// Optional base URL override.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        base_url: Option<String>,
        /// Optional model override.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        model: Option<String>,
    },
    /// Anthropic config.
    Anthropic {
        /// Optional API key override.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        api_key: Option<String>,
        /// Optional base URL override.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        base_url: Option<String>,
        /// Optional model override.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        model: Option<String>,
    },
    /// Gemini config.
    Gemini {
        /// Optional API key override.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        api_key: Option<String>,
        /// Optional base URL override.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        base_url: Option<String>,
        /// Optional model override.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        model: Option<String>,
    },
    /// Ollama config.
    Ollama {
        /// Optional base URL override.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        base_url: Option<String>,
        /// Optional model override.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        model: Option<String>,
    },
}

/// YAML config for tools.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolYamlConfig {
    /// Built-in calculator tool.
    Calculator,
    /// Built-in web fetch tool.
    WebFetch,
    /// Built-in web search tool.
    WebSearch,
    /// Built-in file read tool.
    FileRead {
        /// Optional allowed root directory.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        root: Option<String>,
    },
    /// Built-in file write tool.
    FileWrite {
        /// Optional allowed root directory.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        root: Option<String>,
    },
    /// MCP tool source.
    Mcp {
        /// MCP subprocess command or HTTP endpoint.
        target: String,
        /// Optional API key for web MCP endpoints.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        api_key: Option<String>,
        /// Optional API key header override.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        api_key_header: Option<String>,
        /// Optional API key prefix override.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        api_key_prefix: Option<String>,
        /// Optional extra HTTP headers for web MCP.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        headers: Option<HashMap<String, String>>,
    },
}

fn build_llm(config: LlmYamlConfig) -> Result<Arc<dyn LlmProvider>> {
    match config {
        #[cfg(feature = "openai")]
        LlmYamlConfig::OpenAi {
            api_key,
            base_url,
            model,
        } => {
            let mut builder = OpenAiProvider::from_env();
            if let Some(api_key) = api_key {
                builder = builder.api_key(api_key);
            }
            if let Some(base_url) = base_url {
                builder = builder.base_url(base_url);
            }
            if let Some(model) = model {
                builder = builder.model(model);
            }
            Ok(Arc::new(builder.build()?))
        }
        #[cfg(feature = "azureopenai")]
        LlmYamlConfig::AzureOpenAi {
            api_key,
            base_url,
            model,
        } => {
            let mut builder = AzureOpenAiProvider::from_env();
            if let Some(api_key) = api_key {
                builder = builder.api_key(api_key);
            }
            if let Some(base_url) = base_url {
                builder = builder.base_url(base_url);
            }
            if let Some(model) = model {
                builder = builder.model(model);
            }
            Ok(Arc::new(builder.build()?))
        }
        #[cfg(feature = "anthropic")]
        LlmYamlConfig::Anthropic {
            api_key,
            base_url,
            model,
        } => {
            let mut builder = AnthropicProvider::from_env();
            if let Some(api_key) = api_key {
                builder = builder.api_key(api_key);
            }
            if let Some(base_url) = base_url {
                builder = builder.base_url(base_url);
            }
            if let Some(model) = model {
                builder = builder.model(model);
            }
            Ok(Arc::new(builder.build()?))
        }
        #[cfg(feature = "gemini")]
        LlmYamlConfig::Gemini {
            api_key,
            base_url,
            model,
        } => {
            let mut builder = GeminiProvider::from_env();
            if let Some(api_key) = api_key {
                builder = builder.api_key(api_key);
            }
            if let Some(base_url) = base_url {
                builder = builder.base_url(base_url);
            }
            if let Some(model) = model {
                builder = builder.model(model);
            }
            Ok(Arc::new(builder.build()?))
        }
        #[cfg(feature = "ollama")]
        LlmYamlConfig::Ollama { base_url, model } => {
            let mut builder = OllamaProvider::builder();
            if let Some(base_url) = base_url {
                builder = builder.base_url(base_url);
            }
            if let Some(model) = model {
                builder = builder.model(model);
            }
            Ok(Arc::new(builder.build()?))
        }
        _ => Err(AgentError::InvalidConfiguration(
            "requested LLM provider feature is not enabled in this build".to_string(),
        )),
    }
}

async fn build_tools(configs: Vec<ToolYamlConfig>) -> Result<ToolRegistry> {
    let mut registry = ToolRegistry::new();

    for config in configs {
        registry = match config {
            ToolYamlConfig::Calculator => registry.register(CalculatorTool::new()),
            ToolYamlConfig::WebFetch => registry.register(WebFetchTool::new()),
            ToolYamlConfig::WebSearch => registry.register(WebSearchTool::new()),
            ToolYamlConfig::FileRead { root } => {
                registry.register(FileReadTool::new(root.map(PathBuf::from)))
            }
            ToolYamlConfig::FileWrite { root } => {
                registry.register(FileWriteTool::new(root.map(PathBuf::from)))
            }
            ToolYamlConfig::Mcp {
                target,
                api_key,
                api_key_header,
                api_key_prefix,
                headers,
            } => {
                if target.starts_with("http://") || target.starts_with("https://") {
                    let mut options = agentrs_mcp::WebMcpOptions::new();
                    if let Some(headers) = headers {
                        options = options.headers(headers);
                    }
                    if let Some(api_key) = api_key {
                        options = options.api_key(api_key);
                    }
                    if let Some(api_key_header) = api_key_header {
                        options = options.api_key_header(api_key_header);
                    }
                    if let Some(api_key_prefix) = api_key_prefix {
                        options = options.api_key_prefix(api_key_prefix);
                    }
                    registry
                        .register_mcp_http_with_options(&target, options)
                        .await?
                } else {
                    registry.register_mcp(&target).await?
                }
            }
        };
    }

    Ok(registry)
}

fn apply_agent_options<State, M>(
    mut builder: agentrs_agents::AgentBuilder<State, M>,
    system: Option<String>,
    model: Option<String>,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    max_steps: Option<usize>,
    loop_strategy: LoopStrategy,
) -> agentrs_agents::AgentBuilder<State, M> {
    builder = builder.loop_strategy(loop_strategy);
    if let Some(system) = system {
        builder = builder.system(system);
    }
    if let Some(model) = model {
        builder = builder.model(model);
    }
    if let Some(temperature) = temperature {
        builder = builder.temperature(temperature);
    }
    if let Some(max_tokens) = max_tokens {
        builder = builder.max_tokens(max_tokens);
    }
    if let Some(max_steps) = max_steps {
        builder = builder.max_steps(max_steps);
    }
    builder
}
