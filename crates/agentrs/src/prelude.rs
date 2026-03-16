//! Common imports for end users.

pub use crate::config::{
    load_agent_from_yaml, load_agent_from_yaml_str, load_multi_agent_from_yaml,
    load_multi_agent_from_yaml_str, load_runtime_from_yaml, load_runtime_from_yaml_str,
    AgentYamlConfig, ConfiguredRuntime, LlmYamlConfig, LoopStrategyYamlConfig, MemoryYamlConfig,
    MultiAgentYamlConfig, NamedAgentYamlConfig, RoutingYamlConfig, RuntimeConfig, ToolYamlConfig,
};
pub use agentrs_agents::{Agent, AgentBuilder, AgentConfig, AgentRunner, LoopStrategy};
pub use agentrs_core::{
    Agent as AgentTrait, AgentError, AgentEvent, AgentOutput, CompletionRequest,
    CompletionResponse, ContentPart, LlmProvider, Memory, Message, MessageContent, Result, Role,
    StreamChunk, Tool, ToolCall, ToolContent, ToolDefinition, ToolOutput, Usage,
};
#[cfg(feature = "anthropic")]
pub use agentrs_llm::AnthropicProvider;
#[cfg(feature = "azureopenai")]
pub use agentrs_llm::AzureOpenAiProvider;
#[cfg(feature = "gemini")]
pub use agentrs_llm::GeminiProvider;
#[cfg(feature = "ollama")]
pub use agentrs_llm::OllamaProvider;
#[cfg(feature = "openai")]
pub use agentrs_llm::OpenAiProvider;
#[cfg(feature = "mcp")]
pub use agentrs_mcp::{McpClient, WebMcpOptions};
pub use agentrs_memory::{InMemoryMemory, SlidingWindowMemory, TokenAwareMemory, VectorMemory};
pub use agentrs_multi::{
    AgentGraph, EdgeCondition, GraphEdge, InMemoryBus, MultiAgentOrchestrator, RoutingStrategy,
    SharedConversation,
};
pub use agentrs_tools::tool;
#[cfg(feature = "tool-bash")]
pub use agentrs_tools::BashTool;
#[cfg(feature = "tool-python")]
pub use agentrs_tools::PythonTool;
#[cfg(feature = "tool-fetch")]
pub use agentrs_tools::WebFetchTool;
#[cfg(feature = "tool-search")]
pub use agentrs_tools::WebSearchTool;
pub use agentrs_tools::{
    CalculatorTool, FileReadTool, FileWriteTool, IntoToolOutput, ToolContext, ToolRegistry,
};
pub use async_trait::async_trait;
pub use futures::StreamExt;
