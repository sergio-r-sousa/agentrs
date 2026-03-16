#![forbid(unsafe_code)]

//! Foundational traits and shared types for the `agentrs` SDK.

pub mod error;
pub mod streaming;
pub mod testing;
pub mod types;

use async_trait::async_trait;
use futures::stream::BoxStream;

pub use error::{AgentError, BoxError, McpError, MemoryError, ProviderError, Result, ToolError};
pub use types::{
    AgentEvent, AgentOutput, CompletionRequest, CompletionResponse, ContentPart, Message,
    MessageContent, Role, StopReason, StreamChunk, ToolCall, ToolCallDelta, ToolContent,
    ToolDefinition, ToolOutput, Usage,
};

/// Unified contract for chat-completion capable LLM providers.
#[async_trait]
pub trait LlmProvider: Send + Sync + 'static {
    /// Executes a single non-streaming completion request.
    async fn complete(&self, req: CompletionRequest) -> Result<CompletionResponse>;

    /// Executes a streaming completion request.
    async fn stream(&self, req: CompletionRequest) -> Result<BoxStream<'_, Result<StreamChunk>>>;

    /// Returns the provider name.
    fn name(&self) -> &str;
}

/// Unified contract for agent tools.
#[async_trait]
pub trait Tool: Send + Sync + 'static {
    /// Returns the public tool name.
    fn name(&self) -> &str;

    /// Returns a short tool description.
    fn description(&self) -> &str;

    /// Returns a JSON schema describing the tool input.
    fn schema(&self) -> serde_json::Value;

    /// Executes the tool.
    async fn call(&self, input: serde_json::Value) -> Result<ToolOutput>;
}

/// Contract for agent memory backends.
#[async_trait]
pub trait Memory: Send + Sync + 'static {
    /// Stores a message under a logical key.
    async fn store(&mut self, key: &str, value: Message) -> Result<()>;

    /// Retrieves relevant messages for a query.
    async fn retrieve(&self, query: &str, limit: usize) -> Result<Vec<Message>>;

    /// Returns the full conversation history.
    async fn history(&self) -> Result<Vec<Message>>;

    /// Clears the memory backend.
    async fn clear(&mut self) -> Result<()>;
}

/// Contract for executable agents.
#[async_trait]
pub trait Agent: Send + Sync + 'static {
    /// Runs the agent to completion for a user input.
    async fn run(&mut self, input: &str) -> Result<AgentOutput>;

    /// Runs the agent as an event stream.
    async fn stream_run(&mut self, input: &str) -> Result<BoxStream<'_, Result<AgentEvent>>>;
}
