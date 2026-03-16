//! Shared SDK error types.

use std::error::Error as StdError;

/// Common boxed error type.
pub type BoxError = Box<dyn StdError + Send + Sync>;

/// Shared result type across the SDK.
pub type Result<T> = std::result::Result<T, AgentError>;

/// Errors produced by LLM providers.
#[derive(Debug, thiserror::Error)]
pub enum ProviderError {
    /// The provider is missing an API key.
    #[error("Missing API key - set {env_var}")]
    MissingApiKey {
        /// Name of the required environment variable.
        env_var: &'static str,
    },
    /// A transport-level failure occurred.
    #[error("HTTP request failed: {0}")]
    Http(String),
    /// The remote API returned an error.
    #[error("API error {status}: {message}")]
    Api {
        /// HTTP status code returned by the provider.
        status: u16,
        /// Provider response body or error message.
        message: String,
    },
    /// The provider response could not be parsed.
    #[error("Response parsing failed: {0}")]
    Parse(#[from] serde_json::Error),
    /// The provider reported rate limiting.
    #[error("Rate limited - retry after {retry_after_secs}s")]
    RateLimited {
        /// Suggested retry delay in seconds.
        retry_after_secs: u64,
    },
    /// The model context window was exceeded.
    #[error("Context window exceeded")]
    ContextWindowExceeded,
    /// The provider returned an unexpected payload.
    #[error("Invalid provider response: {0}")]
    InvalidResponse(String),
    /// The requested feature is not supported by the provider.
    #[error("Unsupported provider feature: {0}")]
    Unsupported(&'static str),
}

/// Errors produced by tool execution.
#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    /// The input payload did not match the tool schema.
    #[error("Invalid tool input: {0}")]
    InvalidInput(String),
    /// Tool execution failed.
    #[error("Tool execution failed: {0}")]
    Execution(String),
    /// The tool operation timed out.
    #[error("Tool execution timed out")]
    Timeout,
    /// The tool attempted a forbidden operation.
    #[error("Permission denied: {0}")]
    PermissionDenied(String),
}

/// Errors produced by memory backends.
#[derive(Debug, thiserror::Error)]
pub enum MemoryError {
    /// The backend failed.
    #[error("Memory backend failed: {0}")]
    Backend(String),
    /// Serialization failed.
    #[error("Memory serialization failed: {0}")]
    Serialization(#[from] serde_json::Error),
}

/// Errors produced by MCP clients or adapters.
#[derive(Debug, thiserror::Error)]
pub enum McpError {
    /// The provided command string was invalid.
    #[error("Invalid MCP command")]
    InvalidCommand,
    /// Spawning the MCP server failed.
    #[error("Failed to spawn MCP server: {0}")]
    SpawnFailed(String),
    /// Framing or protocol parsing failed.
    #[error("MCP protocol error: {0}")]
    Protocol(String),
    /// The MCP server reported an error.
    #[error("MCP response error: {0}")]
    Response(String),
    /// The request timed out.
    #[error("MCP request timed out")]
    Timeout,
}

/// Top-level SDK error.
#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    /// Provider integration failure.
    #[error("LLM provider error: {0}")]
    Provider(#[from] ProviderError),
    /// Tool validation or execution failure.
    #[error("Tool error: {0}")]
    ToolError(#[from] ToolError),
    /// Tool execution failed with contextual tool name.
    #[error("Tool execution failed '{name}': {source}")]
    Tool {
        /// Name of the tool that failed.
        name: String,
        #[source]
        /// Original error returned by the tool.
        source: BoxError,
    },
    /// MCP failure.
    #[error("MCP error: {0}")]
    Mcp(#[from] McpError),
    /// Memory failure.
    #[error("Memory error: {0}")]
    Memory(#[from] MemoryError),
    /// Maximum agent iterations were reached.
    #[error("Max steps reached: {steps}")]
    MaxStepsReached {
        /// Number of steps attempted before stopping.
        steps: usize,
    },
    /// JSON serialization failure.
    #[error("Serialization error: {0}")]
    Serde(#[from] serde_json::Error),
    /// I/O failure.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// Required field is absent.
    #[error("Missing required field: {0}")]
    MissingField(&'static str),
    /// Agent was built without an LLM.
    #[error("No LLM provider configured")]
    NoLlmProvider,
    /// Requested tool name is unknown.
    #[error("Tool not found: {0}")]
    ToolNotFound(String),
    /// Requested agent name is unknown.
    #[error("Agent not found: {0}")]
    AgentNotFound(String),
    /// Configuration is invalid.
    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),
    /// Streaming payload was invalid.
    #[error("Invalid stream payload")]
    InvalidStream,
    /// Context window was exceeded.
    #[error("Context window exceeded")]
    ContextWindowExceeded,
    /// Catch-all error message.
    #[error("{0}")]
    Message(String),
}

impl AgentError {
    /// Wraps an arbitrary error with tool context.
    pub fn tool_failure(name: impl Into<String>, source: impl Into<BoxError>) -> Self {
        Self::Tool {
            name: name.into(),
            source: source.into(),
        }
    }
}
