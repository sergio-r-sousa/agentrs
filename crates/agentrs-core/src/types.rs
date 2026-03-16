//! Shared types used across providers, tools, memory, and agents.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// A chat message exchanged with an LLM.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Message {
    /// The actor that produced the message.
    pub role: Role,
    /// The message payload.
    pub content: MessageContent,
    /// Optional tool calls requested by an assistant message.
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Tool-call correlation id for tool result messages.
    pub tool_call_id: Option<String>,
    /// Provider- or application-specific metadata.
    pub metadata: HashMap<String, serde_json::Value>,
}

impl Message {
    /// Creates a system message.
    pub fn system(content: impl Into<String>) -> Self {
        Self::new(Role::System, MessageContent::Text(content.into()))
    }

    /// Creates a user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self::new(Role::User, MessageContent::Text(content.into()))
    }

    /// Creates an assistant message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(Role::Assistant, MessageContent::Text(content.into()))
    }

    /// Creates a message from a raw content payload.
    pub fn new(role: Role, content: MessageContent) -> Self {
        Self {
            role,
            content,
            tool_calls: None,
            tool_call_id: None,
            metadata: HashMap::new(),
        }
    }

    /// Creates a tool result message.
    pub fn tool_result(
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        output: ToolOutput,
    ) -> Self {
        let tool_name = tool_name.into();
        let mut metadata = HashMap::new();
        metadata.insert(
            "tool_name".to_string(),
            serde_json::Value::String(tool_name),
        );
        Self {
            role: Role::Tool,
            content: MessageContent::Text(output.text_content()),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
            metadata,
        }
    }

    /// Returns the textual representation of the message.
    pub fn text_content(&self) -> String {
        self.content.text()
    }
}

/// Message role.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// System or developer instruction.
    System,
    /// End-user input.
    User,
    /// Model-generated response.
    Assistant,
    /// Tool execution result.
    Tool,
}

/// Multimodal content part.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    /// Plain text content.
    Text {
        /// Plain text payload.
        text: String,
    },
    /// Inline image reference.
    Image {
        /// Encoded or remote image location.
        data: String,
        /// Optional MIME type.
        mime_type: Option<String>,
    },
    /// Structured JSON payload.
    Json {
        /// Arbitrary JSON value.
        value: serde_json::Value,
    },
    /// External resource reference.
    Resource {
        /// Resource URI.
        uri: String,
        /// Optional text representation.
        text: Option<String>,
    },
}

/// Message content.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum MessageContent {
    /// Simple text payload.
    Text(String),
    /// Rich multipart payload.
    Parts(Vec<ContentPart>),
}

impl MessageContent {
    /// Returns a plain-text projection of the content.
    pub fn text(&self) -> String {
        match self {
            Self::Text(text) => text.clone(),
            Self::Parts(parts) => parts
                .iter()
                .filter_map(|part| match part {
                    ContentPart::Text { text } => Some(text.clone()),
                    ContentPart::Json { value } => Some(value.to_string()),
                    ContentPart::Resource { text, .. } => text.clone(),
                    ContentPart::Image { .. } => None,
                })
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }
}

/// Tool-call definition sent to a provider.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolDefinition {
    /// Tool name.
    pub name: String,
    /// Tool description.
    pub description: String,
    /// JSON schema for tool arguments.
    pub schema: serde_json::Value,
}

/// Tool call emitted by a provider.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
    /// Provider-generated id.
    pub id: String,
    /// Tool name.
    pub name: String,
    /// Tool arguments.
    pub arguments: serde_json::Value,
}

/// Partial tool-call delta used in streaming.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCallDelta {
    /// Tool call index for chunk aggregation.
    pub index: usize,
    /// Optional id fragment.
    pub id: Option<String>,
    /// Optional name fragment.
    pub name: Option<String>,
    /// Optional arguments fragment.
    pub arguments_delta: Option<String>,
}

/// Unified completion request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
    /// Conversation messages.
    pub messages: Vec<Message>,
    /// Optional tool definitions.
    pub tools: Option<Vec<ToolDefinition>>,
    /// Target model name.
    pub model: String,
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Maximum output token count.
    pub max_tokens: Option<u32>,
    /// Whether streaming is requested.
    pub stream: bool,
    /// Optional out-of-band system instruction.
    pub system: Option<String>,
    /// Provider-specific extra request fields.
    pub extra: HashMap<String, serde_json::Value>,
}

/// Unified completion response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse {
    /// Assistant message.
    pub message: Message,
    /// Reason generation stopped.
    pub stop_reason: StopReason,
    /// Usage information.
    pub usage: Usage,
    /// Model used by the provider.
    pub model: String,
    /// Provider-specific raw payload.
    pub raw: Option<serde_json::Value>,
}

impl CompletionResponse {
    /// Creates a plain-text response.
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            message: Message::assistant(text),
            stop_reason: StopReason::Stop,
            usage: Usage::default(),
            model: String::new(),
            raw: None,
        }
    }

    /// Creates a single tool-call response.
    pub fn tool_call(name: impl Into<String>, arguments: serde_json::Value) -> Self {
        let call = ToolCall {
            id: uuid::Uuid::new_v4().to_string(),
            name: name.into(),
            arguments,
        };
        let mut message = Message::assistant("");
        message.tool_calls = Some(vec![call]);
        Self {
            message,
            stop_reason: StopReason::ToolUse,
            usage: Usage::default(),
            model: String::new(),
            raw: None,
        }
    }
}

/// Stop reason returned by providers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    /// Regular stop sequence or end-of-turn.
    Stop,
    /// Provider-specific end turn marker.
    EndTurn,
    /// The model requested a tool call.
    ToolUse,
    /// Output token limit was hit.
    MaxTokens,
    /// Provider-specific reason.
    Other(String),
}

impl Default for StopReason {
    fn default() -> Self {
        Self::Stop
    }
}

/// Token usage metadata.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct Usage {
    /// Input token count.
    pub input_tokens: u32,
    /// Output token count.
    pub output_tokens: u32,
    /// Aggregate token count.
    pub total_tokens: u32,
}

/// Streaming chunk emitted by providers.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StreamChunk {
    /// Incremental text delta.
    pub delta: String,
    /// Incremental tool-call data.
    pub tool_call_delta: Option<Vec<ToolCallDelta>>,
    /// Optional finish reason once the stream completes.
    pub finish_reason: Option<StopReason>,
}

/// Final output of an agent run.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AgentOutput {
    /// Final text answer.
    pub text: String,
    /// Number of reasoning steps executed.
    pub steps: usize,
    /// Aggregate usage.
    pub usage: Usage,
    /// Final message history snapshot.
    pub messages: Vec<Message>,
    /// Extra metadata.
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Incremental events emitted by streaming agents.
#[derive(Debug, Clone)]
pub enum AgentEvent {
    /// Signals the beginning of a reasoning step.
    Thinking(String),
    /// Signals a tool invocation.
    ToolCall {
        /// Tool name.
        name: String,
        /// Tool arguments.
        input: serde_json::Value,
    },
    /// Signals a completed tool invocation.
    ToolResult {
        /// Tool name.
        name: String,
        /// Human-readable tool output.
        output: String,
    },
    /// Incremental token from the provider.
    Token(String),
    /// Terminal event with final output.
    Done(AgentOutput),
}

/// Tool output payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolOutput {
    /// Structured tool content items.
    pub content: Vec<ToolContent>,
    /// Whether the payload represents an error.
    pub is_error: bool,
}

impl ToolOutput {
    /// Creates a plain-text tool result.
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            content: vec![ToolContent::Text { text: text.into() }],
            is_error: false,
        }
    }

    /// Creates an error tool result.
    pub fn error(text: impl Into<String>) -> Self {
        Self {
            content: vec![ToolContent::Text { text: text.into() }],
            is_error: true,
        }
    }

    /// Returns the textual projection of the output.
    pub fn text_content(&self) -> String {
        self.content
            .iter()
            .filter_map(|item| match item {
                ToolContent::Text { text } => Some(text.clone()),
                ToolContent::Resource { text, .. } => text.clone(),
                ToolContent::Image { .. } => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// Structured tool content item.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ToolContent {
    /// Text content.
    Text {
        /// Plain text payload.
        text: String,
    },
    /// Image content.
    Image {
        /// Encoded image data or URL.
        data: String,
        /// MIME type for the image payload.
        mime_type: String,
    },
    /// External resource content.
    Resource {
        /// Resource URI.
        uri: String,
        /// Optional text representation.
        text: Option<String>,
    },
}
