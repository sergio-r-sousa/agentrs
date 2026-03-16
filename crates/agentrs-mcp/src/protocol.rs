use serde::{Deserialize, Serialize};

use agentrs_core::ToolOutput;

/// MCP JSON-RPC message envelope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpMessage {
    /// Protocol version.
    pub jsonrpc: String,
    /// Optional request id.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<u64>,
    /// Optional RPC method.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub method: Option<String>,
    /// Optional params.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Value>,
    /// Optional result.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    /// Optional error.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<serde_json::Value>,
}

/// Tool metadata returned by MCP servers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpTool {
    /// Public tool name.
    pub name: String,
    /// Tool description.
    #[serde(default)]
    pub description: String,
    /// Input schema.
    #[serde(default, alias = "inputSchema")]
    pub input_schema: serde_json::Value,
}

/// Tool call result payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpCallToolResult {
    /// Structured output content.
    pub content: Vec<serde_json::Value>,
    /// Whether the result is an error.
    #[serde(default)]
    pub is_error: bool,
}

impl McpCallToolResult {
    /// Converts the MCP result to the SDK tool output.
    pub fn into_tool_output(self) -> ToolOutput {
        let text = self
            .content
            .iter()
            .filter_map(|item| item.get("text").and_then(serde_json::Value::as_str))
            .collect::<Vec<_>>()
            .join("\n");
        let mut output = ToolOutput::text(text);
        output.is_error = self.is_error;
        output
    }
}
