use std::{
    collections::HashMap,
    process::Stdio,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    process::{Child, ChildStdin, ChildStdout, Command},
    sync::Mutex,
};

use agentrs_core::{McpError, Result, Tool};

use crate::{
    adapter::McpToolAdapter,
    protocol::{McpCallToolResult, McpMessage, McpTool},
};

const MCP_PROTOCOL_VERSION: &str = "2025-03-26";

/// Configuration for connecting to a web MCP endpoint.
#[derive(Debug, Clone, Default)]
pub struct WebMcpOptions {
    headers: HashMap<String, String>,
    api_key: Option<String>,
    api_key_header: Option<String>,
    api_key_prefix: Option<String>,
}

impl WebMcpOptions {
    /// Creates default web MCP options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds an API key using the default `Authorization: Bearer <token>` header.
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Overrides the header used for the API key.
    pub fn api_key_header(mut self, header_name: impl Into<String>) -> Self {
        self.api_key_header = Some(header_name.into());
        self
    }

    /// Overrides the prefix applied to the API key value.
    pub fn api_key_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.api_key_prefix = Some(prefix.into());
        self
    }

    /// Sets bearer authorization explicitly.
    pub fn bearer_auth(mut self, token: impl Into<String>) -> Self {
        self.api_key = Some(token.into());
        self.api_key_header = Some("Authorization".to_string());
        self.api_key_prefix = Some("Bearer ".to_string());
        self
    }

    /// Adds a custom HTTP header.
    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(name.into(), value.into());
        self
    }

    /// Adds multiple custom HTTP headers.
    pub fn headers(mut self, headers: HashMap<String, String>) -> Self {
        self.headers.extend(headers);
        self
    }

    fn into_headers(mut self) -> HashMap<String, String> {
        if let Some(api_key) = self.api_key {
            let header_name = self
                .api_key_header
                .unwrap_or_else(|| "Authorization".to_string());
            let header_value = match self.api_key_prefix {
                Some(prefix) => format!("{prefix}{api_key}"),
                None if header_name.eq_ignore_ascii_case("authorization") => {
                    format!("Bearer {api_key}")
                }
                None => api_key,
            };
            self.headers.entry(header_name).or_insert(header_value);
        }

        self.headers
    }
}

struct StdioTransport {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

struct HttpTransport {
    client: reqwest::Client,
    endpoint: String,
    headers: HashMap<String, String>,
    session_id: Option<String>,
}

enum McpTransport {
    Stdio(StdioTransport),
    Http(HttpTransport),
}

/// Async MCP client over stdio or Streamable HTTP transport.
pub struct McpClient {
    transport: McpTransport,
    next_id: AtomicU64,
}

impl McpClient {
    /// Spawns an MCP server process and performs the initialize handshake.
    pub async fn spawn(command: &str) -> Result<Self> {
        let parts = command.split_whitespace().collect::<Vec<_>>();
        let Some(program) = parts.first() else {
            return Err(McpError::InvalidCommand.into());
        };

        let mut child = Command::new(program)
            .args(&parts[1..])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|error| McpError::SpawnFailed(error.to_string()))?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| McpError::SpawnFailed("missing child stdin".to_string()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| McpError::SpawnFailed("missing child stdout".to_string()))?;

        let mut client = Self {
            transport: McpTransport::Stdio(StdioTransport {
                child,
                stdin,
                stdout: BufReader::new(stdout),
            }),
            next_id: AtomicU64::new(1),
        };
        client.initialize().await?;
        Ok(client)
    }

    /// Connects to a web MCP endpoint using Streamable HTTP.
    pub async fn connect(endpoint: impl Into<String>) -> Result<Self> {
        Self::connect_with_options(endpoint, WebMcpOptions::default()).await
    }

    /// Connects to a web MCP endpoint using Streamable HTTP with custom headers.
    pub async fn connect_with_headers(
        endpoint: impl Into<String>,
        headers: HashMap<String, String>,
    ) -> Result<Self> {
        Self::connect_with_options(endpoint, WebMcpOptions::new().headers(headers)).await
    }

    /// Connects to a web MCP endpoint using an optional API key.
    pub async fn connect_with_api_key(
        endpoint: impl Into<String>,
        api_key: impl Into<String>,
    ) -> Result<Self> {
        Self::connect_with_options(endpoint, WebMcpOptions::new().api_key(api_key)).await
    }

    /// Connects to a web MCP endpoint using full web options.
    pub async fn connect_with_options(
        endpoint: impl Into<String>,
        options: WebMcpOptions,
    ) -> Result<Self> {
        let mut client = Self {
            transport: McpTransport::Http(HttpTransport {
                client: reqwest::Client::new(),
                endpoint: endpoint.into(),
                headers: options.into_headers(),
                session_id: None,
            }),
            next_id: AtomicU64::new(1),
        };
        client.initialize().await?;
        Ok(client)
    }

    /// Lists all tools exported by the server.
    pub async fn list_tools(&mut self) -> Result<Vec<McpTool>> {
        let response = self
            .call_method("tools/list", serde_json::json!({}))
            .await?;
        let tools = response
            .get("tools")
            .cloned()
            .ok_or_else(|| McpError::Protocol("missing tools field".to_string()))?;
        serde_json::from_value(tools).map_err(Into::into)
    }

    /// Calls a server-side tool.
    pub async fn call_tool(
        &mut self,
        name: &str,
        arguments: serde_json::Value,
    ) -> Result<agentrs_core::ToolOutput> {
        let response = self
            .call_method(
                "tools/call",
                serde_json::json!({
                    "name": name,
                    "arguments": arguments,
                }),
            )
            .await?;

        let result: McpCallToolResult = serde_json::from_value(response)?;
        Ok(result.into_tool_output())
    }

    async fn initialize(&mut self) -> Result<()> {
        let _ = self
            .call_method(
                "initialize",
                serde_json::json!({
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": { "tools": {} },
                    "clientInfo": {
                        "name": "agentrs",
                        "version": env!("CARGO_PKG_VERSION"),
                    }
                }),
            )
            .await?;
        Ok(())
    }

    async fn call_method(
        &mut self,
        method: &str,
        params: serde_json::Value,
    ) -> Result<serde_json::Value> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let message = McpMessage {
            jsonrpc: "2.0".to_string(),
            id: Some(id),
            method: Some(method.to_string()),
            params: Some(params),
            result: None,
            error: None,
        };

        let response = match &mut self.transport {
            McpTransport::Stdio(transport) => call_stdio_method(transport, &message).await?,
            McpTransport::Http(transport) => call_http_method(transport, &message).await?,
        };

        if let Some(error) = response.error {
            return Err(McpError::Response(error.to_string()).into());
        }

        response
            .result
            .ok_or_else(|| McpError::Protocol("missing MCP result".to_string()).into())
    }
}

impl Drop for McpClient {
    fn drop(&mut self) {
        if let McpTransport::Stdio(transport) = &mut self.transport {
            let _ = transport.child.start_kill();
        }
    }
}

/// Spawns an MCP server and converts all exposed tools to `agentrs` tools.
pub async fn spawn_mcp_tools(command: &str) -> Result<Vec<Arc<dyn Tool>>> {
    let mut client = McpClient::spawn(command).await?;
    let tools = client.list_tools().await?;
    let shared_client = Arc::new(Mutex::new(client));

    Ok(tools
        .into_iter()
        .map(|tool| {
            Arc::new(McpToolAdapter::new(Arc::clone(&shared_client), tool)) as Arc<dyn Tool>
        })
        .collect())
}

/// Connects to a web MCP endpoint and converts all exposed tools to `agentrs` tools.
pub async fn connect_mcp_tools(endpoint: &str) -> Result<Vec<Arc<dyn Tool>>> {
    connect_mcp_tools_with_options(endpoint, WebMcpOptions::default()).await
}

/// Connects to a web MCP endpoint with custom headers and converts all exposed tools.
pub async fn connect_mcp_tools_with_headers(
    endpoint: &str,
    headers: HashMap<String, String>,
) -> Result<Vec<Arc<dyn Tool>>> {
    connect_mcp_tools_with_options(endpoint, WebMcpOptions::new().headers(headers)).await
}

/// Connects to a web MCP endpoint using an API key and converts all exposed tools.
pub async fn connect_mcp_tools_with_api_key(
    endpoint: &str,
    api_key: impl Into<String>,
) -> Result<Vec<Arc<dyn Tool>>> {
    connect_mcp_tools_with_options(endpoint, WebMcpOptions::new().api_key(api_key)).await
}

/// Connects to a web MCP endpoint using full web options and converts all exposed tools.
pub async fn connect_mcp_tools_with_options(
    endpoint: &str,
    options: WebMcpOptions,
) -> Result<Vec<Arc<dyn Tool>>> {
    let mut client = McpClient::connect_with_options(endpoint.to_string(), options).await?;
    let tools = client.list_tools().await?;
    let shared_client = Arc::new(Mutex::new(client));

    Ok(tools
        .into_iter()
        .map(|tool| {
            Arc::new(McpToolAdapter::new(Arc::clone(&shared_client), tool)) as Arc<dyn Tool>
        })
        .collect())
}

async fn call_stdio_method(
    transport: &mut StdioTransport,
    message: &McpMessage,
) -> Result<McpMessage> {
    let payload = serde_json::to_vec(message)?;
    transport.stdin.write_all(&payload).await?;
    transport.stdin.write_all(b"\n").await?;
    transport.stdin.flush().await?;

    let mut line = String::new();
    transport.stdout.read_line(&mut line).await?;
    if line.trim().is_empty() {
        return Err(McpError::Protocol("empty MCP response".to_string()).into());
    }

    serde_json::from_str(&line).map_err(Into::into)
}

async fn call_http_method(
    transport: &mut HttpTransport,
    message: &McpMessage,
) -> Result<McpMessage> {
    let mut request = transport
        .client
        .post(&transport.endpoint)
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .header(
            reqwest::header::ACCEPT,
            "application/json, text/event-stream",
        )
        .header("MCP-Protocol-Version", MCP_PROTOCOL_VERSION);

    if let Some(session_id) = &transport.session_id {
        request = request.header("Mcp-Session-Id", session_id.as_str());
    }

    for (name, value) in &transport.headers {
        request = request.header(name.as_str(), value.as_str());
    }

    let response = request
        .json(message)
        .send()
        .await
        .map_err(|error| McpError::Protocol(error.to_string()))?;

    if let Some(session_id) = response
        .headers()
        .get("Mcp-Session-Id")
        .and_then(|value| value.to_str().ok())
    {
        transport.session_id = Some(session_id.to_string());
    }

    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        return Err(McpError::Response(format!("HTTP {}: {}", status.as_u16(), body)).into());
    }

    let content_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default()
        .to_string();

    if content_type.contains("text/event-stream") {
        let body = response
            .text()
            .await
            .map_err(|error| McpError::Protocol(error.to_string()))?;
        parse_sse_response(&body, message.id)
    } else {
        response
            .json::<McpMessage>()
            .await
            .map_err(|error| McpError::Protocol(error.to_string()).into())
    }
}

fn parse_sse_response(payload: &str, expected_id: Option<u64>) -> Result<McpMessage> {
    let mut event_data = String::new();

    for line in payload.lines() {
        if let Some(data) = line.strip_prefix("data:") {
            event_data.push_str(data.trim_start());
            event_data.push('\n');
            continue;
        }

        if line.trim().is_empty() && !event_data.trim().is_empty() {
            if let Some(message) = extract_response_message(&event_data, expected_id)? {
                return Ok(message);
            }
            event_data.clear();
        }
    }

    if let Some(message) = extract_response_message(&event_data, expected_id)? {
        return Ok(message);
    }

    Err(McpError::Protocol("missing JSON-RPC response in SSE stream".to_string()).into())
}

fn extract_response_message(payload: &str, expected_id: Option<u64>) -> Result<Option<McpMessage>> {
    let trimmed = payload.trim();
    if trimmed.is_empty() || trimmed == "[DONE]" {
        return Ok(None);
    }

    let value: serde_json::Value = serde_json::from_str(trimmed)?;
    match value {
        serde_json::Value::Array(items) => {
            for item in items {
                let message: McpMessage = serde_json::from_value(item)?;
                if matches_expected_id(&message, expected_id) {
                    return Ok(Some(message));
                }
            }
            Ok(None)
        }
        other => {
            let message: McpMessage = serde_json::from_value(other)?;
            if matches_expected_id(&message, expected_id) {
                Ok(Some(message))
            } else {
                Ok(None)
            }
        }
    }
}

fn matches_expected_id(message: &McpMessage, expected_id: Option<u64>) -> bool {
    match expected_id {
        Some(id) => message.id == Some(id),
        None => true,
    }
}

#[cfg(test)]
mod tests {
    use super::extract_response_message;

    #[test]
    fn extracts_matching_message_from_single_payload() {
        let message =
            extract_response_message(r#"{"jsonrpc":"2.0","id":7,"result":{"ok":true}}"#, Some(7))
                .expect("payload should parse")
                .expect("response should be found");

        assert_eq!(message.id, Some(7));
        assert_eq!(message.result, Some(serde_json::json!({"ok": true})));
    }

    #[test]
    fn extracts_matching_message_from_batch_payload() {
        let payload = r#"[
            {"jsonrpc":"2.0","method":"notifications/message","params":{"level":"info"}},
            {"jsonrpc":"2.0","id":2,"result":{"tools":[]}}
        ]"#;
        let message = extract_response_message(payload, Some(2))
            .expect("payload should parse")
            .expect("response should be found");

        assert_eq!(message.id, Some(2));
    }

    #[test]
    fn ignores_done_marker() {
        let message = extract_response_message("[DONE]", Some(1)).expect("marker should parse");
        assert!(message.is_none());
    }
}
