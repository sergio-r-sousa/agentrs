use async_stream::try_stream;
use async_trait::async_trait;
use futures::{stream::BoxStream, StreamExt};

use agentrs_core::{
    streaming::{map_stop_reason, parse_sse_chunk},
    CompletionRequest, CompletionResponse, LlmProvider, Message, ProviderError, Result, Role,
    StopReason, StreamChunk, ToolCall, ToolDefinition, Usage,
};

/// OpenAI-compatible provider.
#[derive(Clone)]
pub struct OpenAiProvider {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    model: String,
}

/// Builder for [`OpenAiProvider`].
#[derive(Debug, Clone)]
pub struct OpenAiBuilder {
    api_key: Option<String>,
    base_url: String,
    model: String,
    client: reqwest::Client,
}

impl Default for OpenAiBuilder {
    fn default() -> Self {
        Self {
            api_key: None,
            base_url: "https://api.openai.com/v1".to_string(),
            model: "gpt-4o-mini".to_string(),
            client: reqwest::Client::new(),
        }
    }
}

impl OpenAiProvider {
    /// Creates a builder seeded from environment variables.
    pub fn from_env() -> OpenAiBuilder {
        OpenAiBuilder {
            api_key: std::env::var("OPENAI_API_KEY").ok(),
            base_url: std::env::var("OPENAI_BASE_URL")
                .unwrap_or_else(|_| "https://api.openai.com/v1".to_string()),
            model: std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_string()),
            ..OpenAiBuilder::default()
        }
    }

    /// Creates an empty builder.
    pub fn builder() -> OpenAiBuilder {
        OpenAiBuilder::default()
    }

    fn request_body(&self, request: &CompletionRequest, stream: bool) -> serde_json::Value {
        let messages = request
            .messages
            .iter()
            .map(map_openai_message)
            .collect::<Vec<_>>();
        let tools = request
            .tools
            .as_ref()
            .map(|tools| tools.iter().map(map_openai_tool).collect::<Vec<_>>());

        let mut body = serde_json::json!({
            "model": request.model.clone().if_empty_then(|| self.model.clone()),
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": stream,
        });
        if let Some(tools) = tools {
            body["tools"] = serde_json::Value::Array(tools);
        }
        if let Some(system) = &request.system {
            if let Some(messages) = body
                .get_mut("messages")
                .and_then(serde_json::Value::as_array_mut)
            {
                messages.insert(
                    0,
                    serde_json::json!({ "role": "system", "content": system }),
                );
            }
        }
        if let Some(extra) = serde_json::to_value(&request.extra)
            .ok()
            .and_then(|value| value.as_object().cloned())
        {
            if let Some(map) = body.as_object_mut() {
                map.extend(extra);
            }
        }
        body
    }

    async fn send(&self, request: &CompletionRequest, stream: bool) -> Result<reqwest::Response> {
        let response = self
            .client
            .post(format!(
                "{}/chat/completions",
                self.base_url.trim_end_matches('/')
            ))
            .bearer_auth(&self.api_key)
            .json(&self.request_body(request, stream))
            .send()
            .await
            .map_err(|error| ProviderError::Http(error.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let message = response.text().await.unwrap_or_default();
            return Err(ProviderError::Api {
                status: status.as_u16(),
                message,
            }
            .into());
        }
        Ok(response)
    }
}

impl OpenAiBuilder {
    /// Overrides the API key.
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Overrides the base URL.
    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Sets the default model.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Reuses an existing HTTP client.
    pub fn client(mut self, client: reqwest::Client) -> Self {
        self.client = client;
        self
    }

    /// Builds the provider.
    pub fn build(self) -> Result<OpenAiProvider> {
        let api_key = self.api_key.ok_or(ProviderError::MissingApiKey {
            env_var: "OPENAI_API_KEY",
        })?;
        Ok(OpenAiProvider {
            client: self.client,
            api_key,
            base_url: self.base_url,
            model: self.model,
        })
    }
}

#[async_trait]
impl LlmProvider for OpenAiProvider {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let response = self.send(&request, false).await?;
        let payload = response
            .json::<serde_json::Value>()
            .await
            .map_err(|error| ProviderError::Http(error.to_string()))?;
        map_openai_response(payload)
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<BoxStream<'_, Result<StreamChunk>>> {
        let response = self.send(&request, true).await?;
        let stream = response.bytes_stream();
        Ok(try_stream! {
            futures::pin_mut!(stream);
            while let Some(chunk) = stream.next().await {
                let chunk = chunk.map_err(|error| ProviderError::Http(error.to_string()))?;
                if let Some(parsed) = parse_sse_chunk(chunk)? {
                    yield parsed;
                }
            }
        }
        .boxed())
    }

    fn name(&self) -> &str {
        "openai"
    }
}

fn map_openai_message(message: &Message) -> serde_json::Value {
    let role = match message.role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    };

    let mut value = serde_json::json!({
        "role": role,
        "content": message.text_content(),
    });

    if let Some(tool_calls) = &message.tool_calls {
        value["tool_calls"] =
            serde_json::Value::Array(tool_calls.iter().map(map_openai_tool_call).collect());
    }
    if let Some(tool_call_id) = &message.tool_call_id {
        value["tool_call_id"] = serde_json::Value::String(tool_call_id.clone());
    }
    value
}

fn map_openai_tool(tool: &ToolDefinition) -> serde_json::Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.schema,
        }
    })
}

fn map_openai_tool_call(tool_call: &ToolCall) -> serde_json::Value {
    serde_json::json!({
        "id": tool_call.id,
        "type": "function",
        "function": {
            "name": tool_call.name,
            "arguments": tool_call.arguments.to_string(),
        }
    })
}

fn map_openai_response(payload: serde_json::Value) -> Result<CompletionResponse> {
    let choice = payload
        .get("choices")
        .and_then(serde_json::Value::as_array)
        .and_then(|choices| choices.first())
        .cloned()
        .ok_or_else(|| ProviderError::InvalidResponse("missing choices".to_string()))?;

    let message_value = choice
        .get("message")
        .cloned()
        .ok_or_else(|| ProviderError::InvalidResponse("missing message".to_string()))?;
    let stop_reason = choice
        .get("finish_reason")
        .and_then(serde_json::Value::as_str)
        .map(map_stop_reason)
        .unwrap_or(StopReason::Stop);

    let tool_calls = message_value
        .get("tool_calls")
        .and_then(serde_json::Value::as_array)
        .map(|calls| {
            calls
                .iter()
                .filter_map(|call| {
                    let id = call.get("id")?.as_str()?.to_string();
                    let function = call.get("function")?;
                    let name = function.get("name")?.as_str()?.to_string();
                    let arguments = function
                        .get("arguments")
                        .and_then(serde_json::Value::as_str)
                        .map(|value| serde_json::from_str(value).unwrap_or(serde_json::json!({})))
                        .unwrap_or_else(|| serde_json::json!({}));
                    Some(ToolCall {
                        id,
                        name,
                        arguments,
                    })
                })
                .collect::<Vec<_>>()
        })
        .filter(|calls| !calls.is_empty());

    let mut message = Message::assistant(
        message_value
            .get("content")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default(),
    );
    message.tool_calls = tool_calls;

    let usage = payload
        .get("usage")
        .cloned()
        .map(parse_usage)
        .unwrap_or_default();
    let model = payload
        .get("model")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .to_string();

    Ok(CompletionResponse {
        message,
        stop_reason,
        usage,
        model,
        raw: Some(payload),
    })
}

fn parse_usage(value: serde_json::Value) -> Usage {
    Usage {
        input_tokens: value
            .get("prompt_tokens")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or_default() as u32,
        output_tokens: value
            .get("completion_tokens")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or_default() as u32,
        total_tokens: value
            .get("total_tokens")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or_default() as u32,
    }
}

trait IfEmptyThen {
    fn if_empty_then(self, fallback: impl FnOnce() -> String) -> String;
}

impl IfEmptyThen for String {
    fn if_empty_then(self, fallback: impl FnOnce() -> String) -> String {
        if self.is_empty() {
            fallback()
        } else {
            self
        }
    }
}
