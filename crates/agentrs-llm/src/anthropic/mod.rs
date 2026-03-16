use async_trait::async_trait;
use futures::stream::BoxStream;

use agentrs_core::{
    CompletionRequest, CompletionResponse, LlmProvider, ProviderError, Result, StreamChunk,
};

/// Anthropic provider.
#[derive(Clone)]
pub struct AnthropicProvider {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    model: String,
}

/// Builder for [`AnthropicProvider`].
#[derive(Debug, Clone)]
pub struct AnthropicBuilder {
    api_key: Option<String>,
    base_url: String,
    model: String,
    client: reqwest::Client,
}

impl Default for AnthropicBuilder {
    fn default() -> Self {
        Self {
            api_key: None,
            base_url: "https://api.anthropic.com/v1".to_string(),
            model: "claude-3-5-sonnet-latest".to_string(),
            client: reqwest::Client::new(),
        }
    }
}

impl AnthropicProvider {
    /// Creates a builder seeded from environment variables.
    pub fn from_env() -> AnthropicBuilder {
        AnthropicBuilder {
            api_key: std::env::var("ANTHROPIC_API_KEY").ok(),
            ..AnthropicBuilder::default()
        }
    }

    /// Creates a fresh builder.
    pub fn builder() -> AnthropicBuilder {
        AnthropicBuilder::default()
    }
}

impl AnthropicBuilder {
    /// Sets the API key.
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Sets the base URL.
    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Sets the default model.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Builds the provider.
    pub fn build(self) -> Result<AnthropicProvider> {
        Ok(AnthropicProvider {
            client: self.client,
            api_key: self.api_key.ok_or(ProviderError::MissingApiKey {
                env_var: "ANTHROPIC_API_KEY",
            })?,
            base_url: self.base_url,
            model: self.model,
        })
    }
}

#[async_trait]
impl LlmProvider for AnthropicProvider {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let mut messages = Vec::new();
        let mut system = request.system.clone().unwrap_or_default();
        for message in request.messages {
            if matches!(message.role, agentrs_core::Role::System) {
                if !system.is_empty() {
                    system.push('\n');
                }
                system.push_str(&message.text_content());
                continue;
            }
            messages.push(serde_json::json!({
                "role": match message.role {
                    agentrs_core::Role::User => "user",
                    agentrs_core::Role::Assistant => "assistant",
                    agentrs_core::Role::Tool => "user",
                    agentrs_core::Role::System => "user",
                },
                "content": [{ "type": "text", "text": message.text_content() }]
            }));
        }
        let tools = request.tools.map(|tools| {
            tools
                .into_iter()
                .map(|tool| {
                    serde_json::json!({
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.schema,
                    })
                })
                .collect::<Vec<_>>()
        });

        let response = self
            .client
            .post(format!("{}/messages", self.base_url.trim_end_matches('/')))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&serde_json::json!({
                "model": if request.model.is_empty() { self.model.clone() } else { request.model },
                "messages": messages,
                "system": system,
                "max_tokens": request.max_tokens.unwrap_or(2048),
                "temperature": request.temperature,
                "tools": tools,
            }))
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

        let payload = response
            .json::<serde_json::Value>()
            .await
            .map_err(|error| ProviderError::Http(error.to_string()))?;
        let mut text = String::new();
        let mut tool_calls = Vec::new();
        if let Some(content) = payload.get("content").and_then(serde_json::Value::as_array) {
            for item in content {
                match item.get("type").and_then(serde_json::Value::as_str) {
                    Some("text") => {
                        if let Some(chunk) = item.get("text").and_then(serde_json::Value::as_str) {
                            text.push_str(chunk);
                        }
                    }
                    Some("tool_use") => {
                        tool_calls.push(agentrs_core::ToolCall {
                            id: item
                                .get("id")
                                .and_then(serde_json::Value::as_str)
                                .unwrap_or_default()
                                .to_string(),
                            name: item
                                .get("name")
                                .and_then(serde_json::Value::as_str)
                                .unwrap_or_default()
                                .to_string(),
                            arguments: item
                                .get("input")
                                .cloned()
                                .unwrap_or_else(|| serde_json::json!({})),
                        });
                    }
                    _ => {}
                }
            }
        }
        let mut message = agentrs_core::Message::assistant(text);
        if !tool_calls.is_empty() {
            message.tool_calls = Some(tool_calls);
        }

        Ok(CompletionResponse {
            stop_reason: payload
                .get("stop_reason")
                .and_then(serde_json::Value::as_str)
                .map(agentrs_core::streaming::map_stop_reason)
                .unwrap_or(agentrs_core::StopReason::Stop),
            usage: agentrs_core::Usage {
                input_tokens: payload
                    .get("usage")
                    .and_then(|usage| usage.get("input_tokens"))
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or_default() as u32,
                output_tokens: payload
                    .get("usage")
                    .and_then(|usage| usage.get("output_tokens"))
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or_default() as u32,
                total_tokens: payload
                    .get("usage")
                    .and_then(|usage| usage.get("input_tokens"))
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or_default() as u32
                    + payload
                        .get("usage")
                        .and_then(|usage| usage.get("output_tokens"))
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or_default() as u32,
            },
            model: payload
                .get("model")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .to_string(),
            raw: Some(payload),
            message,
        })
    }

    async fn stream(
        &self,
        _request: CompletionRequest,
    ) -> Result<BoxStream<'_, Result<StreamChunk>>> {
        Err(ProviderError::Unsupported(
            "Anthropic streaming is not yet implemented in this SDK scaffold",
        )
        .into())
    }

    fn name(&self) -> &str {
        "anthropic"
    }
}
