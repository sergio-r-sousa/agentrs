use async_trait::async_trait;
use futures::stream::BoxStream;

use agentrs_core::{
    CompletionRequest, CompletionResponse, LlmProvider, Message, ProviderError, Result, Role,
    StopReason, StreamChunk, Usage,
};

/// Gemini provider.
#[derive(Clone)]
pub struct GeminiProvider {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    model: String,
}

/// Builder for [`GeminiProvider`].
#[derive(Debug, Clone)]
pub struct GeminiBuilder {
    api_key: Option<String>,
    base_url: String,
    model: String,
    client: reqwest::Client,
}

impl Default for GeminiBuilder {
    fn default() -> Self {
        Self {
            api_key: None,
            base_url: "https://generativelanguage.googleapis.com/v1beta/models".to_string(),
            model: "gemini-2.0-flash".to_string(),
            client: reqwest::Client::new(),
        }
    }
}

impl GeminiProvider {
    /// Creates a builder from environment variables.
    pub fn from_env() -> GeminiBuilder {
        GeminiBuilder {
            api_key: std::env::var("GEMINI_API_KEY").ok(),
            ..GeminiBuilder::default()
        }
    }

    /// Creates a new builder.
    pub fn builder() -> GeminiBuilder {
        GeminiBuilder::default()
    }
}

impl GeminiBuilder {
    /// Sets the API key.
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Sets the model.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Sets the base URL.
    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Builds the provider.
    pub fn build(self) -> Result<GeminiProvider> {
        Ok(GeminiProvider {
            client: self.client,
            api_key: self.api_key.ok_or(ProviderError::MissingApiKey {
                env_var: "GEMINI_API_KEY",
            })?,
            base_url: self.base_url,
            model: self.model,
        })
    }
}

#[async_trait]
impl LlmProvider for GeminiProvider {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let mut contents = Vec::new();
        if let Some(system) = request.system.as_ref() {
            contents.push(serde_json::json!({
                "role": "user",
                "parts": [{ "text": format!("System instruction: {system}") }]
            }));
        }
        for message in request.messages {
            let role = match message.role {
                Role::Assistant => "model",
                _ => "user",
            };
            contents.push(serde_json::json!({
                "role": role,
                "parts": [{ "text": message.text_content() }]
            }));
        }

        let response = self
            .client
            .post(format!(
                "{}/{}:generateContent?key={}",
                self.base_url.trim_end_matches('/'),
                if request.model.is_empty() {
                    &self.model
                } else {
                    &request.model
                },
                self.api_key
            ))
            .json(&serde_json::json!({
                "contents": contents,
                "generationConfig": {
                    "temperature": request.temperature,
                    "maxOutputTokens": request.max_tokens,
                }
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
        let text = payload
            .get("candidates")
            .and_then(serde_json::Value::as_array)
            .and_then(|candidates| candidates.first())
            .and_then(|candidate| candidate.get("content"))
            .and_then(|content| content.get("parts"))
            .and_then(serde_json::Value::as_array)
            .map(|parts| {
                parts
                    .iter()
                    .filter_map(|part| part.get("text").and_then(serde_json::Value::as_str))
                    .collect::<String>()
            })
            .unwrap_or_default();

        Ok(CompletionResponse {
            message: Message::assistant(text),
            stop_reason: StopReason::Stop,
            usage: Usage::default(),
            model: request.model,
            raw: Some(payload),
        })
    }

    async fn stream(
        &self,
        _request: CompletionRequest,
    ) -> Result<BoxStream<'_, Result<StreamChunk>>> {
        Err(ProviderError::Unsupported(
            "Gemini streaming is not yet implemented in this SDK scaffold",
        )
        .into())
    }

    fn name(&self) -> &str {
        "gemini"
    }
}
