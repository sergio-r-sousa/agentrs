use async_trait::async_trait;
use futures::stream::BoxStream;

use agentrs_core::{CompletionRequest, CompletionResponse, LlmProvider, Result, StreamChunk};

use crate::openai::{OpenAiBuilder, OpenAiProvider};

/// Ollama provider backed by its OpenAI-compatible endpoint.
#[derive(Clone)]
pub struct OllamaProvider {
    inner: OpenAiProvider,
}

/// Builder for [`OllamaProvider`].
#[derive(Debug, Clone)]
pub struct OllamaBuilder {
    model: String,
    base_url: String,
}

impl Default for OllamaBuilder {
    fn default() -> Self {
        Self {
            model: "llama3.2".to_string(),
            base_url: "http://localhost:11434/v1".to_string(),
        }
    }
}

impl OllamaProvider {
    /// Creates a default Ollama builder.
    pub fn builder() -> OllamaBuilder {
        OllamaBuilder::default()
    }

    /// Creates an Ollama provider with a model.
    pub fn new(model: impl Into<String>) -> Result<Self> {
        Self::builder().model(model).build()
    }
}

impl OllamaBuilder {
    /// Sets the model name.
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
    pub fn build(self) -> Result<OllamaProvider> {
        let inner = OpenAiBuilder::default()
            .api_key("ollama")
            .base_url(self.base_url)
            .model(self.model)
            .build()?;
        Ok(OllamaProvider { inner })
    }
}

#[async_trait]
impl LlmProvider for OllamaProvider {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        self.inner.complete(request).await
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<BoxStream<'_, Result<StreamChunk>>> {
        self.inner.stream(request).await
    }

    fn name(&self) -> &str {
        "ollama"
    }
}
