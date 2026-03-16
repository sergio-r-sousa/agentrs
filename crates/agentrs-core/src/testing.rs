//! Test utilities for SDK users and internal crates.

use std::{collections::VecDeque, sync::Arc};

use async_trait::async_trait;
use futures::{stream, StreamExt};
use tokio::sync::Mutex;

use crate::{
    CompletionRequest, CompletionResponse, LlmProvider, ProviderError, Result, StreamChunk,
};

/// Mock provider for deterministic tests.
#[derive(Clone, Default)]
pub struct MockLlmProvider {
    name: String,
    responses: Arc<Mutex<VecDeque<CompletionResponse>>>,
}

impl MockLlmProvider {
    /// Creates an empty mock provider.
    pub fn new() -> Self {
        Self {
            name: "mock".to_string(),
            responses: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    /// Creates a mock provider seeded with responses.
    pub fn with_responses(responses: Vec<CompletionResponse>) -> Self {
        Self {
            name: "mock".to_string(),
            responses: Arc::new(Mutex::new(VecDeque::from(responses))),
        }
    }

    /// Creates a mock provider from plain text responses.
    pub fn with_text_responses(texts: Vec<impl Into<String>>) -> Self {
        Self::with_responses(texts.into_iter().map(CompletionResponse::text).collect())
    }

    /// Creates a mock provider that first asks for a tool, then returns a final answer.
    pub fn with_tool_call_sequence(
        tool_name: impl Into<String>,
        arguments: serde_json::Value,
        final_text: impl Into<String>,
    ) -> Self {
        Self::with_responses(vec![
            CompletionResponse::tool_call(tool_name, arguments),
            CompletionResponse::text(final_text),
        ])
    }

    /// Queues another response.
    pub async fn push_response(&self, response: CompletionResponse) {
        self.responses.lock().await.push_back(response);
    }

    async fn pop_response(&self) -> Result<CompletionResponse> {
        self.responses.lock().await.pop_front().ok_or_else(|| {
            ProviderError::InvalidResponse("mock provider exhausted".to_string()).into()
        })
    }
}

#[async_trait]
impl LlmProvider for MockLlmProvider {
    async fn complete(&self, _req: CompletionRequest) -> Result<CompletionResponse> {
        self.pop_response().await
    }

    async fn stream(
        &self,
        _req: CompletionRequest,
    ) -> Result<futures::stream::BoxStream<'_, Result<StreamChunk>>> {
        let response = self.pop_response().await?;
        let text = response.message.text_content();
        let chunks = text
            .split_whitespace()
            .map(|token| {
                Ok(StreamChunk {
                    delta: format!("{token} "),
                    tool_call_delta: None,
                    finish_reason: None,
                })
            })
            .chain(std::iter::once(Ok(StreamChunk {
                delta: String::new(),
                tool_call_delta: None,
                finish_reason: Some(response.stop_reason),
            })))
            .collect::<Vec<_>>();

        Ok(stream::iter(chunks).boxed())
    }

    fn name(&self) -> &str {
        &self.name
    }
}
