#![forbid(unsafe_code)]

//! LLM providers and provider registry for `agentrs`.

mod registry;

/// Anthropic provider implementations.
#[cfg(feature = "anthropic")]
pub mod anthropic;
/// Azure OpenAI provider implementations.
#[cfg(feature = "azureopenai")]
pub mod azureopenai;
/// Gemini provider implementations.
#[cfg(feature = "gemini")]
pub mod gemini;
/// Ollama provider implementations.
#[cfg(feature = "ollama")]
pub mod ollama;
/// OpenAI-compatible provider implementations.
#[cfg(feature = "openai")]
pub mod openai;

pub use registry::ProviderRegistry;

#[cfg(feature = "anthropic")]
pub use anthropic::{AnthropicBuilder, AnthropicProvider};
#[cfg(feature = "azureopenai")]
pub use azureopenai::{AzureOpenAiBuilder, AzureOpenAiProvider};
#[cfg(feature = "gemini")]
pub use gemini::{GeminiBuilder, GeminiProvider};
#[cfg(feature = "ollama")]
pub use ollama::{OllamaBuilder, OllamaProvider};
#[cfg(feature = "openai")]
pub use openai::{OpenAiBuilder, OpenAiProvider};
