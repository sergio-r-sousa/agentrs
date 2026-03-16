use std::sync::Arc;

use async_trait::async_trait;

use agentrs_core::{Memory, Message, Result, Role};

use crate::SearchableMemory;

/// Counts approximate tokens for a string.
pub trait Tokenizer: Send + Sync + 'static {
    /// Returns the estimated token count.
    fn count(&self, text: &str) -> usize;
}

/// Lightweight tokenizer approximation that avoids external dependencies.
#[derive(Debug, Clone, Copy, Default)]
pub struct ApproximateTokenizer;

impl Tokenizer for ApproximateTokenizer {
    fn count(&self, text: &str) -> usize {
        ((text.chars().count() as f32) / 3.5).ceil() as usize
    }
}

/// Memory backend that trims history to fit a token budget.
pub struct TokenAwareMemory {
    messages: Vec<Message>,
    max_tokens: usize,
    tokenizer: Arc<dyn Tokenizer>,
}

impl TokenAwareMemory {
    /// Creates a token-aware backend with the default tokenizer.
    pub fn new(max_tokens: usize) -> Self {
        Self {
            messages: Vec::new(),
            max_tokens,
            tokenizer: Arc::new(ApproximateTokenizer),
        }
    }

    /// Creates a token-aware backend with a custom tokenizer.
    pub fn with_tokenizer(max_tokens: usize, tokenizer: Arc<dyn Tokenizer>) -> Self {
        Self {
            messages: Vec::new(),
            max_tokens,
            tokenizer,
        }
    }

    fn total_tokens(&self) -> usize {
        self.messages
            .iter()
            .map(|message| self.tokenizer.count(&message.text_content()))
            .sum()
    }

    fn trim_to_budget(&mut self) {
        while self.total_tokens() > self.max_tokens && self.messages.len() > 1 {
            if let Some(index) = self
                .messages
                .iter()
                .position(|message| !matches!(message.role, Role::System))
            {
                self.messages.remove(index);
            } else {
                break;
            }
        }
    }
}

#[async_trait]
impl Memory for TokenAwareMemory {
    async fn store(&mut self, _key: &str, value: Message) -> Result<()> {
        self.messages.push(value);
        self.trim_to_budget();
        Ok(())
    }

    async fn retrieve(&self, query: &str, limit: usize) -> Result<Vec<Message>> {
        let query = query.to_lowercase();
        Ok(self
            .messages
            .iter()
            .filter(|message| message.text_content().to_lowercase().contains(&query))
            .take(limit)
            .cloned()
            .collect())
    }

    async fn history(&self) -> Result<Vec<Message>> {
        Ok(self.messages.clone())
    }

    async fn clear(&mut self) -> Result<()> {
        self.messages.clear();
        Ok(())
    }
}

#[async_trait]
impl SearchableMemory for TokenAwareMemory {
    async fn token_count(&self) -> Result<usize> {
        Ok(self.total_tokens())
    }
}
