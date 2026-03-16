use async_trait::async_trait;

use agentrs_core::{Memory, Message, Result};

use crate::SearchableMemory;

/// Default in-process memory backend.
#[derive(Debug, Clone, Default)]
pub struct InMemoryMemory {
    messages: Vec<Message>,
    max_messages: Option<usize>,
}

impl InMemoryMemory {
    /// Creates an empty memory backend.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates an in-memory backend capped to the most recent messages.
    pub fn with_max_messages(max_messages: usize) -> Self {
        Self {
            messages: Vec::new(),
            max_messages: Some(max_messages),
        }
    }

    fn trim(&mut self) {
        let Some(max_messages) = self.max_messages else {
            return;
        };

        if self.messages.len() <= max_messages {
            return;
        }

        let overflow = self.messages.len() - max_messages;
        self.messages.drain(0..overflow);
    }
}

#[async_trait]
impl Memory for InMemoryMemory {
    async fn store(&mut self, _key: &str, value: Message) -> Result<()> {
        self.messages.push(value);
        self.trim();
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
impl SearchableMemory for InMemoryMemory {
    async fn token_count(&self) -> Result<usize> {
        Ok(self
            .messages
            .iter()
            .map(|message| message.text_content().chars().count() / 4)
            .sum())
    }
}
