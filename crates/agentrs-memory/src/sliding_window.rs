use std::collections::VecDeque;

use async_trait::async_trait;

use agentrs_core::{Memory, Message, Result, Role};

use crate::SearchableMemory;

/// Memory backend that keeps a fixed number of recent non-system messages.
#[derive(Debug, Clone)]
pub struct SlidingWindowMemory {
    system_message: Option<Message>,
    messages: VecDeque<Message>,
    window_size: usize,
}

impl SlidingWindowMemory {
    /// Creates a new sliding-window memory.
    pub fn new(window_size: usize) -> Self {
        Self {
            system_message: None,
            messages: VecDeque::new(),
            window_size,
        }
    }
}

#[async_trait]
impl Memory for SlidingWindowMemory {
    async fn store(&mut self, _key: &str, value: Message) -> Result<()> {
        if matches!(value.role, Role::System) {
            self.system_message = Some(value);
            return Ok(());
        }

        self.messages.push_back(value);
        while self.messages.len() > self.window_size {
            self.messages.pop_front();
        }

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
        let mut history = Vec::with_capacity(self.messages.len() + 1);
        if let Some(system_message) = &self.system_message {
            history.push(system_message.clone());
        }
        history.extend(self.messages.iter().cloned());
        Ok(history)
    }

    async fn clear(&mut self) -> Result<()> {
        self.messages.clear();
        Ok(())
    }
}

#[async_trait]
impl SearchableMemory for SlidingWindowMemory {
    async fn token_count(&self) -> Result<usize> {
        Ok(self
            .messages
            .iter()
            .map(|message| message.text_content().chars().count() / 4)
            .sum())
    }
}
