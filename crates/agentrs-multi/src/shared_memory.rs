use std::sync::Arc;

use tokio::sync::RwLock;

use agentrs_core::{Message, Result};

/// Shared conversation state used by multiple agents.
#[derive(Clone, Default)]
pub struct SharedConversation {
    messages: Arc<RwLock<Vec<Message>>>,
}

impl SharedConversation {
    /// Creates a new shared conversation.
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a message tagged with its source agent.
    pub async fn add(&self, agent_name: &str, mut message: Message) -> Result<()> {
        message.metadata.insert(
            "agent".to_string(),
            serde_json::Value::String(agent_name.to_string()),
        );
        self.messages.write().await.push(message);
        Ok(())
    }

    /// Returns the full shared conversation.
    pub async fn get_all(&self) -> Vec<Message> {
        self.messages.read().await.clone()
    }
}
