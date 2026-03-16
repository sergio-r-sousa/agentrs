#![cfg(feature = "redis")]

use std::sync::Arc;

use async_trait::async_trait;
use redis::AsyncCommands;
use tokio::sync::Mutex;

use agentrs_core::{Memory, MemoryError, Message, Result};

use crate::SearchableMemory;

/// Redis-backed persistent conversation memory.
pub struct RedisMemory {
    connection: Arc<Mutex<redis::aio::ConnectionManager>>,
    session_id: String,
    ttl_seconds: u64,
}

impl RedisMemory {
    /// Connects to Redis and creates a session-scoped memory backend.
    pub async fn new(redis_url: &str, session_id: impl Into<String>) -> Result<Self> {
        let client = redis::Client::open(redis_url)
            .map_err(|error| MemoryError::Backend(error.to_string()))?;
        let connection = redis::aio::ConnectionManager::new(client)
            .await
            .map_err(|error| MemoryError::Backend(error.to_string()))?;

        Ok(Self {
            connection: Arc::new(Mutex::new(connection)),
            session_id: session_id.into(),
            ttl_seconds: 3600,
        })
    }

    fn key(&self) -> String {
        format!("agentrs:session:{}", self.session_id)
    }
}

#[async_trait]
impl Memory for RedisMemory {
    async fn store(&mut self, _key: &str, value: Message) -> Result<()> {
        let payload = serde_json::to_string(&value)?;
        let key = self.key();
        let mut connection = self.connection.lock().await;

        connection
            .rpush::<_, _, ()>(&key, payload)
            .await
            .map_err(|error| MemoryError::Backend(error.to_string()))?;
        connection
            .expire::<_, ()>(&key, self.ttl_seconds as i64)
            .await
            .map_err(|error| MemoryError::Backend(error.to_string()))?;
        Ok(())
    }

    async fn retrieve(&self, query: &str, limit: usize) -> Result<Vec<Message>> {
        let history = self.history().await?;
        let query = query.to_lowercase();
        Ok(history
            .into_iter()
            .filter(|message| message.text_content().to_lowercase().contains(&query))
            .take(limit)
            .collect())
    }

    async fn history(&self) -> Result<Vec<Message>> {
        let key = self.key();
        let mut connection = self.connection.lock().await;
        let payloads = connection
            .lrange::<_, Vec<String>>(&key, 0, -1)
            .await
            .map_err(|error| MemoryError::Backend(error.to_string()))?;

        payloads
            .into_iter()
            .map(|payload| serde_json::from_str(&payload).map_err(Into::into))
            .collect()
    }

    async fn clear(&mut self) -> Result<()> {
        let key = self.key();
        let mut connection = self.connection.lock().await;
        connection
            .del::<_, ()>(&key)
            .await
            .map_err(|error| MemoryError::Backend(error.to_string()))?;
        Ok(())
    }
}

#[async_trait]
impl SearchableMemory for RedisMemory {
    async fn token_count(&self) -> Result<usize> {
        Ok(self
            .history()
            .await?
            .into_iter()
            .map(|message| message.text_content().chars().count() / 4)
            .sum())
    }
}
