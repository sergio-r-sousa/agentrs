#![forbid(unsafe_code)]

//! Memory backends for `agentrs`.

mod in_memory;
#[cfg(feature = "redis")]
mod redis_backend;
mod sliding_window;
mod token_aware;
mod vector;

use async_trait::async_trait;

pub use in_memory::InMemoryMemory;
#[cfg(feature = "redis")]
pub use redis_backend::RedisMemory;
pub use sliding_window::SlidingWindowMemory;
pub use token_aware::{ApproximateTokenizer, TokenAwareMemory, Tokenizer};
pub use vector::{
    Embedder, InMemoryVectorStore, SimpleEmbedder, VectorMemory, VectorSearchResult, VectorStore,
};

use agentrs_core::{Memory, Message, Result};

/// Shared extension trait for memory utilities.
#[async_trait]
pub trait SearchableMemory: Memory {
    /// Returns the approximate token count for the current history.
    async fn token_count(&self) -> Result<usize>;

    /// Semantic or lexical search over memory.
    async fn search(&self, query: &str, limit: usize) -> Result<Vec<Message>> {
        self.retrieve(query, limit).await
    }
}
