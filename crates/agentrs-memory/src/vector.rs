use std::{cmp::Ordering, sync::Arc};

use async_trait::async_trait;
use tokio::sync::RwLock;

use agentrs_core::{Memory, Message, Result};

use crate::{InMemoryMemory, SearchableMemory};

/// Search result returned by a vector store.
#[derive(Debug, Clone)]
pub struct VectorSearchResult {
    /// Similarity score.
    pub score: f32,
    /// Stored payload.
    pub payload: Message,
}

/// Computes embeddings for messages.
#[async_trait]
pub trait Embedder: Send + Sync + 'static {
    /// Generates an embedding vector.
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;
}

/// Persists and searches embedding vectors.
#[async_trait]
pub trait VectorStore: Send + Sync + 'static {
    /// Upserts a vector and payload.
    async fn upsert(&self, id: String, vector: Vec<f32>, payload: Message) -> Result<()>;

    /// Searches the store.
    async fn search(&self, query: Vec<f32>, limit: usize) -> Result<Vec<VectorSearchResult>>;

    /// Clears all stored vectors.
    async fn clear(&self) -> Result<()>;
}

/// Small deterministic embedder useful for tests and local demos.
#[derive(Debug, Clone, Copy, Default)]
pub struct SimpleEmbedder;

#[async_trait]
impl Embedder for SimpleEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let mut buckets = vec![0.0_f32; 16];
        for (index, byte) in text.bytes().enumerate() {
            buckets[index % 16] += f32::from(byte) / 255.0;
        }
        Ok(buckets)
    }
}

/// In-memory vector store with cosine similarity search.
#[derive(Default)]
pub struct InMemoryVectorStore {
    items: RwLock<Vec<(String, Vec<f32>, Message)>>,
}

impl InMemoryVectorStore {
    /// Creates an empty store.
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl VectorStore for InMemoryVectorStore {
    async fn upsert(&self, id: String, vector: Vec<f32>, payload: Message) -> Result<()> {
        let mut items = self.items.write().await;
        if let Some(existing) = items.iter_mut().find(|item| item.0 == id) {
            *existing = (id, vector, payload);
        } else {
            items.push((id, vector, payload));
        }
        Ok(())
    }

    async fn search(&self, query: Vec<f32>, limit: usize) -> Result<Vec<VectorSearchResult>> {
        let items = self.items.read().await;
        let mut scored = items
            .iter()
            .map(|(_, vector, payload)| VectorSearchResult {
                score: cosine_similarity(vector, &query),
                payload: payload.clone(),
            })
            .collect::<Vec<_>>();

        scored.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(Ordering::Equal)
        });
        scored.truncate(limit);
        Ok(scored)
    }

    async fn clear(&self) -> Result<()> {
        self.items.write().await.clear();
        Ok(())
    }
}

/// Memory backend that combines recent history with semantic retrieval.
pub struct VectorMemory<E = SimpleEmbedder, S = InMemoryVectorStore> {
    embedder: Arc<E>,
    store: Arc<S>,
    recent: InMemoryMemory,
}

impl VectorMemory<SimpleEmbedder, InMemoryVectorStore> {
    /// Creates a vector memory with built-in components.
    pub fn new() -> Self {
        Self {
            embedder: Arc::new(SimpleEmbedder),
            store: Arc::new(InMemoryVectorStore::new()),
            recent: InMemoryMemory::new(),
        }
    }
}

impl<E, S> VectorMemory<E, S>
where
    E: Embedder,
    S: VectorStore,
{
    /// Creates a vector memory with custom embedder and store.
    pub fn with_components(embedder: Arc<E>, store: Arc<S>) -> Self {
        Self {
            embedder,
            store,
            recent: InMemoryMemory::new(),
        }
    }
}

#[async_trait]
impl<E, S> Memory for VectorMemory<E, S>
where
    E: Embedder,
    S: VectorStore,
{
    async fn store(&mut self, key: &str, value: Message) -> Result<()> {
        let vector = self.embedder.embed(&value.text_content()).await?;
        self.store
            .upsert(
                format!("{key}-{}", uuid::Uuid::new_v4()),
                vector,
                value.clone(),
            )
            .await?;
        self.recent.store(key, value).await
    }

    async fn retrieve(&self, query: &str, limit: usize) -> Result<Vec<Message>> {
        let vector = self.embedder.embed(query).await?;
        Ok(self
            .store
            .search(vector, limit)
            .await?
            .into_iter()
            .map(|result| result.payload)
            .collect())
    }

    async fn history(&self) -> Result<Vec<Message>> {
        self.recent.history().await
    }

    async fn clear(&mut self) -> Result<()> {
        self.store.clear().await?;
        self.recent.clear().await
    }
}

#[async_trait]
impl<E, S> SearchableMemory for VectorMemory<E, S>
where
    E: Embedder,
    S: VectorStore,
{
    async fn token_count(&self) -> Result<usize> {
        Ok(self
            .recent
            .history()
            .await?
            .into_iter()
            .map(|message| message.text_content().chars().count() / 4)
            .sum())
    }

    async fn search(&self, query: &str, limit: usize) -> Result<Vec<Message>> {
        self.retrieve(query, limit).await
    }
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    if left.len() != right.len() || left.is_empty() {
        return 0.0;
    }

    let dot = left.iter().zip(right).map(|(l, r)| l * r).sum::<f32>();
    let left_norm = left.iter().map(|value| value * value).sum::<f32>().sqrt();
    let right_norm = right.iter().map(|value| value * value).sum::<f32>().sqrt();

    if left_norm == 0.0 || right_norm == 0.0 {
        0.0
    } else {
        dot / (left_norm * right_norm)
    }
}
