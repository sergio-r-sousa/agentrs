use async_trait::async_trait;
use futures::{stream::BoxStream, StreamExt};
use tokio::sync::broadcast;
use tokio_stream::wrappers::BroadcastStream;

use agentrs_core::{AgentOutput, Result};

/// Events emitted during orchestration.
#[derive(Debug, Clone)]
pub enum OrchestratorEvent {
    /// An agent completed execution.
    AgentCompleted {
        /// Name of the completed agent.
        agent: String,
        /// Output produced by the agent.
        output: AgentOutput,
    },
}

/// Pluggable event bus for orchestration observability.
#[async_trait]
pub trait EventBus: Send + Sync {
    /// Publishes an event.
    async fn publish(&self, event: OrchestratorEvent) -> Result<()>;

    /// Subscribes to future events.
    async fn subscribe(&self) -> Result<BoxStream<'static, OrchestratorEvent>>;
}

/// In-memory broadcast event bus.
#[derive(Clone)]
pub struct InMemoryBus {
    sender: broadcast::Sender<OrchestratorEvent>,
}

impl InMemoryBus {
    /// Creates a bus with the requested capacity.
    pub fn new(capacity: usize) -> Self {
        let (sender, _) = broadcast::channel(capacity);
        Self { sender }
    }
}

#[async_trait]
impl EventBus for InMemoryBus {
    async fn publish(&self, event: OrchestratorEvent) -> Result<()> {
        let _ = self.sender.send(event);
        Ok(())
    }

    async fn subscribe(&self) -> Result<BoxStream<'static, OrchestratorEvent>> {
        Ok(BroadcastStream::new(self.sender.subscribe())
            .filter_map(|result| async move { result.ok() })
            .boxed())
    }
}
