#![forbid(unsafe_code)]

//! Multi-agent orchestration primitives for `agentrs`.

mod communication;
mod graph;
mod orchestrator;
mod shared_memory;

pub use communication::{EventBus, InMemoryBus, OrchestratorEvent};
pub use graph::{AgentGraph, AgentGraphBuilder, EdgeCondition, GraphEdge};
pub use orchestrator::{MultiAgentOrchestrator, MultiAgentOrchestratorBuilder, RoutingStrategy};
pub use shared_memory::SharedConversation;
