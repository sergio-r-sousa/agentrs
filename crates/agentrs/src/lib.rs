#![forbid(unsafe_code)]

//! Facade crate for the `agentrs` SDK.

pub use agentrs_agents as agents;
pub use agentrs_core as core;
pub use agentrs_llm as llm;
#[cfg(feature = "mcp")]
pub use agentrs_mcp as mcp;
pub use agentrs_memory as memory;
pub use agentrs_multi as multi;
pub use agentrs_tools as tools;

pub mod config;
pub mod prelude;
