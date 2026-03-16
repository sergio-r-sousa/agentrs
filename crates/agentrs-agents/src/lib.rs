#![forbid(unsafe_code)]

//! Agent builders and execution loops for `agentrs`.

mod builder;
mod runner;

pub use builder::{Agent, AgentBuilder, NoLlm, WithLlm};
pub use runner::{AgentConfig, AgentRunner, LoopStrategy};
