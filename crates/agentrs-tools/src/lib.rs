#![forbid(unsafe_code)]

//! Tool registry, built-in tools, and tool macros for `agentrs`.

mod builtin;
mod registry;

use std::collections::HashMap;

pub use agentrs_macros::tool;
#[cfg(feature = "bash")]
pub use builtin::BashTool;
#[cfg(feature = "python")]
pub use builtin::PythonTool;
pub use builtin::{CalculatorTool, FileReadTool, FileWriteTool, WebFetchTool, WebSearchTool};
pub use registry::ToolRegistry;

pub use agentrs_core::{Tool, ToolContent, ToolOutput};

/// Internal re-exports used by the proc macros.
pub mod __private {
    pub use async_trait;
    pub use schemars;
    pub use serde_json;
}

/// Shared contextual utilities available to tools built with the macro API.
#[derive(Debug, Clone)]
pub struct ToolContext {
    /// Shared HTTP client.
    pub http: reqwest::Client,
    /// Ad-hoc metadata bag.
    pub metadata: HashMap<String, serde_json::Value>,
}

impl Default for ToolContext {
    fn default() -> Self {
        Self {
            http: reqwest::Client::new(),
            metadata: HashMap::new(),
        }
    }
}

/// Converts common Rust values into [`ToolOutput`].
pub trait IntoToolOutput {
    /// Converts `self` into a structured tool output.
    fn into_tool_output(self) -> ToolOutput;
}

impl IntoToolOutput for ToolOutput {
    fn into_tool_output(self) -> ToolOutput {
        self
    }
}

impl IntoToolOutput for String {
    fn into_tool_output(self) -> ToolOutput {
        ToolOutput::text(self)
    }
}

impl IntoToolOutput for &str {
    fn into_tool_output(self) -> ToolOutput {
        ToolOutput::text(self)
    }
}

impl IntoToolOutput for serde_json::Value {
    fn into_tool_output(self) -> ToolOutput {
        ToolOutput::text(self.to_string())
    }
}
