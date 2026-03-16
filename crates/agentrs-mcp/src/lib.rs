#![forbid(unsafe_code)]

//! Model Context Protocol support for `agentrs`.

mod adapter;
mod client;
mod protocol;

pub use adapter::McpToolAdapter;
pub use client::{
    connect_mcp_tools, connect_mcp_tools_with_api_key, connect_mcp_tools_with_headers,
    connect_mcp_tools_with_options, spawn_mcp_tools, McpClient, WebMcpOptions,
};
pub use protocol::{McpCallToolResult, McpMessage, McpTool};
