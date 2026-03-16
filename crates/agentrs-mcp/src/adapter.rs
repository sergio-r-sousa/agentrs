use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::Mutex;

use agentrs_core::{Result, Tool, ToolOutput};

use crate::{client::McpClient, protocol::McpTool};

/// Adapts an MCP tool definition to the local [`Tool`] trait.
pub struct McpToolAdapter {
    client: Arc<Mutex<McpClient>>,
    definition: McpTool,
}

impl McpToolAdapter {
    /// Creates a new adapter.
    pub fn new(client: Arc<Mutex<McpClient>>, definition: McpTool) -> Self {
        Self { client, definition }
    }
}

#[async_trait]
impl Tool for McpToolAdapter {
    fn name(&self) -> &str {
        &self.definition.name
    }

    fn description(&self) -> &str {
        &self.definition.description
    }

    fn schema(&self) -> serde_json::Value {
        self.definition.input_schema.clone()
    }

    async fn call(&self, input: serde_json::Value) -> Result<ToolOutput> {
        let mut client = self.client.lock().await;
        client.call_tool(&self.definition.name, input).await
    }
}
