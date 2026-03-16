use std::{collections::HashMap, sync::Arc};

use agentrs_core::{AgentError, Result, Tool, ToolDefinition, ToolOutput};

/// Registry of named tools available to an agent.
#[derive(Clone, Default)]
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a tool and returns the updated registry.
    pub fn register(mut self, tool: impl Tool + 'static) -> Self {
        self.tools.insert(tool.name().to_string(), Arc::new(tool));
        self
    }

    /// Registers an already boxed tool.
    pub fn register_boxed(mut self, tool: Arc<dyn Tool>) -> Self {
        self.tools.insert(tool.name().to_string(), tool);
        self
    }

    /// Looks up a tool by name.
    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).cloned()
    }

    /// Returns whether a tool exists.
    pub fn contains(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// Returns the number of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Returns true when no tools are registered.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Converts the registry to LLM-facing tool definitions.
    pub fn to_definitions(&self) -> Vec<ToolDefinition> {
        self.tools
            .values()
            .map(|tool| ToolDefinition {
                name: tool.name().to_string(),
                description: tool.description().to_string(),
                schema: tool.schema(),
            })
            .collect()
    }

    /// Calls a tool by name.
    pub async fn call(&self, name: &str, input: serde_json::Value) -> Result<ToolOutput> {
        let Some(tool) = self.get(name) else {
            return Err(AgentError::ToolNotFound(name.to_string()));
        };
        tool.call(input).await
    }

    /// Merges two registries.
    pub fn merge(mut self, other: ToolRegistry) -> Self {
        self.tools.extend(other.tools);
        self
    }

    /// Registers all tools exposed by an MCP server.
    #[cfg(feature = "mcp")]
    pub async fn register_mcp(mut self, command: &str) -> Result<Self> {
        let tools = if command.starts_with("http://") || command.starts_with("https://") {
            agentrs_mcp::connect_mcp_tools(command).await?
        } else {
            agentrs_mcp::spawn_mcp_tools(command).await?
        };
        for tool in tools {
            self = self.register_boxed(tool);
        }
        Ok(self)
    }

    /// Registers all tools exposed by a web MCP endpoint.
    #[cfg(feature = "mcp")]
    pub async fn register_mcp_http(mut self, endpoint: &str) -> Result<Self> {
        let tools = agentrs_mcp::connect_mcp_tools(endpoint).await?;
        for tool in tools {
            self = self.register_boxed(tool);
        }
        Ok(self)
    }

    /// Registers all tools exposed by a web MCP endpoint using an optional API key.
    #[cfg(feature = "mcp")]
    pub async fn register_mcp_http_with_api_key(
        mut self,
        endpoint: &str,
        api_key: impl Into<String>,
    ) -> Result<Self> {
        let tools = agentrs_mcp::connect_mcp_tools_with_api_key(endpoint, api_key).await?;
        for tool in tools {
            self = self.register_boxed(tool);
        }
        Ok(self)
    }

    /// Registers all tools exposed by a web MCP endpoint with full transport options.
    #[cfg(feature = "mcp")]
    pub async fn register_mcp_http_with_options(
        mut self,
        endpoint: &str,
        options: agentrs_mcp::WebMcpOptions,
    ) -> Result<Self> {
        let tools = agentrs_mcp::connect_mcp_tools_with_options(endpoint, options).await?;
        for tool in tools {
            self = self.register_boxed(tool);
        }
        Ok(self)
    }
}
