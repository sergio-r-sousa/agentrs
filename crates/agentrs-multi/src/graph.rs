use std::collections::HashMap;

use agentrs_core::{AgentError, AgentOutput, Result};

/// Directed agent graph.
#[derive(Clone, Default)]
pub struct AgentGraph {
    nodes: HashMap<String, String>,
    edges: Vec<GraphEdge>,
    entry: Option<String>,
}

/// Builder for [`AgentGraph`].
#[derive(Clone, Default)]
pub struct AgentGraphBuilder {
    graph: AgentGraph,
}

/// Graph edge between two agent nodes.
#[derive(Clone)]
pub struct GraphEdge {
    /// Source node name.
    pub from: String,
    /// Target node name.
    pub to: String,
    /// Edge predicate.
    pub condition: EdgeCondition,
}

/// Transition predicate for graph orchestration.
#[derive(Clone)]
pub enum EdgeCondition {
    /// Always traverse this edge.
    Always,
    /// Traverse when output contains a keyword.
    Contains(String),
    /// Terminal edge marker.
    End,
}

impl AgentGraph {
    /// Starts building a graph.
    pub fn builder() -> AgentGraphBuilder {
        AgentGraphBuilder::default()
    }

    pub(crate) fn entry(&self) -> Result<&str> {
        self.entry.as_deref().ok_or_else(|| {
            AgentError::InvalidConfiguration("graph entry point not set".to_string())
        })
    }

    pub(crate) fn next(&self, current: &str, output: &AgentOutput) -> Option<String> {
        self.edges
            .iter()
            .find(|edge| {
                edge.from == current
                    && match &edge.condition {
                        EdgeCondition::Always => true,
                        EdgeCondition::Contains(keyword) => output.text.contains(keyword),
                        EdgeCondition::End => false,
                    }
            })
            .map(|edge| edge.to.clone())
    }
}

impl AgentGraphBuilder {
    /// Adds a node by name.
    pub fn node(mut self, name: impl Into<String>) -> Self {
        let name = name.into();
        self.graph.nodes.insert(name.clone(), name);
        self
    }

    /// Adds an edge.
    pub fn edge(
        mut self,
        from: impl Into<String>,
        to: impl Into<String>,
        condition: EdgeCondition,
    ) -> Self {
        self.graph.edges.push(GraphEdge {
            from: from.into(),
            to: to.into(),
            condition,
        });
        self
    }

    /// Sets the graph entry node.
    pub fn entry(mut self, entry: impl Into<String>) -> Self {
        self.graph.entry = Some(entry.into());
        self
    }

    /// Finalizes the graph.
    pub fn build(self) -> Result<AgentGraph> {
        let entry = self.graph.entry()?;
        if !self.graph.nodes.contains_key(entry) {
            return Err(AgentError::InvalidConfiguration(format!(
                "graph entry node '{entry}' is not registered"
            )));
        }
        Ok(self.graph)
    }
}
