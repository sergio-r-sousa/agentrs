use std::{collections::HashMap, sync::Arc};

use agentrs_core::LlmProvider;

/// Runtime registry for LLM providers.
#[derive(Clone, Default)]
pub struct ProviderRegistry {
    providers: HashMap<String, Arc<dyn LlmProvider>>,
}

impl ProviderRegistry {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a provider.
    pub fn register(mut self, provider: impl LlmProvider + 'static) -> Self {
        self.providers
            .insert(provider.name().to_string(), Arc::new(provider));
        self
    }

    /// Returns a provider by name.
    pub fn get(&self, name: &str) -> Option<Arc<dyn LlmProvider>> {
        self.providers.get(name).cloned()
    }
}
