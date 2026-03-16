# agentrs

`agentrs` is a production-oriented, async-native Rust SDK for building AI agents with a unified provider interface, composable tools, Model Context Protocol (MCP) integration, configurable memory, multi-agent orchestration, and YAML-driven runtime loading.

The project is designed for developers who want an ergonomic Rust API without giving up architectural flexibility. You can start with a single assistant in a few lines, then scale to tool-using agents, web MCP integrations, and orchestrated multi-agent workflows without changing the core programming model.

## Why agentrs

- Provider-agnostic abstractions for OpenAI, Azure OpenAI, Anthropic, Gemini, and Ollama
- Async-first execution built on `tokio`, `futures`, and `async_trait`
- Tool calling with built-in tools, MCP tools, and macro-generated custom tools
- Memory backends for simple chat history, sliding windows, token budgets, and vector retrieval
- Multiple agent loop strategies such as ReAct, chain-of-thought style single-pass execution, and plan-and-execute
- Multi-agent orchestration for sequential and parallel execution
- YAML configuration support for loading single-agent and multi-agent runtimes without hand-writing builders
- Feature-gated architecture so consumers only compile the integrations they need

## Architecture

`agentrs` is organized as a workspace of focused crates:

```text
agentrs/
├── crates/agentrs-core     # Core traits, shared types, streaming helpers, errors
├── crates/agentrs-llm      # LLM provider implementations
├── crates/agentrs-tools    # Tool registry, built-in tools, proc-macro integration
├── crates/agentrs-mcp      # MCP stdio and Streamable HTTP clients
├── crates/agentrs-memory   # Memory backends
├── crates/agentrs-agents   # Agent builders and execution loops
├── crates/agentrs-multi    # Multi-agent orchestration
└── crates/agentrs          # Facade crate, prelude, YAML runtime loader
```

### Core Design Principles

- `agentrs-core` defines the contracts: `LlmProvider`, `Tool`, `Memory`, and `Agent`
- `agentrs-llm` adapts concrete model providers into a common chat-completion interface
- `agentrs-tools` keeps tool execution decoupled from providers and agent loops
- `agentrs-mcp` lets the SDK consume both local MCP servers and remote web MCP endpoints
- `agentrs-memory` keeps memory as a pluggable concern rather than coupling it into a runner
- `agentrs-agents` focuses on agent behavior and loop control
- `agentrs-multi` composes multiple agents without changing the single-agent contract
- `agentrs` re-exports the public API and adds configuration loading for production setups

## Technologies Used

- `tokio` for async runtime, process management, filesystem access, and concurrency primitives
- `futures` and `tokio-stream` for streaming and async composition
- `async-trait` for async trait contracts across providers, tools, memory, and agents
- `serde`, `serde_json`, and `serde_yaml` for configuration, requests, and schema-like payloads
- `reqwest` for HTTP transport to model providers and web MCP endpoints
- `thiserror` for structured error handling
- `schemars` for JSON Schema generation in macro-generated tools
- `proc-macro`, `syn`, and `quote` for the `#[tool]` developer ergonomics layer

## Feature Flags

The facade crate exposes the following main features:

```toml
[dependencies]
agentrs = { version = "0.1", features = ["openai", "mcp", "tool-search"] }
```

Available features:

- `openai`
- `azureopenai`
- `anthropic`
- `ollama`
- `gemini`
- `memory-redis`
- `tool-fetch`
- `tool-search`
- `tool-fs`
- `tool-bash`
- `tool-python`
- `mcp`
- `full`

The default build enables `openai`, `azureopenai`, `tool-fetch`, `tool-search`, `tool-fs`, and `mcp`.

## Installation

```toml
[dependencies]
agentrs = { version = "0.1", features = ["openai", "mcp", "tool-fetch", "tool-search"] }
tokio = { version = "1", features = ["full"] }
anyhow = "1"
```

For local models with Ollama:

```toml
[dependencies]
agentrs = { version = "0.1", features = ["ollama", "tool-fs"] }
tokio = { version = "1", features = ["full"] }
anyhow = "1"
```

## Quick Start

```rust
use agentrs::prelude::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let llm = OpenAiProvider::from_env().model("gpt-4o-mini").build()?;

    let mut agent = Agent::builder()
        .llm(llm)
        .system("You are a concise assistant.")
        .tool(CalculatorTool::new())
        .loop_strategy(LoopStrategy::ReAct { max_steps: 4 })
        .build()?;

    let output = agent.run("What is 7 * (8 + 1)?").await?;
    println!("{}", output.text);

    Ok(())
}
```

## Provider Integrations

All providers implement the same `LlmProvider` contract.

### OpenAI

```rust
use agentrs::prelude::*;

let llm = OpenAiProvider::from_env()
    .model("gpt-4o-mini")
    .build()?;
```

### Azure OpenAI

```rust
use agentrs::prelude::*;

let llm = AzureOpenAiProvider::from_env()
    .base_url("https://your-resource.openai.azure.com/openai/v1")
    .model("gpt-4o")
    .build()?;
```

### Anthropic

```rust
use agentrs::prelude::*;

let llm = AnthropicProvider::from_env()
    .model("claude-3-5-sonnet-latest")
    .build()?;
```

### Gemini

```rust
use agentrs::prelude::*;

let llm = GeminiProvider::from_env()
    .model("gemini-2.0-flash")
    .build()?;
```

### Ollama

```rust
use agentrs::prelude::*;

let llm = OllamaProvider::builder()
    .base_url("http://localhost:11434/v1")
    .model("llama3.2")
    .build()?;
```

## Tools

Tools are independent components that can be registered in a `ToolRegistry` and exposed to the model as callable functions.

### Built-in Tools

Available built-in tools include:

- `CalculatorTool`
- `WebFetchTool`
- `WebSearchTool`
- `FileReadTool`
- `FileWriteTool`
- `BashTool` behind `tool-bash`
- `PythonTool` behind `tool-python`

### Registering Tools

```rust
use agentrs::prelude::*;

let tools = ToolRegistry::new()
    .register(CalculatorTool::new())
    .register(WebSearchTool::new())
    .register(FileReadTool::new(Some(".".into())));
```

### Custom Tools with `#[tool]`

```rust
use agentrs::prelude::*;

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
struct ReverseInput {
    text: String,
}

#[tool(name = "reverse_text", description = "Reverse a string")]
async fn reverse_text(input: ReverseInput) -> Result<String> {
    Ok(input.text.chars().rev().collect())
}

let tools = ToolRegistry::new().register(ReverseTextTool::new());
```

## Memory Backends

Memory is pluggable and independent from the agent loop.

### In-Memory History

```rust
use agentrs::prelude::*;

let memory = InMemoryMemory::new();
```

### Sliding Window Memory

```rust
use agentrs::prelude::*;

let memory = SlidingWindowMemory::new(12);
```

### Token-Aware Memory

```rust
use agentrs::prelude::*;

let memory = TokenAwareMemory::new(4_000);
```

### Vector Memory

```rust
use agentrs::prelude::*;

let memory = VectorMemory::new();
```

## Agent Execution Strategies

### ReAct

Best when the model may need to call tools iteratively.

```rust
let mut agent = Agent::builder()
    .llm(llm)
    .tools(tools)
    .loop_strategy(LoopStrategy::ReAct { max_steps: 8 })
    .build()?;
```

### Chain-of-Thought Style Single Pass

Best for direct reasoning without tools.

```rust
let mut agent = Agent::builder()
    .llm(llm)
    .loop_strategy(LoopStrategy::CoT)
    .build()?;
```

### Plan and Execute

Useful for tasks that benefit from an explicit planning stage.

```rust
let mut agent = Agent::builder()
    .llm(llm)
    .tools(tools)
    .loop_strategy(LoopStrategy::PlanAndExecute { max_steps: 6 })
    .build()?;
```

### Streaming Runs

```rust
use agentrs::prelude::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let llm = OpenAiProvider::from_env().model("gpt-4o-mini").build()?;
    let mut agent = Agent::builder().llm(llm).build()?;

    let mut stream = agent.stream_run("Say hello in five words").await?;
    while let Some(event) = stream.next().await {
        match event? {
            AgentEvent::Token(token) => print!("{token}"),
            AgentEvent::Done(output) => println!("\nDone: {}", output.text),
            _ => {}
        }
    }

    Ok(())
}
```

## MCP Integration

`agentrs` supports both local MCP servers over `stdio` and remote MCP endpoints over Streamable HTTP.

### Local MCP Server

```rust
use agentrs::prelude::*;

let tools = ToolRegistry::new()
    .register_mcp("npx -y @modelcontextprotocol/server-filesystem .")
    .await?;
```

### Remote MCP Endpoint

```rust
use agentrs::prelude::*;

let tools = ToolRegistry::new()
    .register_mcp_http("https://mcp.context7.com/mcp")
    .await?;
```

### Remote MCP with Optional API Key

```rust
use agentrs::prelude::*;

let options = WebMcpOptions::new()
    .api_key("your-api-key")
    .api_key_header("X-Context7-API-Key")
    .api_key_prefix("");

let tools = ToolRegistry::new()
    .register_mcp_http_with_options("https://mcp.context7.com/mcp", options)
    .await?;
```

### MCP Example

```rust
use agentrs::prelude::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let llm = OpenAiProvider::from_env().model("gpt-4o-mini").build()?;

    let mut mcp_options = WebMcpOptions::new();
    if let Ok(api_key) = std::env::var("CONTEXT7_API_KEY") {
        mcp_options = mcp_options
            .api_key(api_key)
            .api_key_header("X-Context7-API-Key")
            .api_key_prefix("");
    }

    let tools = ToolRegistry::new()
        .register(CalculatorTool::new())
        .register_mcp_http_with_options("https://mcp.context7.com/mcp", mcp_options)
        .await?;

    let mut agent = Agent::builder().llm(llm).tools(tools).build()?;
    let output = agent
        .run("Use Context7 MCP to find Tokio documentation for spawning tasks.")
        .await?;

    println!("{}", output.text);
    Ok(())
}
```

## Multi-Agent Orchestration

`agentrs-multi` composes existing agents into higher-level workflows.

### Sequential Routing

```rust
use agentrs::prelude::*;

let researcher = Agent::builder()
    .llm(OpenAiProvider::from_env().model("gpt-4o-mini").build()?)
    .system("You gather facts.")
    .tool(WebSearchTool::new())
    .build()?;

let writer = Agent::builder()
    .llm(OpenAiProvider::from_env().model("gpt-4o-mini").build()?)
    .system("You write the final answer.")
    .build()?;

let mut orchestrator = MultiAgentOrchestrator::builder()
    .add_agent("researcher", researcher)
    .add_agent("writer", writer)
    .routing(RoutingStrategy::Sequential(vec![
        "researcher".to_string(),
        "writer".to_string(),
    ]))
    .build()?;

let output = orchestrator.run("Explain Tokio task spawning").await?;
println!("{}", output.text);
```

### Parallel Routing

```rust
use agentrs::prelude::*;

let mut orchestrator = MultiAgentOrchestrator::builder()
    .add_agent("researcher_a", researcher_a)
    .add_agent("researcher_b", researcher_b)
    .routing(RoutingStrategy::Parallel(vec![
        "researcher_a".to_string(),
        "researcher_b".to_string(),
    ]))
    .build()?;
```

## YAML Runtime Loading

One of the main production features of `agentrs` is the ability to create single agents and multi-agent runtimes from `.yaml` files.

This is useful when:

- prompts should change without recompiling code
- environments use different providers or tools
- orchestrations are controlled by deployment configuration
- you want a clean separation between application code and agent definitions

### Load a Single Agent from YAML

```rust
use agentrs::prelude::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut agent = load_agent_from_yaml("examples/configs/single-agent.yaml").await?;
    let output = agent.run("What is 21 * 2?").await?;
    println!("{}", output.text);
    Ok(())
}
```

Example YAML:

```yaml
kind: agent
name: calculator-assistant
llm:
  provider: open_ai
  model: gpt-4o-mini
system: You are a concise assistant that uses tools when helpful.
memory:
  type: sliding_window
  window_size: 8
tools:
  - type: calculator
loop_strategy:
  type: re_act
  max_steps: 4
temperature: 0.1
max_tokens: 512
```

### Load a Multi-Agent Orchestrator from YAML

```rust
use agentrs::prelude::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut orchestrator = load_multi_agent_from_yaml("examples/configs/multi-agent.yaml").await?;
    let output = orchestrator
        .run("Research and explain how Tokio tasks are spawned")
        .await?;
    println!("{}", output.text);
    Ok(())
}
```

Example YAML:

```yaml
kind: multi_agent
agents:
  - name: researcher
    llm:
      provider: open_ai
      model: gpt-4o-mini
    system: You gather technical facts and use documentation tools.
    tools:
      - type: web_search
      - type: mcp
        target: https://mcp.context7.com/mcp
        api_key_header: X-Context7-API-Key
        api_key_prefix: ""
    loop_strategy:
      type: re_act
      max_steps: 5
  - name: writer
    llm:
      provider: open_ai
      model: gpt-4o-mini
    system: You write a polished final answer from the previous agent output.
    memory:
      type: token_aware
      max_tokens: 1200
    loop_strategy:
      type: chain_of_thought
routing:
  type: sequential
  order:
    - researcher
    - writer
```

### Load a Generic Runtime from YAML

```rust
use agentrs::prelude::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut runtime = load_runtime_from_yaml("examples/configs/runtime-agent.yaml").await?;
    let output = runtime.run("Summarize Rust async best practices").await?;
    println!("{}", output.text);
    Ok(())
}
```

This generic loader is useful when a deployment can switch between single-agent and multi-agent modes.

## Error Handling

The SDK uses structured errors from `agentrs-core`:

- provider transport and API failures
- tool validation and execution failures
- memory backend failures
- MCP protocol and transport failures
- invalid configuration and missing fields
- max-step exhaustion in agent loops

Typical usage:

```rust
use agentrs::prelude::*;

match agent.run("hello").await {
    Ok(output) => println!("{}", output.text),
    Err(error) => eprintln!("agent failed: {error}"),
}
```

## Practical Examples Summary

The repository includes runnable examples for the main features:

- `examples/simple_agent.rs`
- `examples/tool_use.rs`
- `examples/streaming.rs`
- `examples/multi_agent.rs`
- `examples/mcp_integration.rs`
- `examples/yaml_single_agent.rs`
- `examples/yaml_multi_agent.rs`
- `examples/yaml_runtime.rs`

Run them with:

```bash
cargo run --example simple_agent
cargo run --example tool_use
cargo run --example streaming
cargo run --example multi_agent
cargo run --example mcp_integration
cargo run --example yaml_single_agent
cargo run --example yaml_multi_agent
cargo run --example yaml_runtime
```

## Environment Variables

Common environment variables used by the provider builders:

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `OPENAI_MODEL`
- `AZURE_OPENAI_KEY`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_MODEL`
- `ANTHROPIC_API_KEY`
- `GEMINI_API_KEY`
- `CONTEXT7_API_KEY`

For Ollama you usually only need the local service running at `http://localhost:11434`.

## Testing and Validation

Typical commands used during development:

```bash
cargo fmt --all
cargo test --workspace
cargo check --examples
```

## Current Scope and Notes

- OpenAI, Azure OpenAI, and Ollama currently provide the most complete request/stream integration
- Anthropic and Gemini support `complete()` and can be used today, but their streaming implementations are intentionally conservative in this version
- Multi-agent YAML currently supports sequential and parallel routing
- YAML config is intended to cover the practical runtime surface, not every internal experimental type

## Roadmap Ideas

- YAML support for graph and supervisor orchestration
- richer streaming behavior for all providers
- tool catalogs and dynamic custom tool loading
- deployment-focused validation and schema export for YAML configs

## License

Licensed under either:

- MIT license
- Apache License, Version 2.0

at your option.
