#![allow(missing_docs)]

use agentrs::core::testing::MockLlmProvider;
use agentrs::prelude::*;

#[tokio::test]
async fn loads_agent_from_yaml_string() {
    let yaml = r#"
name: yaml-agent
llm:
  provider: open_ai
  api_key: test-key
  base_url: http://localhost:1234/v1
  model: test-model
system: You are loaded from YAML.
tools:
  - type: calculator
loop_strategy:
  type: chain_of_thought
"#;

    let result = load_agent_from_yaml_str(yaml).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn loads_multi_agent_from_yaml_string() {
    let yaml = r#"
agents:
  - name: first
    llm:
      provider: open_ai
      api_key: test-key
      base_url: http://localhost:1234/v1
      model: test-model
    system: First agent.
  - name: second
    llm:
      provider: open_ai
      api_key: test-key
      base_url: http://localhost:1234/v1
      model: test-model
    system: Second agent.
routing:
  type: parallel
  agents:
    - first
    - second
"#;

    let result = load_multi_agent_from_yaml_str(yaml).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn loads_runtime_from_yaml_string() {
    let yaml = r#"
kind: agent
llm:
  provider: open_ai
  api_key: test-key
  base_url: http://localhost:1234/v1
  model: test-model
tools:
  - type: calculator
"#;

    let result = load_runtime_from_yaml_str(yaml).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn agent_executes_tool_calls() {
    let llm = MockLlmProvider::with_tool_call_sequence(
        "calculator",
        serde_json::json!({ "expression": "2 + 2" }),
        "The answer is 4.",
    );

    let mut agent = Agent::builder()
        .llm(llm)
        .tool(CalculatorTool::new())
        .build()
        .unwrap();

    let output = agent.run("What is 2 + 2?").await.unwrap();
    assert!(output.text.contains('4'));
    assert!(output.steps >= 1);
}

#[tokio::test]
async fn sequential_multi_agent_pipeline() {
    let researcher = Agent::builder()
        .llm(MockLlmProvider::with_text_responses(vec![
            "Rust uses ownership to prevent data races.",
        ]))
        .build()
        .unwrap();
    let writer = Agent::builder()
        .llm(MockLlmProvider::with_text_responses(vec![
            "Final answer: ownership enables safe concurrency.",
        ]))
        .build()
        .unwrap();

    let mut orchestrator = MultiAgentOrchestrator::builder()
        .add_agent("researcher", researcher)
        .add_agent("writer", writer)
        .routing(RoutingStrategy::Sequential(vec![
            "researcher".to_string(),
            "writer".to_string(),
        ]))
        .build()
        .unwrap();

    let output = orchestrator.run("Explain Rust concurrency").await.unwrap();
    assert!(output.text.contains("Final answer"));
}
