#![allow(missing_docs)]

use agentrs::prelude::*;
use dotenvy::dotenv;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenv().ok();

    let llm = OpenAiProvider::from_env().build()?;

    let researcher = Agent::builder()
        .llm(llm.clone())
        .system("You research the topic and return facts.")
        .tool(WebSearchTool::new())
        .build()?;
    let writer = Agent::builder()
        .llm(llm)
        .system("You write a polished answer from the input.")
        .build()?;

    let mut orchestrator = MultiAgentOrchestrator::builder()
        .add_agent("researcher", researcher)
        .add_agent("writer", writer)
        .routing(RoutingStrategy::Sequential(vec![
            "researcher".to_string(),
            "writer".to_string(),
        ]))
        .build()?;

    let output = orchestrator
        .run("Explain why Rust's ownership helps concurrency")
        .await?;
    println!("{}", output.text);
    Ok(())
}
