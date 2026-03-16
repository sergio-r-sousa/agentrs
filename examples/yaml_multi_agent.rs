#![allow(missing_docs)]

use agentrs::prelude::*;
use dotenvy::dotenv;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenv().ok();
    let mut orchestrator = load_multi_agent_from_yaml("examples/configs/multi-agent.yaml").await?;
    let output = orchestrator
        .run("Research and explain how Tokio tasks are spawned")
        .await?;
    println!("{}", output.text);
    Ok(())
}
