#![allow(missing_docs)]

use agentrs::prelude::*;
use dotenvy::dotenv;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenv().ok();
    let mut agent = load_agent_from_yaml("examples/configs/single-agent.yaml").await?;
    let output = agent.run("What is 21 * 2?").await?;
    println!("{}", output.text);
    Ok(())
}
