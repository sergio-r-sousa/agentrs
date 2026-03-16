#![allow(missing_docs)]

use agentrs::prelude::*;
use dotenvy::dotenv;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenv().ok();
    
    let llm = OpenAiProvider::from_env().build()?;

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
