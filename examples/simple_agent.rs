#![allow(missing_docs)]

use agentrs::prelude::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let llm = OpenAiProvider::from_env()
        .model("openai/gpt-oss-20b")
        .base_url("https://integrate.api.nvidia.com/v1")
        .api_key("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")
        .build()?;

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
