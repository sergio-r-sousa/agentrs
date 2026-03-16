#![allow(missing_docs)]

use agentrs::prelude::*;
use dotenvy::dotenv;

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
struct ReverseInput {
    text: String,
}

#[tool(name = "reverse_text", description = "Reverse a string")]
async fn reverse_text(input: ReverseInput) -> Result<String> {
    Ok(input.text.chars().rev().collect())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenv().ok();
    
    let llm = OpenAiProvider::from_env().build()?;

    let mut agent = Agent::builder()
        .llm(llm)
        .tool(ReverseTextTool::new())
        .build()?;
    let output = agent.run("Use reverse_text to invert 'rust'").await?;
    println!("{}", output.text);
    Ok(())
}
