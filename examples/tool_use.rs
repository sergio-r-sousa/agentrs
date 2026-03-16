#![allow(missing_docs)]

use agentrs::prelude::*;

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
    let llm = OpenAiProvider::from_env()
        .model("openai/gpt-oss-20b")
        .base_url("https://integrate.api.nvidia.com/v1")
        .api_key("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")
        .build()?;

    let mut agent = Agent::builder()
        .llm(llm)
        .tool(ReverseTextTool::new())
        .build()?;
    let output = agent.run("Use reverse_text to invert 'rust'").await?;
    println!("{}", output.text);
    Ok(())
}
