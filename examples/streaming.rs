#![allow(missing_docs)]

use agentrs::prelude::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let llm = OpenAiProvider::from_env()
        .model("openai/gpt-oss-20b")
        .base_url("https://integrate.api.nvidia.com/v1")
        .api_key("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")
        .build()?;

    let mut agent = Agent::builder().llm(llm).build()?;
    let mut stream = agent.stream_run("Say hello in five words").await?;
    while let Some(event) = stream.next().await {
        match event? {
            AgentEvent::Token(token) => print!("{token}"),
            AgentEvent::Done(_) => println!(),
            _ => {}
        }
    }
    Ok(())
}
