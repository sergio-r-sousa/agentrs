#![allow(missing_docs)]

use agentrs::prelude::*;
use dotenvy::dotenv;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenv().ok();
    
    let llm = OpenAiProvider::from_env()
        .model("openai/gpt-oss-20b").build()?;

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
