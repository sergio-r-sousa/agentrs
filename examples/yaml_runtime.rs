#![allow(missing_docs)]

use agentrs::prelude::*;
use dotenvy::dotenv;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenv().ok();
    let mut runtime = load_runtime_from_yaml("examples/configs/runtime-agent.yaml").await?;
    let output = runtime.run("Summarize Rust async best practices").await?;
    println!("{}", output.text);
    Ok(())
}
