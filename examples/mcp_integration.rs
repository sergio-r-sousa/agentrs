#![allow(missing_docs)]

use agentrs::prelude::*;
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let llm = OpenAiProvider::from_env().model("gpt-4o-mini").build()?;

    let mut mcp_options = WebMcpOptions::new();
    if let Ok(api_key) = std::env::var("CONTEXT7_API_KEY") {
        mcp_options = mcp_options
            .api_key(api_key)
            .api_key_header("X-Context7-API-Key")
            .api_key_prefix("");
    }

    let tools = ToolRegistry::new()
        .register(CalculatorTool::new())
        .register_mcp_http_with_options("https://mcp.context7.com/mcp", mcp_options)
        .await?;

    let mut agent = Agent::builder().llm(llm).tools(tools).build()?;
    let output = agent
        .run("Use Context7 MCP to find Tokio documentation for spawning tasks.")
        .await?;
    println!("{}", output.text);
    Ok(())
}
