use std::path::{Path, PathBuf};

#[cfg(any(feature = "bash", feature = "python"))]
use std::process::Stdio;
#[cfg(feature = "bash")]
use std::time::Duration;

#[cfg(feature = "filesystem")]
use agentrs_core::AgentError;
use agentrs_core::{Result, Tool, ToolError, ToolOutput};
use async_trait::async_trait;
#[cfg(feature = "filesystem")]
use tokio::fs;
#[cfg(any(feature = "bash", feature = "python"))]
use tokio::process::Command;
#[cfg(feature = "bash")]
use tokio::time::timeout;

/// Math evaluation tool.
#[derive(Debug, Clone, Default)]
pub struct CalculatorTool;

impl CalculatorTool {
    /// Creates a calculator tool.
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for CalculatorTool {
    fn name(&self) -> &str {
        "calculator"
    }

    fn description(&self) -> &str {
        "Evaluate mathematical expressions safely."
    }

    fn schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate"
                }
            },
            "required": ["expression"]
        })
    }

    async fn call(&self, input: serde_json::Value) -> Result<ToolOutput> {
        let expression = input
            .get("expression")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| ToolError::InvalidInput("missing 'expression'".to_string()))?;

        let result = evaluate_expression(expression)
            .map_err(|error| ToolError::Execution(error.to_string()))?;
        Ok(ToolOutput::text(result.to_string()))
    }
}

fn evaluate_expression(expression: &str) -> std::result::Result<f64, ExpressionError> {
    let mut parser = ExpressionParser::new(expression);
    let value = parser.parse_expression()?;
    parser.skip_whitespace();
    if parser.is_eof() {
        Ok(value)
    } else {
        Err(ExpressionError::UnexpectedToken {
            position: parser.position,
        })
    }
}

#[derive(Debug)]
enum ExpressionError {
    UnexpectedEnd,
    UnexpectedToken { position: usize },
    DivisionByZero,
    InvalidNumber { position: usize },
}

impl std::fmt::Display for ExpressionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnexpectedEnd => formatter.write_str("unexpected end of expression"),
            Self::UnexpectedToken { position } => {
                write!(formatter, "unexpected token at position {position}")
            }
            Self::DivisionByZero => formatter.write_str("division by zero"),
            Self::InvalidNumber { position } => {
                write!(formatter, "invalid number starting at position {position}")
            }
        }
    }
}

impl std::error::Error for ExpressionError {}

struct ExpressionParser<'a> {
    input: &'a str,
    position: usize,
}

impl<'a> ExpressionParser<'a> {
    fn new(input: &'a str) -> Self {
        Self { input, position: 0 }
    }

    fn parse_expression(&mut self) -> std::result::Result<f64, ExpressionError> {
        let mut value = self.parse_term()?;
        loop {
            self.skip_whitespace();
            match self.peek_char() {
                Some('+') => {
                    self.consume_char();
                    value += self.parse_term()?;
                }
                Some('-') => {
                    self.consume_char();
                    value -= self.parse_term()?;
                }
                _ => return Ok(value),
            }
        }
    }

    fn parse_term(&mut self) -> std::result::Result<f64, ExpressionError> {
        let mut value = self.parse_factor()?;
        loop {
            self.skip_whitespace();
            match self.peek_char() {
                Some('*') => {
                    self.consume_char();
                    value *= self.parse_factor()?;
                }
                Some('/') => {
                    self.consume_char();
                    let divisor = self.parse_factor()?;
                    if divisor == 0.0 {
                        return Err(ExpressionError::DivisionByZero);
                    }
                    value /= divisor;
                }
                _ => return Ok(value),
            }
        }
    }

    fn parse_factor(&mut self) -> std::result::Result<f64, ExpressionError> {
        self.skip_whitespace();
        match self.peek_char() {
            Some('(') => {
                self.consume_char();
                let value = self.parse_expression()?;
                self.skip_whitespace();
                match self.consume_char() {
                    Some(')') => Ok(value),
                    Some(_) => Err(ExpressionError::UnexpectedToken {
                        position: self.position.saturating_sub(1),
                    }),
                    None => Err(ExpressionError::UnexpectedEnd),
                }
            }
            Some('+') => {
                self.consume_char();
                self.parse_factor()
            }
            Some('-') => {
                self.consume_char();
                Ok(-self.parse_factor()?)
            }
            Some(char) if char.is_ascii_digit() || char == '.' => self.parse_number(),
            Some(_) => Err(ExpressionError::UnexpectedToken {
                position: self.position,
            }),
            None => Err(ExpressionError::UnexpectedEnd),
        }
    }

    fn parse_number(&mut self) -> std::result::Result<f64, ExpressionError> {
        let start = self.position;
        let mut seen_dot = false;

        while let Some(char) = self.peek_char() {
            if char.is_ascii_digit() {
                self.consume_char();
            } else if char == '.' && !seen_dot {
                seen_dot = true;
                self.consume_char();
            } else {
                break;
            }
        }

        self.input[start..self.position]
            .parse::<f64>()
            .map_err(|_| ExpressionError::InvalidNumber { position: start })
    }

    fn skip_whitespace(&mut self) {
        while matches!(self.peek_char(), Some(char) if char.is_whitespace()) {
            self.consume_char();
        }
    }

    fn peek_char(&self) -> Option<char> {
        self.input[self.position..].chars().next()
    }

    fn consume_char(&mut self) -> Option<char> {
        let char = self.peek_char()?;
        self.position += char.len_utf8();
        Some(char)
    }

    fn is_eof(&self) -> bool {
        self.position >= self.input.len()
    }
}

/// Fetches HTTP content from a URL.
#[cfg(feature = "fetch")]
#[derive(Debug, Clone, Default)]
pub struct WebFetchTool {
    client: reqwest::Client,
}

#[cfg(feature = "fetch")]
impl WebFetchTool {
    /// Creates a web fetch tool.
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }
}

#[cfg(feature = "fetch")]
#[async_trait]
impl Tool for WebFetchTool {
    fn name(&self) -> &str {
        "web_fetch"
    }

    fn description(&self) -> &str {
        "Fetch a URL and return the text body."
    }

    fn schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "url": { "type": "string" },
                "max_chars": { "type": "integer", "default": 4000 }
            },
            "required": ["url"]
        })
    }

    async fn call(&self, input: serde_json::Value) -> Result<ToolOutput> {
        let url = input
            .get("url")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| ToolError::InvalidInput("missing 'url'".to_string()))?;
        let max_chars = input
            .get("max_chars")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(4000) as usize;

        let response = self
            .client
            .get(url)
            .send()
            .await
            .map_err(|error| ToolError::Execution(error.to_string()))?;
        let body = response
            .text()
            .await
            .map_err(|error| ToolError::Execution(error.to_string()))?;

        let output = body.chars().take(max_chars).collect::<String>();
        Ok(ToolOutput::text(output))
    }
}

#[cfg(not(feature = "fetch"))]
#[derive(Debug, Clone, Default)]
pub struct WebFetchTool;

#[cfg(not(feature = "fetch"))]
impl WebFetchTool {
    /// Creates a placeholder web fetch tool.
    pub fn new() -> Self {
        Self
    }
}

/// Searches the web using DuckDuckGo instant answers.
#[cfg(feature = "search")]
#[derive(Debug, Clone, Default)]
pub struct WebSearchTool {
    client: reqwest::Client,
}

#[cfg(feature = "search")]
impl WebSearchTool {
    /// Creates a search tool.
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }
}

#[cfg(feature = "search")]
#[async_trait]
impl Tool for WebSearchTool {
    fn name(&self) -> &str {
        "web_search"
    }

    fn description(&self) -> &str {
        "Search the web for current public information."
    }

    fn schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": { "type": "string" },
                "max_results": { "type": "integer", "default": 5 }
            },
            "required": ["query"]
        })
    }

    async fn call(&self, input: serde_json::Value) -> Result<ToolOutput> {
        let query = input
            .get("query")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| ToolError::InvalidInput("missing 'query'".to_string()))?;
        let max_results = input
            .get("max_results")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(5) as usize;

        let response = self
            .client
            .get("https://api.duckduckgo.com/")
            .query(&[("q", query), ("format", "json"), ("no_html", "1")])
            .send()
            .await
            .map_err(|error| ToolError::Execution(error.to_string()))?;
        let payload = response
            .json::<serde_json::Value>()
            .await
            .map_err(|error| ToolError::Execution(error.to_string()))?;

        let mut lines = Vec::new();
        if let Some(abstract_text) = payload
            .get("AbstractText")
            .and_then(serde_json::Value::as_str)
        {
            if !abstract_text.is_empty() {
                lines.push(format!("Summary: {abstract_text}"));
            }
        }

        if let Some(related) = payload
            .get("RelatedTopics")
            .and_then(serde_json::Value::as_array)
        {
            for item in related.iter().take(max_results) {
                if let Some(text) = item.get("Text").and_then(serde_json::Value::as_str) {
                    lines.push(format!("- {text}"));
                }
            }
        }

        if lines.is_empty() {
            lines.push(format!("No instant answer results found for '{query}'."));
        }

        Ok(ToolOutput::text(lines.join("\n")))
    }
}

#[cfg(not(feature = "search"))]
#[derive(Debug, Clone, Default)]
pub struct WebSearchTool;

#[cfg(not(feature = "search"))]
impl WebSearchTool {
    /// Creates a placeholder search tool.
    pub fn new() -> Self {
        Self
    }
}

/// Reads a file from allowed roots.
#[cfg(feature = "filesystem")]
#[derive(Debug, Clone)]
pub struct FileReadTool {
    allowed_roots: Vec<PathBuf>,
}

#[cfg(feature = "filesystem")]
impl Default for FileReadTool {
    fn default() -> Self {
        Self::new(std::env::current_dir().ok())
    }
}

#[cfg(feature = "filesystem")]
impl FileReadTool {
    /// Creates a file read tool.
    pub fn new(root: impl Into<Option<PathBuf>>) -> Self {
        Self {
            allowed_roots: root.into().into_iter().collect(),
        }
    }

    fn resolve(&self, path: &Path) -> Result<PathBuf> {
        resolve_path(&self.allowed_roots, path)
    }
}

#[cfg(feature = "filesystem")]
#[async_trait]
impl Tool for FileReadTool {
    fn name(&self) -> &str {
        "file_read"
    }

    fn description(&self) -> &str {
        "Read UTF-8 text files from allowed paths."
    }

    fn schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": { "type": "string" }
            },
            "required": ["path"]
        })
    }

    async fn call(&self, input: serde_json::Value) -> Result<ToolOutput> {
        let path = input
            .get("path")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| ToolError::InvalidInput("missing 'path'".to_string()))?;
        let path = self.resolve(Path::new(path))?;
        let content = fs::read_to_string(path).await?;
        Ok(ToolOutput::text(content))
    }
}

#[cfg(not(feature = "filesystem"))]
#[derive(Debug, Clone, Default)]
pub struct FileReadTool;

#[cfg(not(feature = "filesystem"))]
impl FileReadTool {
    /// Creates a placeholder file read tool.
    pub fn new(_root: impl Into<Option<PathBuf>>) -> Self {
        Self
    }
}

/// Writes a file to allowed roots.
#[cfg(feature = "filesystem")]
#[derive(Debug, Clone)]
pub struct FileWriteTool {
    allowed_roots: Vec<PathBuf>,
}

#[cfg(feature = "filesystem")]
impl Default for FileWriteTool {
    fn default() -> Self {
        Self::new(std::env::current_dir().ok())
    }
}

#[cfg(feature = "filesystem")]
impl FileWriteTool {
    /// Creates a file write tool.
    pub fn new(root: impl Into<Option<PathBuf>>) -> Self {
        Self {
            allowed_roots: root.into().into_iter().collect(),
        }
    }

    fn resolve(&self, path: &Path) -> Result<PathBuf> {
        resolve_path(&self.allowed_roots, path)
    }
}

#[cfg(feature = "filesystem")]
#[async_trait]
impl Tool for FileWriteTool {
    fn name(&self) -> &str {
        "file_write"
    }

    fn description(&self) -> &str {
        "Write UTF-8 text files to allowed paths."
    }

    fn schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": { "type": "string" },
                "content": { "type": "string" }
            },
            "required": ["path", "content"]
        })
    }

    async fn call(&self, input: serde_json::Value) -> Result<ToolOutput> {
        let path = input
            .get("path")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| ToolError::InvalidInput("missing 'path'".to_string()))?;
        let content = input
            .get("content")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| ToolError::InvalidInput("missing 'content'".to_string()))?;
        let path = self.resolve(Path::new(path))?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).await?;
        }
        fs::write(path, content).await?;
        Ok(ToolOutput::text("ok"))
    }
}

#[cfg(not(feature = "filesystem"))]
#[derive(Debug, Clone, Default)]
pub struct FileWriteTool;

#[cfg(not(feature = "filesystem"))]
impl FileWriteTool {
    /// Creates a placeholder file write tool.
    pub fn new(_root: impl Into<Option<PathBuf>>) -> Self {
        Self
    }
}

/// Executes shell commands.
#[cfg(feature = "bash")]
#[derive(Debug, Clone, Default)]
pub struct BashTool;

#[cfg(feature = "bash")]
impl BashTool {
    /// Creates a bash tool.
    pub fn new() -> Self {
        Self
    }
}

#[cfg(feature = "bash")]
#[async_trait]
impl Tool for BashTool {
    fn name(&self) -> &str {
        "bash"
    }

    fn description(&self) -> &str {
        "Execute shell commands and capture stdout/stderr."
    }

    fn schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": { "type": "string" },
                "cwd": { "type": "string" },
                "timeout_secs": { "type": "integer", "default": 30 }
            },
            "required": ["command"]
        })
    }

    async fn call(&self, input: serde_json::Value) -> Result<ToolOutput> {
        let command = input
            .get("command")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| ToolError::InvalidInput("missing 'command'".to_string()))?;
        let cwd = input.get("cwd").and_then(serde_json::Value::as_str);
        let timeout_secs = input
            .get("timeout_secs")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(30);

        let mut child = shell_command(command);
        if let Some(cwd) = cwd {
            child.current_dir(cwd);
        }
        child.stdout(Stdio::piped()).stderr(Stdio::piped());

        let output = timeout(Duration::from_secs(timeout_secs), child.output())
            .await
            .map_err(|_| ToolError::Timeout)??;
        let mut text = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr);
        if !stderr.trim().is_empty() {
            if !text.is_empty() {
                text.push('\n');
            }
            text.push_str(&stderr);
        }
        Ok(ToolOutput::text(text))
    }
}

/// Executes a Python snippet.
#[cfg(feature = "python")]
#[derive(Debug, Clone, Default)]
pub struct PythonTool;

#[cfg(feature = "python")]
impl PythonTool {
    /// Creates a Python tool.
    pub fn new() -> Self {
        Self
    }
}

#[cfg(feature = "python")]
#[async_trait]
impl Tool for PythonTool {
    fn name(&self) -> &str {
        "python"
    }

    fn description(&self) -> &str {
        "Execute inline Python code."
    }

    fn schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "script": { "type": "string" }
            },
            "required": ["script"]
        })
    }

    async fn call(&self, input: serde_json::Value) -> Result<ToolOutput> {
        let script = input
            .get("script")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| ToolError::InvalidInput("missing 'script'".to_string()))?;

        let output = Command::new("python")
            .arg("-c")
            .arg(script)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await?;
        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        Ok(ToolOutput::text(format!("{stdout}{stderr}")))
    }
}

#[cfg(feature = "filesystem")]
fn resolve_path(allowed_roots: &[PathBuf], path: &Path) -> Result<PathBuf> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
    };
    let canonical = absolute.canonicalize().or_else(|_| {
        absolute
            .parent()
            .map(|parent| {
                parent
                    .canonicalize()
                    .map(|root| root.join(path.file_name().unwrap_or_default()))
            })
            .unwrap_or_else(|| Err(std::io::Error::from(std::io::ErrorKind::NotFound)))
    })?;

    if allowed_roots.is_empty() {
        return Ok(canonical);
    }

    let allowed = allowed_roots.iter().any(|root| {
        root.canonicalize()
            .map(|canonical_root| canonical.starts_with(canonical_root))
            .unwrap_or(false)
    });

    if allowed {
        Ok(canonical)
    } else {
        Err(AgentError::ToolError(ToolError::PermissionDenied(
            canonical.display().to_string(),
        )))
    }
}

#[cfg(feature = "bash")]
fn shell_command(command: &str) -> Command {
    if cfg!(target_os = "windows") {
        let mut cmd = Command::new("cmd");
        cmd.arg("/C").arg(command);
        cmd
    } else {
        let mut cmd = Command::new("sh");
        cmd.arg("-lc").arg(command);
        cmd
    }
}
