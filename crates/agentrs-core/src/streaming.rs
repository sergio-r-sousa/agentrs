//! Streaming helpers.

use bytes::Bytes;

use crate::{AgentError, Result, StopReason, StreamChunk, ToolCallDelta};

/// Parses a single SSE data chunk into a unified [`StreamChunk`].
pub fn parse_sse_chunk(bytes: Bytes) -> Result<Option<StreamChunk>> {
    let text = std::str::from_utf8(&bytes).map_err(|_| AgentError::InvalidStream)?;

    for line in text.lines() {
        let Some(payload) = line.strip_prefix("data:") else {
            continue;
        };

        let payload = payload.trim();
        if payload.is_empty() {
            continue;
        }

        if payload == "[DONE]" {
            return Ok(None);
        }

        let value: serde_json::Value = serde_json::from_str(payload)?;
        return Ok(Some(map_openai_like_chunk(&value)));
    }

    Ok(None)
}

fn map_openai_like_chunk(value: &serde_json::Value) -> StreamChunk {
    let choice = value
        .get("choices")
        .and_then(|choices| choices.as_array())
        .and_then(|choices| choices.first())
        .cloned()
        .unwrap_or_default();

    let delta_value = choice.get("delta").cloned().unwrap_or_default();
    let delta = delta_value
        .get("content")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .to_string();

    let tool_call_delta = delta_value
        .get("tool_calls")
        .and_then(serde_json::Value::as_array)
        .map(|items| {
            items
                .iter()
                .map(|item| ToolCallDelta {
                    index: item
                        .get("index")
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or_default() as usize,
                    id: item
                        .get("id")
                        .and_then(serde_json::Value::as_str)
                        .map(ToOwned::to_owned),
                    name: item
                        .get("function")
                        .and_then(|function| function.get("name"))
                        .and_then(serde_json::Value::as_str)
                        .map(ToOwned::to_owned),
                    arguments_delta: item
                        .get("function")
                        .and_then(|function| function.get("arguments"))
                        .and_then(serde_json::Value::as_str)
                        .map(ToOwned::to_owned),
                })
                .collect::<Vec<_>>()
        })
        .filter(|items| !items.is_empty());

    let finish_reason = choice
        .get("finish_reason")
        .and_then(serde_json::Value::as_str)
        .map(map_stop_reason);

    StreamChunk {
        delta,
        tool_call_delta,
        finish_reason,
    }
}

/// Maps a provider stop-reason string into the unified enum.
pub fn map_stop_reason(value: &str) -> StopReason {
    match value {
        "stop" | "end_turn" => StopReason::Stop,
        "tool_use" | "tool_calls" => StopReason::ToolUse,
        "length" | "max_tokens" => StopReason::MaxTokens,
        other => StopReason::Other(other.to_string()),
    }
}
