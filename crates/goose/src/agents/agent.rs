use std::collections::HashMap;

use anyhow::Result;
use async_trait::async_trait;
use futures::stream::BoxStream;
use serde_json::Value;

use super::extension::{ExtensionConfig, ExtensionResult};
use crate::message::Message;
use crate::providers::base::ProviderUsage;
use mcp_core::prompt::Prompt;
use mcp_core::protocol::GetPromptResult;

/// Core trait defining the behavior of an Agent
#[async_trait]
pub trait Agent: Send + Sync {
    /// Create a stream that yields each message as it's generated by the agent
    async fn reply(&self, messages: &[Message]) -> Result<BoxStream<'_, Result<Message>>>;

    /// Add a new MCP client to the agent
    async fn add_extension(&mut self, config: ExtensionConfig) -> ExtensionResult<()>;

    /// Remove an extension by name
    async fn remove_extension(&mut self, name: &str);

    /// List all extensions
    // TODO this needs to also include status so we can tell if extensions are dropped
    async fn list_extensions(&self) -> Vec<String>;

    /// Pass through a JSON-RPC request to a specific extension
    async fn passthrough(&self, extension: &str, request: Value) -> ExtensionResult<Value>;

    /// Get the total usage of the agent
    async fn usage(&self) -> Vec<ProviderUsage>;

    /// Add custom text to be included in the system prompt
    async fn extend_system_prompt(&mut self, extension: String);

    /// Handle a confirmation response for a tool request
    async fn handle_confirmation(&self, request_id: String, confirmed: bool);

    /// Override the system prompt with custom text
    async fn override_system_prompt(&mut self, template: String);

    /// Lists all prompts from all extensions
    async fn list_extension_prompts(&self) -> HashMap<String, Vec<Prompt>>;

    /// Get a prompt result with the given name and arguments
    /// Returns the prompt text that would be used as user input
    async fn get_prompt(&self, name: &str, arguments: Value) -> Result<GetPromptResult>;
}
