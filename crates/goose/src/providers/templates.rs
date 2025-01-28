use serde_json::json;

use mcp_core::Tool;
use mcp_core::ToolCall;
use mcp_core::content::TextContent;
use crate::message::{Message, MessageContent};
use crate::providers::openai::OpenAiProvider;
use crate::model::ModelConfig;
use crate::providers::base::Provider;

#[derive(Clone, serde::Serialize)]
pub struct TemplatedToolConfig {
    pub stop_tokens: Vec<String>,
}

impl TemplatedToolConfig {
    pub fn deepseek_style() -> Self {
        Self {
            stop_tokens: vec![],  // No special stop tokens needed anymore
        }
    }
}

pub struct TemplateContext<'a> {
    pub system: Option<&'a str>,
    pub tools: Option<&'a [Tool]>,
}

#[derive(serde::Serialize)]
pub struct TemplateRenderer {
    config: TemplatedToolConfig,
    #[serde(skip)]
    parser_provider: OpenAiProvider,
}

impl TemplateRenderer {
    pub fn new(config: TemplatedToolConfig) -> Self {
        // Create OpenAI provider with o1-mini model
        let model_config = ModelConfig::new("gpt-4o-mini".to_string());
        let parser_provider = OpenAiProvider::from_env(model_config)
            .expect("Failed to initialize OpenAI provider for tool parsing");
        
        Self {
            config,
            parser_provider,
        }
    }

    pub fn get_stop_tokens(&self) -> &[String] {
        &self.config.stop_tokens
    }

    pub fn render(&self, context: TemplateContext) -> String {
        let mut output = String::new();

        // Add system message if present
        if let Some(system) = context.system {
            output.push_str(system);
            output.push_str("\n\n");
        }

        // Add tools if present
        if let Some(tools) = context.tools {
            if !tools.is_empty() {
                output.push_str("Available tools:\n");
                for tool in tools {
                    // Create the desired schema format
                    let desired_schema = json!({
                        "name": {
                            "type": "string"
                        },
                        "parameters": tool.input_schema,
                        "required": ["name", "parameters"]
                    });
                    output.push_str(&format!("- Tool name: {}\nTool description: {}\nTool input schema: {}\n", tool.name, tool.description, desired_schema));
                }
                output.push_str("\nTo use a tool, respond with a JSON object with 'name' and 'parameters' fields.\n\n");
                output.push_str("Only use tools when needed. For general questions, respond directly without using tools.\n\n");
            }
        }

        output
    }

    pub async fn parse_tool_calls(&self, response: &str, tools: &[Tool]) -> Result<Vec<ToolCall>, anyhow::Error> {
        // Create a message with the response
        let mut message = Message::user();
        message.content = vec![MessageContent::Text(TextContent {
            text: response.to_string(),
            annotations: None,
        })];
        
        // Use the OpenAI provider to parse the response, passing the tools directly
        // This way the model will use its native function calling capabilities
        let (completion, _) = self.parser_provider.complete("You are a helpful assistant.", &[message], tools).await?;
        
        // Extract any tool calls from the response
        let mut tool_calls = Vec::new();
        
        for content in completion.content.iter() {
            if let MessageContent::ToolRequest(tool_request) = content {
                if let Ok(tool_call) = &tool_request.tool_call {
                    tool_calls.push(tool_call.clone());
                }
            }
        }
        
        Ok(tool_calls)
    }
}