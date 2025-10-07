use clap::{
    Args,
    Subcommand,
};
use crossterm::execute;
use crossterm::style::{
    self,
    Color,
};

use crate::cli::chat::{
    ChatError,
    ChatSession,
    ChatState,
};
use crate::cli::experiment::experiment_manager::{
    ExperimentManager,
    ExperimentName,
};
use crate::os::Os;

#[derive(Debug, PartialEq, Args)]
pub struct TangentArgs {
    #[command(subcommand)]
    pub subcommand: Option<TangentSubcommand>,
}

#[derive(Debug, PartialEq, Subcommand)]
pub enum TangentSubcommand {
    /// Exit tangent mode and keep the last conversation entry (user question + assistant response)
    Tail,
    Compact,
}

impl TangentArgs {
    async fn send_tangent_telemetry(os: &Os, session: &ChatSession, duration_seconds: i64) {
        if let Err(err) = os
            .telemetry
            .send_tangent_mode_session(
                &os.database,
                session.conversation.conversation_id().to_string(),
                crate::telemetry::TelemetryResult::Succeeded,
                crate::telemetry::core::TangentModeSessionArgs { duration_seconds },
            )
            .await
        {
            tracing::warn!(?err, "Failed to send tangent mode session telemetry");
        }
    }

    pub async fn execute(self, os: &mut Os, session: &mut ChatSession) -> Result<ChatState, ChatError> {
        // Check if tangent mode is enabled
        if !ExperimentManager::is_enabled(os, ExperimentName::TangentMode) {
            execute!(
                session.stderr,
                style::SetForegroundColor(Color::Red),
                style::Print("\nTangent mode is disabled. Enable it with: q settings chat.enableTangentMode true\n"),
                style::SetForegroundColor(Color::Reset)
            )?;
            return Ok(ChatState::PromptUser {
                skip_printing_tools: true,
            });
        }

        match self.subcommand {
            Some(TangentSubcommand::Tail) => {
                // Check if checkpoint is enabled
                if ExperimentManager::is_enabled(os, ExperimentName::Checkpoint) {
                    execute!(
                        session.stderr,
                        style::SetForegroundColor(Color::Yellow),
                        style::Print(
                            "⚠️ Checkpoint is disabled while in tangent mode. Please exit tangent mode if you want to use checkpoint.\n"
                        ),
                        style::SetForegroundColor(Color::Reset),
                    )?;
                }
                if session.conversation.is_in_tangent_mode() {
                    let duration_seconds = session.conversation.get_tangent_duration_seconds().unwrap_or(0);
                    session.conversation.exit_tangent_mode_with_tail();
                    Self::send_tangent_telemetry(os, session, duration_seconds).await;

                    execute!(
                        session.stderr,
                        style::SetForegroundColor(Color::DarkGrey),
                        style::Print("Restored conversation from checkpoint ("),
                        style::SetForegroundColor(Color::Yellow),
                        style::Print("↯"),
                        style::SetForegroundColor(Color::DarkGrey),
                        style::Print(") with last conversation entry preserved.\n"),
                        style::SetForegroundColor(Color::Reset)
                    )?;
                } else {
                    execute!(
                        session.stderr,
                        style::SetForegroundColor(Color::Red),
                        style::Print("You need to be in tangent mode to use tail.\n"),
                        style::SetForegroundColor(Color::Reset)
                    )?;
                }
            },
            Some(TangentSubcommand::Compact) => {
                if ExperimentManager::is_enabled(os, ExperimentName::Checkpoint) {
                    execute!(
                        session.stderr,
                        style::SetForegroundColor(Color::Yellow),
                        style::Print(
                            "⚠️ Checkpoint is disabled while in tangent mode. Please exit tangent mode if you want to use checkpoint.\n"
                        ),
                        style::SetForegroundColor(Color::Reset),
                    )?;
                }
                if session.conversation.is_in_tangent_mode() {
                    let duration_seconds = session.conversation.get_tangent_duration_seconds().unwrap_or(0);
                    let summary = session.compact_tangent_conversation(os).await.unwrap();
                    session.conversation.exit_tangent_mode_with_compact(summary);
                    Self::send_tangent_telemetry(os, session, duration_seconds).await;

                    execute!(
                        session.stderr,
                        style::SetForegroundColor(Color::Green),
                        style::Print("✔ Tangent conversation compacted and summarized!\n"),
                        style::SetForegroundColor(Color::Reset)
                    )?;
                } else {
                    execute!(
                        session.stderr,
                        style::SetForegroundColor(Color::Red),
                        style::Print("You need to be in tangent mode to use /tangent compact.\n"),
                        style::SetForegroundColor(Color::Reset)
                    )?;
                }
            },
            None => {
                if session.conversation.is_in_tangent_mode() {
                    let duration_seconds = session.conversation.get_tangent_duration_seconds().unwrap_or(0);
                    session.conversation.exit_tangent_mode();
                    Self::send_tangent_telemetry(os, session, duration_seconds).await;

                    execute!(
                        session.stderr,
                        style::SetForegroundColor(Color::DarkGrey),
                        style::Print("Restored conversation from checkpoint ("),
                        style::SetForegroundColor(Color::Yellow),
                        style::Print("↯"),
                        style::SetForegroundColor(Color::DarkGrey),
                        style::Print("). - Returned to main conversation.\n"),
                        style::SetForegroundColor(Color::Reset)
                    )?;
                } else {
                    // Check if checkpoint is enabled
                    if ExperimentManager::is_enabled(os, ExperimentName::Checkpoint) {
                        execute!(
                            session.stderr,
                            style::SetForegroundColor(Color::Yellow),
                            style::Print(
                                "⚠️ Checkpoint is disabled while in tangent mode. Please exit tangent mode if you want to use checkpoint.\n"
                            ),
                            style::SetForegroundColor(Color::Reset),
                        )?;
                    }

                    session.conversation.enter_tangent_mode();

                    // Get the configured tangent mode key for display
                    let tangent_key_char = match os
                        .database
                        .settings
                        .get_string(crate::database::settings::Setting::TangentModeKey)
                    {
                        Some(key) if key.len() == 1 => key.chars().next().unwrap_or('t'),
                        _ => 't', // Default to 't' if setting is missing or invalid
                    };
                    let tangent_key_display = format!("ctrl + {}", tangent_key_char.to_lowercase());

                    execute!(
                        session.stderr,
                        style::SetForegroundColor(Color::DarkGrey),
                        style::Print("Created a conversation checkpoint ("),
                        style::SetForegroundColor(Color::Yellow),
                        style::Print("↯"),
                        style::SetForegroundColor(Color::DarkGrey),
                        style::Print("). Use "),
                        style::SetForegroundColor(Color::Green),
                        style::Print(&tangent_key_display),
                        style::SetForegroundColor(Color::DarkGrey),
                        style::Print(" or "),
                        style::SetForegroundColor(Color::Green),
                        style::Print("/tangent"),
                        style::SetForegroundColor(Color::DarkGrey),
                        style::Print(" to restore the conversation later.\n"),
                        style::Print(
                            "Note: this functionality is experimental and may change or be removed in the future.\n"
                        ),
                        style::SetForegroundColor(Color::Reset)
                    )?;
                }
            },
        }

        Ok(ChatState::PromptUser {
            skip_printing_tools: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::cli::agent::Agents;
    use crate::cli::chat::conversation::ConversationState;
    use crate::cli::chat::tool_manager::ToolManager;
    use crate::os::Os;

    #[tokio::test]
    async fn test_tangent_mode_duration_tracking() {
        let mut os = Os::new().await.unwrap();
        let agents = Agents::default();
        let mut tool_manager = ToolManager::default();
        let mut conversation = ConversationState::new(
            "test_conv_id",
            agents,
            tool_manager.load_tools(&mut os, &mut vec![]).await.unwrap(),
            tool_manager,
            None,
            &os,
            false, // mcp_enabled
        )
        .await;

        // Test entering tangent mode
        assert!(!conversation.is_in_tangent_mode());
        conversation.enter_tangent_mode();
        assert!(conversation.is_in_tangent_mode());

        // Should have a duration
        let duration = conversation.get_tangent_duration_seconds();
        assert!(duration.is_some());
        assert!(duration.unwrap() >= 0);

        // Test exiting tangent mode
        conversation.exit_tangent_mode();
        assert!(!conversation.is_in_tangent_mode());
        assert!(conversation.get_tangent_duration_seconds().is_none());
    }
}
