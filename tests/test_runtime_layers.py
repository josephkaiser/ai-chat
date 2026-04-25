import unittest

from src.python.ai_chat.runtime_layers import (
    TURN_KIND_RUNTIME_FILE,
    TURN_KIND_VISIBLE_CHAT,
    build_model_history,
    compose_model_message,
    compose_runtime_turn,
    normalize_turn_kind,
    visible_message_kind_sql,
)


class RuntimeLayersTests(unittest.TestCase):
    def test_compose_visible_chat_keeps_effective_message_clean_and_model_message_augmented(self):
        envelope = compose_runtime_turn(
            raw_message="Refine the homepage copy.",
            attachment_context="Attachment: brief.md",
            runtime_context="\n<active_draft>\npath: index.html\n</active_draft>",
            turn_kind=TURN_KIND_VISIBLE_CHAT,
        )

        self.assertEqual(envelope.saved_user_message, "Refine the homepage copy.\n\nAttachment: brief.md")
        self.assertEqual(envelope.effective_message, "Refine the homepage copy.\n\nAttachment: brief.md")
        self.assertTrue(envelope.model_message.endswith("</active_draft>"))
        self.assertEqual([layer.name for layer in envelope.layers], ["attachments", "runtime_context"])

    def test_compose_slash_turn_uses_slash_request_for_execution(self):
        envelope = compose_runtime_turn(
            raw_message="/code fix the navbar",
            attachment_context="Attachment: screenshot.png",
            slash_request="fix the navbar\n\nAttachment: screenshot.png",
            runtime_context="\n<active_draft>\npath: src/web/app.js\n</active_draft>",
        )

        self.assertEqual(envelope.saved_user_message, "/code fix the navbar\n\nAttachment: screenshot.png")
        self.assertEqual(envelope.effective_message, "fix the navbar\n\nAttachment: screenshot.png")
        self.assertIn("<active_draft>", envelope.model_message)

    def test_build_model_history_rewrites_latest_user_turn_only_for_model(self):
        history = [
            {"role": "user", "content": "Explain the diff."},
            {"role": "assistant", "content": "Here is the diff."},
            {"role": "user", "content": "Refine the homepage copy."},
        ]

        model_history = build_model_history(
            history,
            effective_message="Refine the homepage copy.",
            model_message="Refine the homepage copy.\n<active_draft>\npath: index.html\n</active_draft>",
        )

        self.assertEqual(history[-1]["content"], "Refine the homepage copy.")
        self.assertIn("<active_draft>", model_history[-1]["content"])

    def test_compose_model_message_rebases_runtime_context_on_rewritten_request(self):
        model_message = compose_model_message(
            "Save or update the workspace file index.html for this request: refine the homepage copy",
            runtime_context="\n<active_draft>\npath: index.html\n</active_draft>",
        )

        self.assertTrue(model_message.startswith("Save or update the workspace file index.html"))
        self.assertNotIn("\nyes\n", model_message.lower())
        self.assertIn("<active_draft>", model_message)

    def test_turn_kind_and_visible_sql_helpers_are_stable(self):
        self.assertEqual(normalize_turn_kind("runtime_file"), TURN_KIND_RUNTIME_FILE)
        self.assertEqual(normalize_turn_kind("anything-else"), TURN_KIND_VISIBLE_CHAT)
        self.assertEqual(
            visible_message_kind_sql("m.kind"),
            "COALESCE(m.kind, 'visible_chat') = 'visible_chat'",
        )


if __name__ == "__main__":
    unittest.main()
