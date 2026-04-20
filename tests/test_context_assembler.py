import unittest

from src.python.ai_chat.context_assembler import (
    ContextBundle,
    ContextSection,
    assemble_inspect_context,
    assemble_verify_context,
    build_recent_context,
)
from src.python.ai_chat.deep_runtime import DeepSession


def make_session() -> DeepSession:
    return DeepSession(
        websocket=object(),
        conversation_id="conv",
        message="Improve the document flow",
        task_request="Improve the document flow",
        history=[{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}],
        system_prompt="System",
        max_tokens=1024,
        features=type("Features", (), {"workspace_write": True})(),
        context="user: Hello\nassistant: Hi",
        workspace_enabled=True,
        workspace_facts="Found app.js and harness.py",
        workspace_snapshot={"root": "/tmp/workspace", "total_files": 4, "user_file_count": 3, "total_dirs": 2},
        recent_product_feedback_summary="Users want chat-first editing.",
        recent_product_feedback_artifact_path=".ai/recent-feedback.md",
        task_board_path=".ai/task-board.md",
        plan={
            "deliverable": "A cleaner chat-first workflow",
            "verifier_checks": ["Confirm runtime turns stay hidden", "Confirm chat stays visible"],
        },
        build_summary="Split chat and runtime turns.",
        changed_files=["src/web/app.js", "src/python/harness.py"],
        agent_outputs={
            "agent_a_role": "builder",
            "agent_b_role": "verifier",
            "output_a": "Implemented the new split.",
            "output_b": "Need to confirm retrieval stays clean.",
        },
        verification_summary="Looks good so far.",
        scope_audit_summary="Still need a better context layer.",
    )


class ContextAssemblerTests(unittest.TestCase):
    def test_context_bundle_renders_by_priority(self):
        bundle = ContextBundle(
            phase="test",
            sections=[
                ContextSection("later", "Later", "B", priority=20),
                ContextSection("first", "First", "A", priority=10),
                ContextSection("empty", "Empty", "", priority=0),
            ],
        )

        self.assertEqual(bundle.render(), "First:\nA\n\nLater:\nB")

    def test_build_recent_context_formats_latest_messages(self):
        history = [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
            {"role": "user", "content": "three"},
        ]

        self.assertEqual(
            build_recent_context(history, limit=2),
            "assistant: two\nuser: three",
        )

    def test_assemble_inspect_context_includes_feedback_and_snapshot(self):
        session = make_session()

        bundle = assemble_inspect_context(
            session,
            workspace_snapshot_formatter=lambda snapshot: f"root={snapshot.get('root')}",
        )
        rendered = bundle.render()

        self.assertIn("Conversation context:\nuser: Hello", rendered)
        self.assertIn("Recent product feedback to treat as failure signals for this pass", rendered)
        self.assertIn("[[artifact:.ai/recent-feedback.md]]", rendered)
        self.assertIn("Deterministic workspace snapshot:\nroot=/tmp/workspace", rendered)

    def test_assemble_verify_context_uses_structured_sections(self):
        session = make_session()

        rendered = assemble_verify_context(session).render()

        self.assertIn("Verifier checks:\n1. Confirm runtime turns stay hidden", rendered)
        self.assertIn("Files changed:\nsrc/web/app.js, src/python/harness.py", rendered)
        self.assertIn("Verification focus:\nUse read-only tools and commands to verify likely assumptions.", rendered)


if __name__ == "__main__":
    unittest.main()
