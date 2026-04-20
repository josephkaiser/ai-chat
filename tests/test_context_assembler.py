import unittest

from src.python.ai_chat.context_assembler import (
    ContextBundle,
    ContextRetrievalAdapters,
    ContextSection,
    assemble_inspect_context,
    assemble_plan_context_async,
    assemble_synthesis_context,
    assemble_verify_context_async,
    assemble_verify_context,
    build_recent_context,
)
from src.python.ai_chat.context_selection_program import (
    DEFAULT_CONTEXT_SELECTION_PROGRAM,
    ContextCandidate,
    ContextSelectionInputs,
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
                ContextSection("first", "First", "A", priority=10, metadata={"selection_rank": "0"}),
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
        self.assertGreaterEqual(bundle.candidate_count, 4)
        self.assertTrue(bundle.scores)

    def test_assemble_verify_context_uses_structured_sections(self):
        session = make_session()

        rendered = assemble_verify_context(session).render()

        self.assertIn("Verifier checks:\n1. Confirm runtime turns stay hidden", rendered)
        self.assertIn("Files changed:\n- src/web/app.js", rendered)
        self.assertIn("Verification focus:\nUse read-only tools and commands to verify likely assumptions.", rendered)

    def test_selection_program_prefers_file_evidence_for_file_requests(self):
        decision = DEFAULT_CONTEXT_SELECTION_PROGRAM.run(
            ContextSelectionInputs(
                phase="verify",
                request_text="verify the app.js file changes",
                candidates=[
                    ContextCandidate(
                        key="request",
                        title="User request",
                        content="verify the app.js file changes",
                        priority=10,
                        required=True,
                        phase_hints=("verify",),
                    ),
                    ContextCandidate(
                        key="files",
                        title="Changed files",
                        content="- src/web/app.js\n- src/python/harness.py",
                        priority=70,
                        tags=("files", "workspace"),
                        phase_hints=("verify",),
                    ),
                    ContextCandidate(
                        key="feedback",
                        title="Feedback",
                        content="Users prefer chat-first editing.",
                        priority=20,
                        tags=("feedback",),
                    ),
                ],
                max_sections=2,
            )
        )

        self.assertEqual(decision.selected_keys, ["request", "files"])

    def test_assemble_synthesis_context_tracks_omitted_candidates(self):
        session = make_session()
        session.scope_audit = {"gaps": ["Need stronger retrieval evidence."]}

        bundle = assemble_synthesis_context(session)
        rendered = bundle.render()

        self.assertIn("Artifacts to inspect:\n[[artifact:.ai/task-board.md]]", rendered)
        self.assertIn("Synthesis focus:\nRead the task board and the most relevant built artifacts before answering.", rendered)
        self.assertGreater(bundle.candidate_count, len(bundle.sections))
        self.assertTrue(bundle.omitted_keys)


class ContextAssemblerAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_plan_context_adds_retrieved_memory_and_workspace_excerpt(self):
        session = make_session()
        session.workspace_snapshot["sample_paths"] = ["src/web/app.js"]

        adapters = ContextRetrievalAdapters(
            conversation_search=lambda _conversation_id, query, _limit: {
                "query": query,
                "matches": [
                    {
                        "role": "user",
                        "snippet": "Earlier we said the app should feel chat-first.",
                        "context_before": "",
                        "context_after": "",
                    }
                ],
            },
            read_workspace_text=lambda _conversation_id, path: (
                "const mode = 'chat';\nfunction sendMessage() { return mode; }"
                if path == "src/web/app.js" else None
            ),
            truncate_output=lambda text, limit: text[:limit],
        )

        bundle = await assemble_plan_context_async(session, retrieval_adapters=adapters)
        rendered = bundle.render()

        self.assertIn("Retrieved chat memory:\n1. user: Earlier we said the app should feel chat-first.", rendered)
        self.assertIn("Relevant workspace excerpts:\n[src/web/app.js]", rendered)

    async def test_async_verify_context_uses_changed_file_excerpt(self):
        session = make_session()

        adapters = ContextRetrievalAdapters(
            conversation_search=lambda _conversation_id, _query, _limit: {"matches": []},
            read_workspace_text=lambda _conversation_id, path: (
                "function rerankContext() { return 'better'; }"
                if path == "src/web/app.js" else None
            ),
            truncate_output=lambda text, limit: text[:limit],
        )

        bundle = await assemble_verify_context_async(session, retrieval_adapters=adapters)
        rendered = bundle.render()

        self.assertIn("Relevant workspace excerpts:\n[src/web/app.js]", rendered)
        self.assertGreaterEqual(bundle.candidate_count, len(bundle.sections))


if __name__ == "__main__":
    unittest.main()
