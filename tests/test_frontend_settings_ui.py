import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
WEB_ROOT = ROOT / "src" / "web"


class FrontendSettingsUiTests(unittest.TestCase):
    def test_index_is_now_a_minimal_chat_and_file_viewer_shell(self):
        html = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
        self.assertIn('id="workspaceSelect"', html)
        self.assertIn('id="conversationList"', html)
        self.assertIn('id="chatMessages"', html)
        self.assertIn('id="composerInput"', html)
        self.assertIn('id="fileList"', html)
        self.assertIn('id="filePreview"', html)
        self.assertIn('type="module" src="/static/app.js', html)
        self.assertNotIn('id="settingsOverlay"', html)
        self.assertNotIn('id="aboutOverlay"', html)
        self.assertNotIn('id="draftShell"', html)

    def test_typescript_frontend_keeps_chat_and_file_viewer_entrypoints(self):
        ts = (WEB_ROOT / "app.ts").read_text(encoding="utf-8")
        self.assertIn("interface ConversationSummary", ts)
        self.assertIn("async function sendCurrentMessage()", ts)
        self.assertIn("async function loadDirectory(path: string)", ts)
        self.assertIn("async function openFile(path: string)", ts)
        self.assertIn("function connectWebSocket()", ts)
        self.assertIn('turn_kind: "visible_chat"', ts)
        self.assertNotIn("function showAbout()", ts)
        self.assertNotIn("function toggleTheme()", ts)

    def test_css_focuses_on_two_panel_chat_and_viewer_layout(self):
        css = (WEB_ROOT / "style.css").read_text(encoding="utf-8")
        self.assertIn(".main-grid", css)
        self.assertIn(".chat-messages", css)
        self.assertIn(".viewer-layout", css)
        self.assertIn(".file-preview-frame", css)
        self.assertIn(".conversation-item.active", css)
        self.assertIn(".status-badge.streaming", css)


if __name__ == "__main__":
    unittest.main()
