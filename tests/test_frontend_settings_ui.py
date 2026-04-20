import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
WEB_ROOT = ROOT / "src" / "web"


class FrontendSettingsUiTests(unittest.TestCase):
    def test_index_is_now_a_minimal_chat_and_file_viewer_shell(self):
        html = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
        self.assertIn('id="sidebarToggle"', html)
        self.assertIn('id="viewerToggle"', html)
        self.assertIn('id="workspaceSelect"', html)
        self.assertIn('id="workspaceSettingsButton"', html)
        self.assertIn('id="refreshContextEvalButton"', html)
        self.assertIn('id="contextEvalReport"', html)
        self.assertIn('id="conversationList"', html)
        self.assertIn('id="chatMessages"', html)
        self.assertIn('id="composerInput"', html)
        self.assertIn('rows="2"', html)
        self.assertIn('id="connectionBadge"', html)
        self.assertIn('id="composerHint"', html)
        self.assertIn('id="fileList"', html)
        self.assertIn('id="filePreview"', html)
        self.assertIn('id="viewerMeta"', html)
        self.assertIn('type="module" src="/static/app.js', html)
        self.assertIn('id="settingsOverlay"', html)
        self.assertIn('id="resetAppButton"', html)
        self.assertNotIn('id="chatTitle"', html)
        self.assertNotIn('id="aboutOverlay"', html)
        self.assertNotIn('id="draftShell"', html)

    def test_typescript_frontend_keeps_chat_and_file_viewer_entrypoints(self):
        ts = (WEB_ROOT / "app.ts").read_text(encoding="utf-8")
        self.assertIn("interface ConversationSummary", ts)
        self.assertIn("async function sendCurrentMessage()", ts)
        self.assertIn("async function loadDirectory(path: string)", ts)
        self.assertIn("async function openFile(path: string, options:", ts)
        self.assertIn("async function loadContextEvalReport()", ts)
        self.assertIn("function renderContextEvalReport()", ts)
        self.assertIn("function workspaceRelativeCapturePath(sourcePath: string)", ts)
        self.assertIn("function connectWebSocket()", ts)
        self.assertIn('turn_kind: "visible_chat"', ts)
        self.assertIn("auto_approve_tool_permissions: true", ts)
        self.assertIn('type: "permission_response"', ts)
        self.assertIn("function syncShellLayout()", ts)
        self.assertIn("function renameConversation(", ts)
        self.assertIn("function deleteConversation(", ts)
        self.assertIn('data-action="rename"', ts)
        self.assertIn('data-action="delete"', ts)
        self.assertIn("modelName", ts)
        self.assertNotIn("function showAbout()", ts)
        self.assertNotIn("function toggleTheme()", ts)

    def test_css_focuses_on_two_panel_chat_and_viewer_layout(self):
        css = (WEB_ROOT / "style.css").read_text(encoding="utf-8")
        self.assertIn(".main-grid", css)
        self.assertIn(".shell-toggle", css)
        self.assertIn(".chat-messages", css)
        self.assertIn(".viewer-layout", css)
        self.assertIn(".file-preview-frame", css)
        self.assertIn(".conversation-item.active", css)
        self.assertIn(".context-eval-report", css)
        self.assertIn(".context-eval-drill-grid", css)
        self.assertIn(".context-eval-button", css)
        self.assertIn(".context-eval-severity.high", css)
        self.assertIn(".status-badge.streaming", css)
        self.assertIn(".status-badge.loading", css)
        self.assertIn(".composer {", css)
        self.assertIn(".composer-runtime", css)
        self.assertIn("position: sticky;", css)
        self.assertIn('body[data-viewer-open="false"] .viewer-panel', css)
        self.assertIn(".settings-overlay[hidden]", css)


if __name__ == "__main__":
    unittest.main()
