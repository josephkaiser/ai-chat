import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class FrontendSettingsUiTests(unittest.TestCase):
    def test_settings_no_longer_contains_workspace_panel_or_runtime_approvals(self):
        html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
        self.assertNotIn('id="settingWorkspacePanel"', html)
        self.assertNotIn("Runtime Tool Approvals", html)

    def test_about_is_exposed_as_separate_overlay(self):
        html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
        js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
        self.assertIn('showAbout()', html)
        self.assertIn('id="aboutOverlay"', html)
        self.assertIn("function showAbout()", js)
        self.assertIn("function closeAbout()", js)
        self.assertNotIn("workspace_panel:", js)
        self.assertNotIn("featureSettings.workspace_panel", js)

    def test_workspace_markup_exposes_artifact_rail(self):
        html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
        js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
        self.assertIn('id="workspaceArtifactList"', html)
        self.assertIn('id="workspaceArtifactsCount"', html)
        self.assertIn(">Recent</div>", html)
        self.assertIn(">All Files</div>", html)
        self.assertIn("managed chat environment", js)
        self.assertNotIn("Create or reuse `.venv`, then install Python packages with pip.", js)

    def test_theme_defaults_to_dark_until_user_chooses_otherwise(self):
        html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
        js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
        self.assertIn('content="#1c1a17"', html)
        self.assertIn("const DEFAULT_THEME = 'dark';", js)
        self.assertIn("applyTheme(localStorage.getItem('theme') || DEFAULT_THEME);", js)

    def test_plan_approval_uses_explicit_approval_copy_and_edit_mode(self):
        html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
        js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
        self.assertIn('id="planApprovalCallout"', html)
        self.assertIn('id="planApprovalApproveButton"', html)
        self.assertIn('>Approve And Run</button>', html)
        self.assertIn('id="planApprovalEditButton"', html)
        self.assertIn('>Edit Steps</button>', html)
        self.assertIn('id="planApprovalRunEditedButton"', html)
        self.assertIn('>Approve Edited Plan</button>', html)
        self.assertIn('>Cancel Edit</button>', html)
        self.assertIn('Ctrl/Cmd+Enter also approves and runs.', html)
        self.assertIn("function startExecutionPlanEdit()", js)
        self.assertIn("function cancelExecutionPlanEdit()", js)
        self.assertIn("function focusExecutionPlanApproveButton()", js)
        self.assertNotIn("Run this plan?", html)
        self.assertNotIn("addMessage(APPROVED_PLAN_EXECUTION_MESSAGE, 'user'", js)

    def test_permission_panel_explains_pause_behavior(self):
        html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
        js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
        self.assertIn('id="permissionPanelNote"', html)
        self.assertIn('>Approve and continue</button>', html)
        self.assertIn('>Pause task</button>', html)
        self.assertNotIn("recentPermissionResponses", js)

    def test_composer_exposes_per_chat_tool_auto_approve_toggle(self):
        html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
        js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
        self.assertIn('id="toolApprovalToggle"', html)
        self.assertIn('id="toolApprovalToggleValue"', html)
        self.assertIn(">Tools</span>", html)
        self.assertIn("function toggleToolApprovalMode()", js)
        self.assertIn("function syncToolApprovalToggle()", js)
        self.assertIn("auto_approve_tool_permissions", js)

    def test_workspace_viewer_can_preview_and_auto_open_image_artifacts(self):
        js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
        css = (ROOT / "static" / "style.css").read_text(encoding="utf-8")
        self.assertIn("function renderImagePreview(targetId, path)", js)
        self.assertIn("function shouldAutoOpenArtifactPreview(path)", js)
        self.assertIn("data.name === 'workspace.run_command' && data.ok !== false && shouldAutoOpenArtifactPreview(data.payload?.open_path)", js)
        self.assertIn("renderImagePreview('inlineViewerPreview', path);", js)
        self.assertIn("['text', 'markdown', 'html', 'csv', 'pdf', 'spreadsheet', 'image', 'archive'].includes(backendKind)", js)
        self.assertIn(".workspace-image-preview", css)

    def test_workspace_viewer_exposes_archive_preview_and_extract_controls(self):
        html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
        js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
        css = (ROOT / "static" / "style.css").read_text(encoding="utf-8")
        self.assertIn('id="inlineViewerExtractButton"', html)
        self.assertIn("function renderArchivePreview(targetId, data = {})", js)
        self.assertIn("function extractInlineViewerArchive()", js)
        self.assertIn("function extractWorkspaceArchive(path)", js)
        self.assertIn(".workspace-archive-preview", css)

    def test_workspace_artifact_rail_no_longer_backfills_with_arbitrary_repo_files(self):
        js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
        self.assertIn("No recent turn artifacts yet. Browse All Files below", js)
        self.assertIn("currentAssistantTurnArtifactPaths.has(entry.path)", js)
        self.assertNotIn("sortWorkspaceFilesByRecent(files).forEach(entry => addEntryPath(entry.path));", js)

    def test_chat_artifact_references_promote_into_viewer_and_split_layout(self):
        js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
        css = (ROOT / "static" / "style.css").read_text(encoding="utf-8")
        self.assertIn("const ARTIFACT_REFERENCE_PATTERN =", js)
        self.assertIn("const ARTIFACT_HELPER_TEXT_PATTERNS =", js)
        self.assertIn("function enhanceMessageArtifactReferences(msg, container, rawContent = '')", js)
        self.assertIn("function maybeAutoOpenReferencedArtifact(msg, rawContent = '')", js)
        self.assertIn("function shouldCollapseArtifactHelperParagraph(paragraph)", js)
        self.assertIn("function isMostRecentAssistantMessage(msg)", js)
        self.assertIn("openWorkspaceFile(resolvedPath);", js)
        self.assertIn("if (typeof payload.open_path === 'string' && payload.open_path)", js)
        self.assertIn("root.classList.toggle('workspace-reader-open', !mobileLayout && showReader);", js)
        self.assertIn("extractArtifactReferences(rawContent).length === 1 && isMostRecentAssistantMessage(msg)", js)
        self.assertIn(".message-artifact-link", css)
        self.assertIn(".chat.workspace-reader-open .input-area-stack", css)
        self.assertIn(".chat.workspace-reader-open .messages", css)

    def test_html_preview_rewrites_local_asset_references_for_workspace_viewer(self):
        js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
        self.assertIn("function prepareHtmlPreviewContent(content, path)", js)
        self.assertIn("function resolveWorkspaceAssetPath(basePath, assetRef)", js)
        self.assertIn("workspaceFileInlineViewUrl(resolvedPath)", js)
        self.assertIn("renderHtmlPreview('inlineViewerPreview', content, path);", js)
        self.assertIn("iframe.srcdoc = prepareHtmlPreviewContent(content, path);", js)


if __name__ == "__main__":
    unittest.main()
