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


if __name__ == "__main__":
    unittest.main()
