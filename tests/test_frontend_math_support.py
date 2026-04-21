import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
WEB_ROOT = ROOT / "src" / "web"


class FrontendMathSupportTests(unittest.TestCase):
    def test_generated_javascript_is_built_from_typescript_source(self):
        ts = (WEB_ROOT / "app.ts").read_text(encoding="utf-8")
        js = (WEB_ROOT / "app.js").read_text(encoding="utf-8")
        html = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
        self.assertIn("interface WorkspaceFilePayload", ts)
        self.assertIn("Generated from src/web/app.ts", js)
        self.assertIn("async function loadWorkspaces(preferredId = \"\")", js)
        self.assertNotIn("interface WorkspaceFilePayload", js)
        self.assertIn("function getKatexRenderer()", ts)
        self.assertIn("function renderMathExpression(", ts)
        self.assertIn("function renderDisplayMathExpression(", ts)
        self.assertIn("function renderMathBlock(", ts)
        self.assertIn("function extractInlineMathSegments(", ts)
        self.assertIn('if (["latex", "tex", "math"].includes(normalized)) return "latex";', ts)
        self.assertIn("if (normalizedLanguage === \"latex\") {", ts)
        self.assertIn("codeBlocks.push(renderDisplayMathExpression(cleanCode));", ts)
        self.assertIn("katex.min.css", html)
        self.assertIn("katex.min.js", html)

    def test_repo_includes_local_frontend_build_entrypoint(self):
        package_json = (ROOT / "package.json").read_text(encoding="utf-8")
        chat_launcher = (ROOT / "chat").read_text(encoding="utf-8")
        build_script = (ROOT / "scripts" / "build_frontend.mjs").read_text(encoding="utf-8")
        harness = (ROOT / "src" / "python" / "harness.py").read_text(encoding="utf-8")
        self.assertIn('"build:frontend"', package_json)
        self.assertIn('"postinstall"', package_json)
        self.assertIn('"start"', package_json)
        self.assertIn("stripTypeScriptTypes", build_script)
        self.assertIn('require("typescript")', build_script)
        self.assertIn("Keeping the checked-in src/web/app.js bundle", build_script)
        self.assertIn('mode: "transform"', build_script)
        self.assertIn("build_frontend_bundle()", chat_launcher)
        self.assertIn("do_install() {", chat_launcher)
        self.assertIn("do_start() {", chat_launcher)
        self.assertGreaterEqual(chat_launcher.count("build_frontend_bundle"), 3)
        self.assertIn("install_target_profile()", chat_launcher)
        self.assertIn("resolve_installed_profile_noninteractive", chat_launcher)
        self.assertIn('while [[ "$#" -gt 0 ]]; do', chat_launcher)
        self.assertIn('docker_compose build chat-app', chat_launcher)
        self.assertNotIn("do_kickstart() {", chat_launcher)
        self.assertNotIn("kickstart) do_kickstart ;;", chat_launcher)
        self.assertIn("def ensure_frontend_bundle()", harness)
        self.assertIn("ensure_frontend_bundle()", harness)

    def test_math_rendering_styles_are_present(self):
        css = (WEB_ROOT / "style.css").read_text(encoding="utf-8")
        self.assertIn(".math-inline", css)
        self.assertIn(".math-block", css)
        self.assertIn(".math-frac", css)
        self.assertIn(".math-sqrt", css)


if __name__ == "__main__":
    unittest.main()
