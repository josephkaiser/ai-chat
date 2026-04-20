import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
WEB_ROOT = ROOT / "src" / "web"


class FrontendMathSupportTests(unittest.TestCase):
    def test_generated_javascript_is_built_from_typescript_source(self):
        ts = (WEB_ROOT / "app.ts").read_text(encoding="utf-8")
        js = (WEB_ROOT / "app.js").read_text(encoding="utf-8")
        self.assertIn("interface WorkspaceFilePayload", ts)
        self.assertIn("Generated from src/web/app.ts", js)
        self.assertIn("async function loadWorkspaces(preferredId = \"\")", js)
        self.assertNotIn("interface WorkspaceFilePayload", js)

    def test_repo_includes_local_frontend_build_entrypoint(self):
        package_json = (ROOT / "package.json").read_text(encoding="utf-8")
        chat_launcher = (ROOT / "chat").read_text(encoding="utf-8")
        build_script = (ROOT / "scripts" / "build_frontend.mjs").read_text(encoding="utf-8")
        harness = (ROOT / "src" / "python" / "harness.py").read_text(encoding="utf-8")
        self.assertIn('"build:frontend"', package_json)
        self.assertIn('"postinstall"', package_json)
        self.assertIn('"start"', package_json)
        self.assertIn("stripTypeScriptTypes", build_script)
        self.assertIn('mode: "transform"', build_script)
        self.assertIn("build_frontend_bundle()", chat_launcher)
        self.assertIn("do_install() {", chat_launcher)
        self.assertIn("do_start() {", chat_launcher)
        self.assertGreaterEqual(chat_launcher.count("build_frontend_bundle"), 3)
        self.assertIn('docker_compose build chat-app', chat_launcher)
        self.assertIn("def ensure_frontend_bundle()", harness)
        self.assertIn("ensure_frontend_bundle()", harness)


if __name__ == "__main__":
    unittest.main()
