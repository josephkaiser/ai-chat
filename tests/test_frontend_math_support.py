import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
WEB_ROOT = ROOT / "src" / "web"


class FrontendMathSupportTests(unittest.TestCase):
    def test_generated_javascript_is_built_from_typescript_source(self):
        ts = (WEB_ROOT / "app.ts").read_text(encoding="utf-8")
        js = (WEB_ROOT / "app.js").read_text(encoding="utf-8")
        html = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
        self.assertIn("import type {", ts)
        self.assertIn("WorkspaceFilePayload,", ts)
        self.assertIn("Generated from src/web/app.ts", js)
        self.assertIn("async function loadWorkspaces(preferredId = \"\")", js)
        self.assertNotIn("WorkspaceFilePayload,", js)
        self.assertIn("function getKatexRenderer()", ts)
        self.assertIn("function renderMathExpression(", ts)
        self.assertIn("function renderDisplayMathExpression(", ts)
        self.assertIn("function renderMathBlock(", ts)
        self.assertIn("function extractInlineMathSegments(", ts)
        self.assertIn("type MarkdownBlock =", ts)
        self.assertIn("function parseMarkdownBlocks(raw: string): MarkdownBlock[]", ts)
        self.assertIn("function renderMarkdownBlocks(blocks: MarkdownBlock[]): string", ts)
        self.assertIn("let activeList: { ordered: boolean; start: number; items: MarkdownListItem[] } | null = null;", ts)
        self.assertIn("return renderMarkdownBlocks(parseMarkdownBlocks(raw));", ts)
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
        model_defaults = (ROOT / "config" / "model-defaults.env").read_text(encoding="utf-8")
        model_override_sample = (ROOT / "config" / "model-overrides.local.env.sample").read_text(encoding="utf-8")
        gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn('"build:frontend"', package_json)
        self.assertIn('"postinstall"', package_json)
        self.assertIn('"start"', package_json)
        self.assertIn("stripTypeScriptTypes", build_script)
        self.assertIn('require("typescript")', build_script)
        self.assertIn("Keeping the checked-in src/web/*.js bundles", build_script)
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
        self.assertIn("ensure_frontend_bundle,", harness)
        self.assertIn("ensure_frontend_bundle(FRONTEND_ASSETS, logger)", harness)
        self.assertIn("MODEL_DEFAULTS_FILE", chat_launcher)
        self.assertIn("MODEL_OVERRIDES_FILE", chat_launcher)
        self.assertIn("load_model_defaults()", chat_launcher)
        self.assertIn("DEFAULT_MODEL_PROFILE=14b", model_defaults)
        self.assertIn("MODEL_NAME=Qwen/Qwen3-14B-AWQ", model_defaults)
        self.assertIn("MODEL_GPU_MEMORY_UTILIZATION=", model_defaults)
        self.assertIn("MODEL_MAX_MODEL_LEN=", model_defaults)
        self.assertIn("MODEL_ENABLE_PREFIX_CACHING=1", model_defaults)
        self.assertIn("MODEL_MAX_MODEL_LEN=16384", model_override_sample)
        self.assertIn("config/model-overrides.local.env", gitignore)
        self.assertIn(".runtime-model.env", gitignore)
        self.assertIn("MODEL_DEFAULTS_FILE", harness)
        self.assertIn("MODEL_OVERRIDES_FILE", harness)
        self.assertIn("load_env_defaults_file", harness)
        self.assertIn("build_model_args_from_env", harness)

    def test_math_rendering_styles_are_present(self):
        css = (WEB_ROOT / "style.css").read_text(encoding="utf-8")
        self.assertIn(".math-inline", css)
        self.assertIn(".math-block", css)
        self.assertIn(".math-frac", css)
        self.assertIn(".math-sqrt", css)


if __name__ == "__main__":
    unittest.main()
