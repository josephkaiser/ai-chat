import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class FrontendMathSupportTests(unittest.TestCase):
    def test_index_loads_katex_assets(self):
        html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
        self.assertIn("katex.min.css", html)
        self.assertIn("katex.min.js", html)
        self.assertIn("auto-render.min.js", html)

    def test_markdown_renderers_apply_math_post_processing(self):
        js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
        self.assertIn("function renderMathContent(container)", js)
        self.assertIn("renderMathContent(contentDiv);", js)
        self.assertIn("renderMathContent(previewEl);", js)


if __name__ == "__main__":
    unittest.main()
