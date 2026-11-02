"""
tests/test_parser.py
--------------------
Automated unit test suite verifying:
1. Nested section subtree isolation (parent does not duplicate child).
2. Direct code ownership (each code block assigned to immediate section).
3. Prose isolation (parent prose does not duplicate child text).
4. Breadcrumb hierarchy preservation across nested sections.
5. API definition isolation (dl.py tags extracted atomically & decomposed).
6. Legitimate identical examples in different documents preserved.
"""

import sys
import unittest
from pathlib import Path
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.parser import PythonDocsParser, DocumentUnit


class TestParserIntegrity(unittest.TestCase):
    def setUp(self):
        self.parser = PythonDocsParser()
        self.fixtures_dir = PROJECT_ROOT / "tests" / "fixtures" / "library"
        self.fixtures_dir.mkdir(parents=True, exist_ok=True)
        self.base_dir = PROJECT_ROOT / "tests" / "fixtures"

    def tearDown(self):
        # Clean up any fixture files created
        for p in self.fixtures_dir.glob("*.html"):
            try:
                p.unlink()
            except OSError:
                pass

    def test_nested_section_isolation_and_code_ownership(self):
        """Verifies parent section does not duplicate child code blocks or prose."""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head><title>Test Module — Python 3.12</title></head>
        <body>
          <div class="body" role="main">
            <section id="parent-section">
              <h1>Parent Heading</h1>
              <p>This is parent introductory prose.</p>
              <pre>parent_code_block()</pre>
              
              <section id="child-section">
                <h2>Child Heading</h2>
                <p>This is child specific prose.</p>
                <pre>child_code_block()</pre>
              </section>
            </section>
          </div>
        </body>
        </html>
        """
        test_file = self.fixtures_dir / "nested_doc.html"
        test_file.write_text(html_content, encoding="utf-8")

        doc = self.parser.parse_file(test_file, base_dir=self.base_dir)
        self.assertIsNotNone(doc)

        units = doc.units
        self.assertEqual(len(units), 2, "Must extract exactly 2 section units (parent and child)")

        parent_unit = next(u for u in units if u.unit_id == "sec_parent-section")
        child_unit = next(u for u in units if u.unit_id == "sec_child-section")

        # 1. Code ownership check
        self.assertEqual(parent_unit.code_blocks, ["parent_code_block()"], "Parent must ONLY own its direct code block")
        self.assertEqual(child_unit.code_blocks, ["child_code_block()"], "Child must ONLY own its direct code block")

        # 2. Prose isolation check
        self.assertIn("parent introductory prose", parent_unit.prose_text)
        self.assertNotIn("child specific prose", parent_unit.prose_text, "Parent prose must NOT duplicate child prose")
        self.assertIn("child specific prose", child_unit.prose_text)

        # 3. Breadcrumb hierarchy check
        self.assertIn("Parent Heading", child_unit.breadcrumbs)
        self.assertIn("Child Heading", child_unit.breadcrumbs)

    def test_multi_level_breadcrumb_hierarchy(self):
        """Verifies 3-level section nesting builds complete breadcrumbs [Title > H1 > H2 > H3]."""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head><title>Hierarchy Test — Python 3.12</title></head>
        <body>
          <div class="body" role="main">
            <section id="level-1">
              <h1>Level 1 Title</h1>
              <p>Level 1 prose.</p>
              <section id="level-2">
                <h2>Level 2 Title</h2>
                <p>Level 2 prose.</p>
                <section id="level-3">
                  <h3>Level 3 Title</h3>
                  <p>Level 3 prose.</p>
                  <pre>level_3_code()</pre>
                </section>
              </section>
            </section>
          </div>
        </body>
        </html>
        """
        test_file = self.fixtures_dir / "hierarchy_doc.html"
        test_file.write_text(html_content, encoding="utf-8")

        doc = self.parser.parse_file(test_file, base_dir=self.base_dir)
        self.assertIsNotNone(doc)

        l3_unit = next((u for u in doc.units if u.unit_id == "sec_level-3"), None)
        self.assertIsNotNone(l3_unit)
        self.assertEqual(l3_unit.code_blocks, ["level_3_code()"])
        
        # Verify full breadcrumb chain
        expected_breadcrumbs = ["Hierarchy Test", "Level 1 Title", "Level 2 Title", "Level 3 Title"]
        self.assertEqual(l3_unit.breadcrumbs, expected_breadcrumbs)

    def test_api_dl_isolation(self):
        """Verifies <dl class='py ...'> tags are extracted as api_symbol and decomposed from prose."""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head><title>API Module — Python 3.12</title></head>
        <body>
          <div class="body" role="main">
            <section id="main-section">
              <h1>API Section</h1>
              <p>Section intro text.</p>
              <dl class="py method">
                <dt class="sig sig-object py" id="pkg.MyClass.do_work">
                  <span class="sig-name descname">do_work</span>(<span>val</span>)
                </dt>
                <dd>
                  <p>Do work method documentation.</p>
                  <pre>obj.do_work(123)</pre>
                </dd>
              </dl>
              <p>Section concluding text.</p>
            </section>
          </div>
        </body>
        </html>
        """
        test_file = self.fixtures_dir / "api_doc.html"
        test_file.write_text(html_content, encoding="utf-8")

        doc = self.parser.parse_file(test_file, base_dir=self.base_dir)
        self.assertIsNotNone(doc)

        api_units = [u for u in doc.units if u.unit_type == "api_symbol"]
        prose_units = [u for u in doc.units if u.unit_type == "prose_section"]

        self.assertEqual(len(api_units), 1)
        self.assertEqual(len(prose_units), 1)

        api_unit = api_units[0]
        prose_unit = prose_units[0]

        # API unit has its code block
        self.assertEqual(api_unit.symbol_name, "pkg.MyClass.do_work")
        self.assertEqual(api_unit.code_blocks, ["obj.do_work(123)"])

        # Prose unit did NOT duplicate the API code block or docstring
        self.assertEqual(prose_unit.code_blocks, [])
        self.assertNotIn("Do work method documentation", prose_unit.prose_text)

    def test_legitimate_repeated_code_in_different_documents(self):
        """Verifies parser does NOT globally deduplicate code blocks that appear in distinct documents."""
        common_code = "import os\nprint(os.getcwd())"
        
        doc1_html = f"""
        <!DOCTYPE html><html><head><title>Doc 1</title></head><body>
        <div class="body" role="main">
          <section id="sec-1"><h1>Doc 1 Sec</h1><pre>{common_code}</pre></section>
        </div></body></html>
        """
        doc2_html = f"""
        <!DOCTYPE html><html><head><title>Doc 2</title></head><body>
        <div class="body" role="main">
          <section id="sec-2"><h1>Doc 2 Sec</h1><pre>{common_code}</pre></section>
        </div></body></html>
        """
        f1 = self.fixtures_dir / "doc1.html"
        f2 = self.fixtures_dir / "doc2.html"
        f1.write_text(doc1_html, encoding="utf-8")
        f2.write_text(doc2_html, encoding="utf-8")

        d1 = self.parser.parse_file(f1, base_dir=self.base_dir)
        d2 = self.parser.parse_file(f2, base_dir=self.base_dir)

        # Both must preserve the code block legitimately
        self.assertEqual(d1.units[0].code_blocks, [common_code])
        self.assertEqual(d2.units[0].code_blocks, [common_code])


if __name__ == "__main__":
    unittest.main()
