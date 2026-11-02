"""
src/parser.py
-------------
Data-aware parser tailored specifically to Python 3.12 Sphinx HTML documentation.
Captures atomic API definition blocks (<dl class="py">), conceptual sections,
exact code blocks with indentation, and version directives (versionadded/changed/deprecated).
"""

import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from bs4 import BeautifulSoup, Tag

# Ensure project root is in sys.path regardless of how script is executed
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import config


@dataclass
class VersionDirective:
    """Represents a version constraint (e.g., Added in 3.12, Deprecated in 3.11)."""
    directive_type: str  # 'added', 'changed', 'deprecated'
    target_version: str  # e.g., '3.12', '3.11'
    raw_text: str        # e.g., 'Added in version 3.12.'


@dataclass
class DocumentUnit:
    """An atomic unit of documentation (either an API entity or a conceptual section)."""
    unit_id: str
    unit_type: str          # 'api_symbol' or 'prose_section'
    symbol_name: Optional[str] = None
    signature: Optional[str] = None
    breadcrumbs: List[str] = field(default_factory=list)
    prose_text: str = ""
    code_blocks: List[str] = field(default_factory=list)
    version_directives: List[VersionDirective] = field(default_factory=list)


@dataclass
class ParsedDocument:
    """Complete document containing atomic documentation units."""
    doc_id: str
    relative_path: str
    title: str
    python_version: str
    units: List[DocumentUnit] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PythonDocsParser:
    """Production-grade parser for Python documentation."""

    # Selectors to strip layout chrome
    NOISE_SELECTORS = [
        "div.sphinxsidebar",
        "div.mobile-nav",
        "div.related",
        "footer",
        "nav",
        "script",
        "style",
        "a.headerlink",  # Removes the '¶' anchor link
    ]

    VERSION_REGEX = re.compile(
        r"(Added|Changed|Deprecated|Removed)\s+(?:in|since)\s+version\s+([0-9]+\.[0-9]+)",
        re.IGNORECASE,
    )

    def is_file_eligible(self, relative_path: Path) -> bool:
        """Filter out non-documentation files, git commit dumps, and old versions."""
        path_str = str(relative_path).lower()

        # 1. Reject noise files
        if any(noise in path_str for noise in ["genindex", "search", "contents", "py-modindex"]):
            return False

        # 2. Reject historical changelogs and the 4.7MB git commit dump
        if "whatsnew/changelog" in path_str:
            return False
        if "whatsnew/" in path_str and not path_str.endswith("whatsnew/3.12.html"):
            return False

        # 3. Whitelist core documentation directories
        valid_prefixes = ("library/", "reference/", "howto/", "tutorial/", "deprecations/", "whatsnew/3.12.html")
        return path_str.startswith(valid_prefixes)

    def parse_file(self, file_path: Path, base_dir: Path) -> Optional[ParsedDocument]:
        """Parse an eligible HTML documentation file into structured DocumentUnits."""
        relative_path = file_path.relative_to(base_dir)
        if not self.is_file_eligible(relative_path):
            return None

        with file_path.open("r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f.read(), "lxml")

        # 1. Remove layout chrome
        for selector in self.NOISE_SELECTORS:
            for node in soup.select(selector):
                node.decompose()

        # 2. Locate main content body
        main_content = soup.find("div", class_="body") or soup.find("div", role="main")
        if not main_content:
            return None

        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True).split("—")[0].strip() if title_tag else file_path.stem
        doc_id = str(relative_path).replace("/", "_").replace(".html", "")

        doc = ParsedDocument(
            doc_id=doc_id,
            relative_path=str(relative_path),
            title=title,
            python_version=config.version.primary_version,
            units=[],
        )

        # 3. First pass: Extract atomic API entities (<dl class="py ...">)
        for dl in main_content.find_all("dl", class_=re.compile(r"py\s+(function|method|class|attribute|data|exception)")):
            unit = self._parse_api_block(dl, doc_title=title)
            if unit:
                doc.units.append(unit)
            # Remove from tree so it is not duplicated in section text
            dl.decompose()

        # 4. Second pass: Extract remaining conceptual sections (<section> or <div class="section">)
        sections = main_content.find_all("section") or main_content.find_all("div", class_="section")
        if not sections:
            unit = self._parse_prose_block(main_content, section_id="main", breadcrumbs=[title])
            if unit and (unit.prose_text or unit.code_blocks or unit.version_directives):
                doc.units.append(unit)
        else:
            for sec in sections:
                sec_id = sec.get("id") or "section"
                heading_tag = sec.find(["h1", "h2", "h3", "h4", "h5", "h6"])
                heading = heading_tag.get_text(strip=True) if heading_tag else title

                # Build hierarchical breadcrumbs by walking up ancestor sections
                ancestor_headings = []
                curr = sec.parent
                while curr and curr != main_content:
                    if isinstance(curr, Tag):
                        is_sec = curr.name == "section"
                        if not is_sec and curr.attrs and "class" in curr.attrs:
                            is_sec = "section" in curr.attrs["class"]
                        if is_sec:
                            h_tag = curr.find(["h1", "h2", "h3", "h4", "h5", "h6"])
                            if h_tag:
                                h_text = h_tag.get_text(strip=True)
                                if h_text and (not ancestor_headings or ancestor_headings[-1] != h_text):
                                    ancestor_headings.append(h_text)
                    curr = curr.parent

                full_breadcrumbs = [title] + list(reversed(ancestor_headings))
                if not full_breadcrumbs or full_breadcrumbs[-1] != heading:
                    full_breadcrumbs.append(heading)

                # Clean adjacent duplicate titles
                cleaned_breadcrumbs = []
                for b in full_breadcrumbs:
                    if not cleaned_breadcrumbs or cleaned_breadcrumbs[-1].lower() != b.lower():
                        cleaned_breadcrumbs.append(b)

                unit = self._parse_prose_block(sec, section_id=sec_id, breadcrumbs=cleaned_breadcrumbs)
                if unit and (unit.prose_text or unit.code_blocks or unit.version_directives):
                    doc.units.append(unit)

        return doc

    def _parse_api_block(self, dl_tag: Tag, doc_title: str) -> Optional[DocumentUnit]:
        """Extract atomic API definition (function, method, class) with signatures and version notes."""
        dt = dl_tag.find("dt", class_="sig")
        dd = dl_tag.find("dd")
        if not dt or not dd:
            return None

        symbol_name = dt.get("id") or dt.get_text(" ", strip=True).split("(")[0].strip()
        signature = dt.get_text(" ", strip=True)

        # Extract code blocks preserving indentation
        code_blocks = [pre.get_text().rstrip() for pre in dd.find_all("pre") if pre.get_text().strip()]

        # Extract version directives
        version_directives = self._extract_version_directives(dd)

        # Extract clean prose
        dd_clone = BeautifulSoup(str(dd), "lxml")
        for pre in dd_clone.find_all("pre"):
            pre.replace_with(" ")
        prose_text = " ".join(dd_clone.get_text().split())

        return DocumentUnit(
            unit_id=f"api_{symbol_name}",
            unit_type="api_symbol",
            symbol_name=symbol_name,
            signature=signature,
            breadcrumbs=[doc_title, symbol_name],
            prose_text=prose_text,
            code_blocks=code_blocks,
            version_directives=version_directives,
        )

    def _parse_prose_block(self, sec_tag: Tag, section_id: str, breadcrumbs: List[str]) -> Optional[DocumentUnit]:
        """Extract general conceptual documentation with strict subtree isolation."""
        sec_clone = BeautifulSoup(str(sec_tag), "lxml")
        root = sec_clone.find(["section", "div"])
        if not root:
            return None

        # Decompose all descendant child sections so this section only owns its direct content
        for child in root.find_all(["section", "div"]):
            if child is not root and isinstance(child, Tag):
                is_sec = child.name == "section"
                if not is_sec and child.attrs and "class" in child.attrs:
                    is_sec = "section" in child.attrs["class"]
                if is_sec:
                    child.decompose()

        # Extract code blocks strictly belonging to this isolated section
        code_blocks = [pre.get_text().rstrip() for pre in root.find_all("pre") if pre.get_text().strip()]

        # Extract version directives strictly belonging to this isolated section
        version_directives = self._extract_version_directives(root)

        # Extract clean prose without code blocks
        for pre in root.find_all("pre"):
            pre.replace_with(" ")
        prose_text = " ".join(root.get_text().split())

        return DocumentUnit(
            unit_id=f"sec_{section_id}",
            unit_type="prose_section",
            breadcrumbs=breadcrumbs,
            prose_text=prose_text,
            code_blocks=code_blocks,
            version_directives=version_directives,
        )

    def _extract_version_directives(self, tag: Tag) -> List[VersionDirective]:
        """Detect and structure version tags like Added in 3.12, Changed in 3.12, etc."""
        directives: List[VersionDirective] = []
        nodes = tag.find_all("div", class_=re.compile(r"version(added|changed)|deprecated"))
        for node in nodes:
            raw_text = node.get_text(" ", strip=True)
            match = self.VERSION_REGEX.search(raw_text)
            if match:
                directive_type = match.group(1).lower()
                target_version = match.group(2)
                directives.append(
                    VersionDirective(
                        directive_type=directive_type,
                        target_version=target_version,
                        raw_text=raw_text,
                    )
                )
        return directives

    def parse_corpus(
        self,
        html_dir: Path,
        output_file: Path,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Batch-process all eligible HTML documents into a structured JSONL dataset."""
        from tqdm import tqdm

        output_file.parent.mkdir(parents=True, exist_ok=True)

        html_files = sorted(list(html_dir.rglob("*.html")))
        eligible_files = [f for f in html_files if self.is_file_eligible(f.relative_to(html_dir))]

        if limit:
            eligible_files = eligible_files[:limit]

        stats = {
            "total_files_scanned": len(html_files),
            "eligible_files": len(eligible_files),
            "successfully_parsed": 0,
            "total_units": 0,
            "api_units": 0,
            "prose_units": 0,
            "code_blocks": 0,
            "version_directives": 0,
        }

        print(f"\n🚀 Starting batch ingestion:")
        print(f"• Total HTML files scanned: {stats['total_files_scanned']}")
        print(f"• Eligible files selected:  {stats['eligible_files']}")
        print(f"• Output destination:       {output_file}\n")

        with output_file.open("w", encoding="utf-8") as out_f:
            for file_path in tqdm(eligible_files, desc="Parsing Corpus", unit="file"):
                try:
                    doc = self.parse_file(file_path, base_dir=html_dir)
                    if not doc or not doc.units:
                        continue

                    stats["successfully_parsed"] += 1
                    stats["total_units"] += len(doc.units)

                    for u in doc.units:
                        if u.unit_type == "api_symbol":
                            stats["api_units"] += 1
                        else:
                            stats["prose_units"] += 1
                        stats["code_blocks"] += len(u.code_blocks)
                        stats["version_directives"] += len(u.version_directives)

                    out_f.write(json.dumps(doc.to_dict(), ensure_ascii=False) + "\n")
                except Exception as e:
                    print(f"⚠️ Error parsing {file_path.name}: {e}")

        return stats


if __name__ == "__main__":
    parser = PythonDocsParser()
    raw_dir = config.version.raw_html_dir
    output_path = config.paths.processed_data_dir / "python-3.12" / "parsed_documents.jsonl"

    stats = parser.parse_corpus(html_dir=raw_dir, output_file=output_path)

    print("\n✅ Batch Corpus Ingestion Complete!")
    print(f"• Files Parsed:          {stats['successfully_parsed']}/{stats['eligible_files']}")
    print(f"• Total Units Extracted: {stats['total_units']:,}")
    print(f"  ├─ API Symbols:        {stats['api_units']:,}")
    print(f"  └─ Prose Sections:     {stats['prose_units']:,}")
    print(f"• Code Blocks Preserved: {stats['code_blocks']:,}")
    print(f"• Version Directives:    {stats['version_directives']:,}")
    print(f"• Saved To:              {output_path}")
