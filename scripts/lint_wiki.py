#!/usr/bin/env python3
"""Deterministic health check for the Obsidian project wiki."""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WIKI = ROOT / "wiki"
CONTENT_DIRS = ("concepts", "comparisons", "queries", "entities")
REQUIRED_FRONTMATTER = ("title:", "created:", "updated:", "type:", "tags:", "sources:", "confidence:")


def knowledge_pages() -> list[Path]:
    pages: list[Path] = []
    for directory in CONTENT_DIRS:
        path = WIKI / directory
        if path.exists():
            pages.extend(sorted(path.glob("*.md")))
    return pages


def main() -> int:
    errors: list[str] = []
    pages = knowledge_pages()
    all_notes = {path.stem: path for path in WIKI.rglob("*.md")}
    index = (WIKI / "index.md").read_text(encoding="utf-8")

    for page in pages:
        text = page.read_text(encoding="utf-8")
        if not text.startswith("---\n"):
            errors.append(f"missing frontmatter: {page.relative_to(WIKI)}")
            continue
        frontmatter = text.split("---", 2)[1]
        for field in REQUIRED_FRONTMATTER:
            if field not in frontmatter:
                errors.append(f"missing {field} {page.relative_to(WIKI)}")
        links = re.findall(r"\[\[([^\]|#]+)", text)
        if len(set(links)) < 2:
            errors.append(f"fewer than two outbound links: {page.relative_to(WIKI)}")
        for link in links:
            if Path(link).name not in all_notes:
                errors.append(f"broken link [[{link}]] in {page.relative_to(WIKI)}")
        if f"[[{page.stem}]]" not in index:
            errors.append(f"not indexed: {page.relative_to(WIKI)}")

    for raw in (WIKI / "raw").rglob("*.md"):
        text = raw.read_text(encoding="utf-8")
        parts = text.split("---", 2)
        if len(parts) != 3:
            errors.append(f"invalid raw frontmatter: {raw.relative_to(WIKI)}")
            continue
        match = re.search(r"^sha256:\s*([0-9a-f]{64})\s*$", parts[1], re.MULTILINE)
        if not match:
            errors.append(f"missing raw sha256: {raw.relative_to(WIKI)}")
            continue
        body = parts[2].lstrip("\r\n")
        actual = hashlib.sha256(body.encode()).hexdigest()
        if actual != match.group(1):
            errors.append(f"raw source drift: {raw.relative_to(WIKI)}")

    if errors:
        print("WIKI LINT: FAIL")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"WIKI LINT: PASS ({len(pages)} knowledge pages, {len(all_notes)} total notes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
