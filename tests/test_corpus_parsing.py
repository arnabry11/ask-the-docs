import json
from pathlib import Path

import pytest

from app.ingestion.chunk import chunk_document
from app.ingestion.download import download_corpus
from app.ingestion.parse import ParsedDocument, ParsedSection, parse_document
from app.ingestion.prepare import prepare_corpus
from app.ingestion.sources import SourceSpec


def rails_spec() -> SourceSpec:
    return SourceSpec(
        document_id="rails:example",
        source="rails",
        version="8.1.4",
        format="markdown",
        fetch_url="https://example.test/example.md",
        source_url="https://guides.rubyonrails.org/v8.1/example.html",
        relative_path="rails/example.md",
    )


def postgres_spec() -> SourceSpec:
    return SourceSpec(
        document_id="postgresql:example",
        source="postgresql",
        version="16",
        format="html",
        fetch_url="https://example.test/example.html",
        source_url="https://www.postgresql.org/docs/16/example.html",
        relative_path="postgresql/example.html",
    )


def test_rails_parser_preserves_heading_path_code_and_link() -> None:
    content = """Do not read this file on GitHub.

Example Guide
=============

Introduction.

## Querying Records

Use `where`.

### Filtering

```ruby
User.where(active: true)
```
"""

    document = parse_document(rails_spec(), content)

    assert document.title == "Example Guide"
    assert [section.path for section in document.sections] == [
        ("Example Guide",),
        ("Example Guide", "Querying Records"),
        ("Example Guide", "Querying Records", "Filtering"),
    ]
    assert document.sections[1].source_url.endswith("#querying-records")
    assert "```ruby\nUser.where(active: true)\n```" in document.sections[2].text
    assert "Do not read" not in " ".join(section.text for section in document.sections)


def test_postgresql_parser_removes_navigation_and_preserves_preformatted_code() -> None:
    content = """<div id="docContent">
      <div class="navheader">Previous page</div>
      <div class="sect1" id="USING-EXPLAIN">
        <h1>Using EXPLAIN</h1><p>Read query plans.</p>
        <div class="toc">Table of Contents</div>
        <div class="sect2" id="USING-EXPLAIN-BASICS">
          <h2>Basics</h2><p>Inspect a <a href="sql-explain.html">plan</a>.</p>
          <pre>EXPLAIN SELECT 1;\n</pre>
        </div>
      </div>
    </div>"""

    document = parse_document(postgres_spec(), content)

    assert [section.path for section in document.sections] == [
        ("Using EXPLAIN",),
        ("Using EXPLAIN", "Basics"),
    ]
    assert document.sections[1].source_url.endswith("#USING-EXPLAIN-BASICS")
    assert (
        "[plan](https://www.postgresql.org/docs/16/sql-explain.html)" in document.sections[1].text
    )
    assert "```\nEXPLAIN SELECT 1;\n```" in document.sections[1].text
    assert "Previous page" not in " ".join(section.text for section in document.sections)
    assert "Table of Contents" not in " ".join(section.text for section in document.sections)


def test_chunker_caps_prose_and_overlaps_without_splitting_code() -> None:
    prose = " ".join(f"word{i}" for i in range(70))
    code = "```sql\n" + "SELECT some_long_column FROM records;\n" * 5 + "```"
    document = ParsedDocument(
        document_id="rails:example",
        source="rails",
        version="8.1.4",
        title="Example",
        source_url="https://example.test/example",
        content_hash="a" * 64,
        sections=(
            ParsedSection(
                ("Example", "Section"), "https://example.test/example#section", f"{prose}\n\n{code}"
            ),
        ),
    )

    chunks = chunk_document(document, max_tokens=25, overlap_tokens=5)
    code_chunks = [chunk for chunk in chunks if "```sql" in chunk.text]

    assert len(chunks) > 2
    assert len(code_chunks) == 1
    assert code_chunks[0].text == code
    assert code_chunks[0].token_count > 25
    assert all(chunk.token_count <= 25 for chunk in chunks if chunk not in code_chunks)
    assert chunks[0].section_path == ("Example", "Section")
    assert chunks[0].source_url.endswith("#section")
    assert chunks[0].ordinal == 0
    assert chunks == chunk_document(document, max_tokens=25, overlap_tokens=5)
    assert any(
        set(first.text.split()[-3:]) & set(second.text.split()[:5])
        for first, second in zip(chunks, chunks[1:], strict=False)
    )


def test_prepare_is_repeatable_and_rejects_tampered_input(tmp_path: Path) -> None:
    spec = rails_spec()
    raw = b"Example\n=======\n\n## Querying\n\nUse an index.\n"

    class FakeClient:
        def fetch(self, url: str) -> bytes:
            assert url == spec.fetch_url
            return raw

    input_dir = tmp_path / "raw"
    output = tmp_path / "chunks.jsonl"
    download_corpus([spec], input_dir, FakeClient())

    first = prepare_corpus([spec], input_dir, output, max_tokens=30, overlap_tokens=3)
    before = output.read_bytes()
    second = prepare_corpus([spec], input_dir, output, max_tokens=30, overlap_tokens=3)
    rows = [json.loads(line) for line in before.splitlines()]

    assert first == second
    assert (first.documents, first.sections, first.chunks) == (1, 1, 1)
    assert output.read_bytes() == before
    assert rows[0]["section_path"] == ["Example", "Querying"]
    assert rows[0]["text"] == "Use an index."

    (input_dir / spec.relative_path).write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="integrity check"):
        prepare_corpus([spec], input_dir, output)
    assert output.read_bytes() == before


def test_invalid_chunk_settings_fail() -> None:
    document = parse_document(rails_spec(), "Example\n=======\n\nContent.\n")
    with pytest.raises(ValueError, match="max_tokens"):
        chunk_document(document, max_tokens=10, overlap_tokens=10)
