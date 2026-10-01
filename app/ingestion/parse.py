import hashlib
import re
import unicodedata
from dataclasses import dataclass
from urllib.parse import urljoin

from bs4 import BeautifulSoup, Tag
from bs4.element import NavigableString, PageElement
from markdown_it import MarkdownIt
from markdown_it.token import Token

from app.ingestion.sources import SourceSpec

MARKDOWN = MarkdownIt("commonmark")
WHITESPACE = re.compile(r"\s+")
SLUG_CHARS = re.compile(r"[^a-z0-9]+")
HTML_BLOCKS = {"p", "pre", "li", "dt", "dd"}


@dataclass(frozen=True)
class ParsedSection:
    path: tuple[str, ...]
    source_url: str
    text: str


@dataclass(frozen=True)
class ParsedDocument:
    document_id: str
    source: str
    version: str
    title: str
    source_url: str
    content_hash: str
    sections: tuple[ParsedSection, ...]


@dataclass(frozen=True)
class _Heading:
    level: int
    title: str
    start_line: int
    end_line: int
    anchor: str | None


def parse_document(spec: SourceSpec, content: str) -> ParsedDocument:
    if spec.format == "markdown":
        title, sections = _parse_markdown(content, spec.source_url)
    else:
        title, sections = _parse_html(content, spec.source_url)
    if not sections:
        raise ValueError(f"No content sections found in {spec.document_id}")
    return ParsedDocument(
        document_id=spec.document_id,
        source=spec.source,
        version=spec.version,
        title=title,
        source_url=spec.source_url,
        content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        sections=tuple(sections),
    )


def _parse_markdown(content: str, source_url: str) -> tuple[str, list[ParsedSection]]:
    lines = content.splitlines()
    tokens = MARKDOWN.parse(content)
    headings: list[_Heading] = []
    seen_slugs: dict[str, int] = {}
    for index, token in enumerate(tokens):
        if token.type != "heading_open" or token.map is None:
            continue
        title = _inline_text(tokens[index + 1])
        level = int(token.tag[1:])
        slug = _slug(title)
        duplicate = seen_slugs.get(slug, 0)
        seen_slugs[slug] = duplicate + 1
        anchor = f"{slug}-{duplicate}" if duplicate else slug
        headings.append(
            _Heading(level, title, token.map[0], token.map[1], anchor if level > 1 else None)
        )

    if not headings:
        raise ValueError("Markdown document has no headings")
    title = headings[0].title
    sections: list[ParsedSection] = []
    stack: list[_Heading] = []
    for index, heading in enumerate(headings):
        while stack and stack[-1].level >= heading.level:
            stack.pop()
        stack.append(heading)
        next_start = headings[index + 1].start_line if index + 1 < len(headings) else len(lines)
        body = "\n".join(lines[heading.end_line : next_start]).strip()
        if body:
            url = f"{source_url}#{heading.anchor}" if heading.anchor else source_url
            sections.append(ParsedSection(tuple(item.title for item in stack), url, body))
    return title, sections


def _parse_html(content: str, source_url: str) -> tuple[str, list[ParsedSection]]:
    soup = BeautifulSoup(content, "html.parser")
    root = soup.select_one("#docContent")
    if root is None:
        raise ValueError("PostgreSQL page has no #docContent")
    for unwanted in root.select(".navheader, .navfooter, .toc, .indexterm, a.id_link"):
        unwanted.decompose()

    title = ""
    sections: list[ParsedSection] = []
    stack: list[tuple[int, str]] = []
    blocks: list[str] = []
    current_url = source_url

    def flush() -> None:
        if stack and blocks:
            sections.append(
                ParsedSection(tuple(name for _, name in stack), current_url, "\n\n".join(blocks))
            )
            blocks.clear()

    for node in root.descendants:
        if not isinstance(node, Tag):
            continue
        if node.name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            flush()
            level = int(node.name[1:])
            heading_text = _clean_text(node.get_text(" ", strip=True))
            if not title:
                title = heading_text
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading_text))
            anchor = _html_anchor(node, root)
            current_url = f"{source_url}#{anchor}" if anchor else source_url
        elif node.name in HTML_BLOCKS and stack:
            if node.name == "li" and node.find(["p", "pre", "li"]):
                continue
            if node.name in {"p", "pre"} and node.find_parent(["p", "pre"]):
                continue
            block = _html_block(node, source_url)
            if block:
                blocks.append(block)
    flush()
    if not title:
        raise ValueError("PostgreSQL page has no headings")
    return title, sections


def _inline_text(token: Token) -> str:
    children = token.children or []
    parts = [
        " " if child.type in {"softbreak", "hardbreak"} else child.content
        for child in children
        if child.type in {"text", "code_inline", "image", "softbreak", "hardbreak"}
    ]
    return _clean_text("".join(parts))


def _clean_text(value: str) -> str:
    return WHITESPACE.sub(" ", value).strip().removesuffix(" #")


def _slug(value: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return SLUG_CHARS.sub("-", ascii_text.lower()).strip("-")


def _html_anchor(heading: Tag, root: Tag) -> str | None:
    if heading.has_attr("id"):
        return str(heading["id"])
    for parent in heading.parents:
        if parent is root:
            break
        if isinstance(parent, Tag) and parent.has_attr("id"):
            return str(parent["id"])
    return None


def _html_block(node: Tag, source_url: str) -> str:
    if node.name == "pre":
        return f"```\n{node.get_text().rstrip()}\n```"
    return _clean_text("".join(_render_inline(child, source_url) for child in node.children))


def _render_inline(node: PageElement, source_url: str) -> str:
    if isinstance(node, NavigableString):
        return str(node)
    if not isinstance(node, Tag):
        return ""
    content = "".join(_render_inline(child, source_url) for child in node.children)
    if node.name == "a" and node.get("href"):
        return f"[{_clean_text(content)}]({urljoin(source_url, str(node['href']))})"
    if node.name in {"code", "tt"}:
        return f"`{content}`"
    if node.name == "br":
        return " "
    return content
