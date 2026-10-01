import hashlib
import re
from dataclasses import dataclass

import tiktoken
from markdown_it import MarkdownIt

from app.ingestion.parse import ParsedDocument, ParsedSection

DEFAULT_MAX_TOKENS = 400
DEFAULT_OVERLAP_TOKENS = 40
ENCODING = tiktoken.get_encoding("cl100k_base")
MARKDOWN = MarkdownIt("commonmark")
BLANK_LINES = re.compile(r"\n\s*\n")
WORD_ENDS = re.compile(r"\S+\s*")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    document_hash: str
    source: str
    version: str
    title: str
    section_path: tuple[str, ...]
    source_url: str
    ordinal: int
    text: str
    token_count: int


@dataclass(frozen=True)
class _Atom:
    text: str
    code: bool


def chunk_document(
    document: ParsedDocument,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[Chunk]:
    if max_tokens <= 0 or overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("Expected max_tokens > 0 and 0 <= overlap_tokens < max_tokens")

    chunks: list[Chunk] = []
    for section in document.sections:
        for text in _chunk_section(section, max_tokens, overlap_tokens):
            ordinal = len(chunks)
            digest = hashlib.sha256(
                f"{document.document_id}\0{ordinal}\0{text}".encode()
            ).hexdigest()[:20]
            chunks.append(
                Chunk(
                    chunk_id=f"{document.document_id}:{digest}",
                    document_id=document.document_id,
                    document_hash=document.content_hash,
                    source=document.source,
                    version=document.version,
                    title=document.title,
                    section_path=section.path,
                    source_url=section.source_url,
                    ordinal=ordinal,
                    text=text,
                    token_count=_tokens(text),
                )
            )
    return chunks


def _chunk_section(section: ParsedSection, max_tokens: int, overlap_tokens: int) -> list[str]:
    result: list[str] = []
    current: list[_Atom] = []
    previous_prose = ""

    def flush() -> None:
        nonlocal previous_prose
        if not current:
            return
        result.append(_join(current))
        previous_prose = "\n\n".join(atom.text for atom in current if not atom.code)
        current.clear()

    for atom in _section_atoms(section.text, max_tokens):
        if current and _tokens(_join([*current, atom])) > max_tokens:
            flush()
        if not current and not atom.code and previous_prose and overlap_tokens:
            prefix = _suffix(previous_prose, atom.text, overlap_tokens, max_tokens)
            if prefix:
                current.append(_Atom(prefix, False))
        current.append(atom)
        if atom.code and _tokens(_join(current)) > max_tokens:
            # A single code block is kept intact even when it exceeds the cap.
            flush()
    flush()
    return result


def _section_atoms(text: str, max_tokens: int) -> list[_Atom]:
    lines = text.splitlines()
    code_ranges = [
        (token.map[0], token.map[1], token)
        for token in MARKDOWN.parse(text)
        if token.type in {"fence", "code_block"} and token.map is not None
    ]
    atoms: list[_Atom] = []
    start = 0
    for code_start, code_end, token in code_ranges:
        atoms.extend(_prose_atoms("\n".join(lines[start:code_start]), max_tokens))
        if token.type == "code_block":
            code = f"```\n{token.content.rstrip()}\n```"
        else:
            code = "\n".join(lines[code_start:code_end]).strip()
        atoms.append(_Atom(code, True))
        start = code_end
    atoms.extend(_prose_atoms("\n".join(lines[start:]), max_tokens))
    return atoms


def _prose_atoms(text: str, max_tokens: int) -> list[_Atom]:
    atoms: list[_Atom] = []
    for paragraph in BLANK_LINES.split(text.strip()):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if _tokens(paragraph) <= max_tokens:
            atoms.append(_Atom(paragraph, False))
        else:
            atoms.extend(_Atom(part, False) for part in _split_long_prose(paragraph, max_tokens))
    return atoms


def _split_long_prose(text: str, max_tokens: int) -> list[str]:
    parts: list[str] = []
    current = ""
    for match in WORD_ENDS.finditer(text):
        word = match.group()
        candidate = current + word
        if current and _tokens(candidate.strip()) > max_tokens:
            parts.append(current.strip())
            current = ""
        if _tokens(word.strip()) > max_tokens:
            parts.extend(_split_long_word(word.strip(), max_tokens))
        else:
            current += word
    if current.strip():
        parts.append(current.strip())
    return parts


def _split_long_word(word: str, max_tokens: int) -> list[str]:
    parts: list[str] = []
    while word:
        end = len(word)
        while _tokens(word[:end]) > max_tokens:
            end //= 2
            if end == 0:
                raise ValueError("Token cap is too small for a single Unicode character")
        # Grow to the largest character boundary that fits the token cap.
        while end < len(word) and _tokens(word[: end + 1]) <= max_tokens:
            end += 1
        parts.append(word[:end])
        word = word[end:]
    return parts


def _suffix(text: str, next_text: str, overlap_tokens: int, max_tokens: int) -> str:
    words = text.split()
    suffix: list[str] = []
    for word in reversed(words):
        candidate = " ".join([word, *suffix])
        if (
            _tokens(candidate) > overlap_tokens
            or _tokens(f"{candidate}\n\n{next_text}") > max_tokens
        ):
            break
        suffix.insert(0, word)
    return " ".join(suffix)


def _join(atoms: list[_Atom]) -> str:
    return "\n\n".join(atom.text for atom in atoms)


def _tokens(text: str) -> int:
    return len(ENCODING.encode(text))
