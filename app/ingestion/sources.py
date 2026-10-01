import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG = REPO_ROOT / "corpus" / "sources.json"
RAILS_SLUG = re.compile(r"[a-z0-9_]+\Z")
POSTGRES_SLUG = re.compile(r"[a-z0-9-]+\Z")
COMMIT_SHA = re.compile(r"[0-9a-f]{40}\Z")


@dataclass(frozen=True)
class SourceSpec:
    document_id: str
    source: Literal["rails", "postgresql"]
    version: str
    format: Literal["markdown", "html"]
    fetch_url: str
    source_url: str
    relative_path: str


def load_source_specs(path: Path = DEFAULT_CATALOG) -> list[SourceSpec]:
    catalog = json.loads(path.read_text(encoding="utf-8"))
    if catalog["schema_version"] != 1:
        raise ValueError("Unsupported source catalog version")

    rails = catalog["rails"]
    commit = rails["commit"]
    if not COMMIT_SHA.fullmatch(commit):
        raise ValueError("Rails source must be pinned to a commit SHA")

    specs: list[SourceSpec] = []
    for slug in rails["guides"]:
        if not RAILS_SLUG.fullmatch(slug):
            raise ValueError(f"Invalid Rails guide slug: {slug}")
        specs.append(
            SourceSpec(
                document_id=f"rails:{slug}",
                source="rails",
                version=rails["version"],
                format="markdown",
                fetch_url=(
                    f"https://raw.githubusercontent.com/rails/rails/{commit}/guides/source/{slug}.md"
                ),
                source_url=f"https://guides.rubyonrails.org/v{rails['guide_series']}/{slug}.html",
                relative_path=f"rails/{slug}.md",
            )
        )

    postgres = catalog["postgresql"]
    for slug in postgres["pages"]:
        if not POSTGRES_SLUG.fullmatch(slug):
            raise ValueError(f"Invalid PostgreSQL page slug: {slug}")
        url = f"https://www.postgresql.org/docs/{postgres['version']}/{slug}.html"
        specs.append(
            SourceSpec(
                document_id=f"postgresql:{slug}",
                source="postgresql",
                version=postgres["version"],
                format="html",
                fetch_url=url,
                source_url=url,
                relative_path=f"postgresql/{slug}.html",
            )
        )

    if len({spec.document_id for spec in specs}) != len(specs):
        raise ValueError("Source catalog contains duplicate documents")
    return specs
