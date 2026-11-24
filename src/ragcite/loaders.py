"""Turn files on disk into `(Document, [Page])`.

The only interesting requirement here: *keep the page boundaries*. A citation
that says "p.3" is only worth something if page 3 is where the text actually
came from, so pagination is preserved from the loader all the way to the
chunker instead of being reconstructed later.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path

from ragcite.types import Document, Page

TEXT_SUFFIXES = {".md", ".txt", ".rst", ".markdown", ".text"}
PDF_SUFFIXES = {".pdf"}
SUPPORTED = TEXT_SUFFIXES | PDF_SUFFIXES


class UnsupportedFile(Exception):
    pass


def iter_files(paths: list[str] | list[Path], include: tuple[str, ...]) -> Iterator[Path]:
    """Expand files and directories into a sorted, de-duplicated file list."""
    seen: set[Path] = set()
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            found: list[Path] = []
            for pattern in include:
                found.extend(p.rglob(pattern))
            candidates = sorted(found)
        elif p.is_file():
            candidates = [p]
        else:
            raise FileNotFoundError(f"no such file or directory: {p}")
        for c in candidates:
            rp = c.resolve()
            if rp in seen or c.name.startswith("."):
                continue
            seen.add(rp)
            yield c


def load_file(path: str | Path) -> tuple[Document, list[Page]]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in PDF_SUFFIXES:
        pages = _load_pdf(path)
    elif suffix in TEXT_SUFFIXES:
        pages = _load_text(path)
    else:
        raise UnsupportedFile(
            f"{path}: unsupported extension {suffix!r} (supported: {sorted(SUPPORTED)})"
        )

    raw = path.read_bytes()
    content_hash = hashlib.sha256(raw).hexdigest()
    doc_id = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:16]
    return (
        Document(
            doc_id=doc_id,
            path=str(path),
            title=_title_for(path, pages),
            n_pages=len([p for p in pages if p.number is not None]),
            content_hash=content_hash,
            meta={"suffix": suffix, "bytes": len(raw)},
        ),
        pages,
    )


def _load_text(path: Path) -> list[Page]:
    text = path.read_text(encoding="utf-8", errors="replace")
    # Unpaginated formats get a single page with number=None; the chunker then
    # records line numbers instead so citations stay precise.
    return [Page(number=None, text=text)]


def _load_pdf(path: Path) -> list[Page]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise ImportError("pypdf is required to read PDFs: pip install pypdf") from exc

    reader = PdfReader(str(path))
    pages: list[Page] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:  # noqa: BLE001 - a broken page shouldn't kill the ingest
            text = ""
        pages.append(Page(number=i, text=text))
    if not any(p.text.strip() for p in pages):
        raise UnsupportedFile(
            f"{path}: no extractable text (scanned PDF?). OCR it first -- see docs/03-pipeline.md"
        )
    return pages


def _title_for(path: Path, pages: list[Page]) -> str:
    """Prefer a leading markdown H1, fall back to the filename."""
    head = pages[0].text.lstrip() if pages else ""
    for line in head.splitlines()[:5]:
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return path.name
