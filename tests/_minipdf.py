"""A ~60-line PDF writer, used only by the tests.

Page-level citations are the headline claim of this project, so the test suite
should prove them against a real paginated PDF rather than a stand-in. Pulling
in reportlab just to generate fixtures would be a heavy dependency for one
test, so this emits the smallest valid PDF that pypdf can extract text from:
one content stream per page, Helvetica, one line per string.
"""

from __future__ import annotations

from pathlib import Path


def write_pdf(path: str | Path, pages: list[list[str]]) -> Path:
    """Write `pages` (a list of lines per page) to a minimal PDF."""
    path = Path(path)
    objects: list[bytes] = []

    font_obj = 3 + 2 * len(pages)  # pages occupy 3 .. 3+2n-1
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(len(pages)))

    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode())

    for i, lines in enumerate(pages):
        content = _content_stream(lines)
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 {font_obj} 0 R >> >> "
                f"/Contents {4 + 2 * i} 0 R >>"
            ).encode()
        )
        objects.append(
            f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"\nendstream"
        )

    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for n, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{n} 0 obj\n".encode() + body + b"\nendobj\n"

    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets[1:]:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n"
    ).encode()

    path.write_bytes(bytes(out))
    return path


def _content_stream(lines: list[str]) -> bytes:
    parts = [b"BT", b"/F1 12 Tf", b"72 720 Td", b"14 TL"]
    for line in lines:
        escaped = line.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        parts.append(f"({escaped}) Tj".encode())
        parts.append(b"T*")
    parts.append(b"ET")
    return b"\n".join(parts)
