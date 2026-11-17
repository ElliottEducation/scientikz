"""Compile a standalone TikZ document to PDF / PNG / SVG with pdflatex + pdftocairo."""
from __future__ import annotations

import functools
import os
import subprocess
import tempfile


class RenderError(RuntimeError):
    pass


@functools.lru_cache(maxsize=256)
def render(doc: str, dpi: int = 220) -> dict[str, bytes]:
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "fig.tex"), "w", encoding="utf-8").write(doc)
        try:
            subprocess.run(["pdflatex", "-no-shell-escape", "-interaction=nonstopmode", "-halt-on-error", "fig.tex"],
                           cwd=d, capture_output=True, timeout=30, check=True)
        except subprocess.TimeoutExpired:
            raise RenderError("编译超时 / LaTeX timed out")
        except subprocess.CalledProcessError:
            log = open(os.path.join(d, "fig.log"), encoding="utf-8", errors="replace").read().splitlines()
            first = next((i for i, l in enumerate(log) if l.startswith("!")), None)
            raise RenderError("\n".join(log[first:first + 6]) if first is not None else "LaTeX error")
        pdf = os.path.join(d, "fig.pdf")
        subprocess.run(["pdftocairo", "-png", "-r", str(dpi), "-singlefile", pdf, os.path.join(d, "fig")],
                       check=True, timeout=30)
        subprocess.run(["pdftocairo", "-svg", pdf, os.path.join(d, "fig.svg")], check=True, timeout=30)
        return {ext: open(os.path.join(d, f"fig.{ext}"), "rb").read() for ext in ("pdf", "png", "svg")}
