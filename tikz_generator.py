# tikz_generator.py
# Core generator for scientikz

import re
import math
from dataclasses import dataclass
from typing import Literal, Tuple, Optional

Category = Literal["Math","Physics","Statistics","Chemistry"]

@dataclass
class GenerationResult:
    latex: str
    category: Category
    summary: str

# Utility format
def _fmt(x) -> str:
    try:
        xv = float(x)
    except:
        return str(x)
    s = f"{xv:.3f}".rstrip("0").rstrip(".")
    return "0" if s == "-0" else s

# Detect category
def detect_category(prompt: str) -> Category:
    p = prompt.lower()
    if "stats:" in p or "scatter" in p or "hist" in p:
        return "Statistics"
    if "physics:" in p or "circuit" in p or "force" in p:
        return "Physics"
    if "chem:" in p or "molecule" in p or "chemfig" in p:
        return "Chemistry"
    return "Math"

# ---------------------------
# Math simple plot
# ---------------------------
def gen_math_plot(prompt: str) -> Tuple[str,bool]:
    m = re.search(r"plot\s*y\s*=\s*([^;]+?)\s+from\s+([-\d\.]+)\s+to\s+([-\d\.]+)", prompt, re.IGNORECASE)
    if m:
        expr = m.group(1).strip()
        x0, x1 = m.group(2), m.group(3)
    else:
        expr = "sin(deg(x))"
        x0, x1 = "-6.28","6.28"
    body = rf"""
\begin{{tikzpicture}}
\begin{{axis}}[axis lines=middle,xlabel=$x$,ylabel=$y$,domain={x0}:{x1},samples=200,grid=both]
\addplot {{{expr}}};
\end{{axis}}
\end{{tikzpicture}}
"""
    return body, True

# ---------------------------
# Statistics generators
# ---------------------------
def _parse_num_list(txt: str):
    txt = txt.strip().strip("[]")
    out = []
    for t in txt.split(","):
        t = t.strip()
        try:
            out.append(float(t))
        except:
            pass
    return out

def gen_statistics_directive(prompt: str) -> Tuple[str,bool]:
    p = prompt.lower()
    if "mode=scatter" in p:
        mx = re.search(r"x=\[([^\]]+)\]", prompt)
        my = re.search(r"y=\[([^\]]+)\]", prompt)
        xs = _parse_num_list(mx.group(1)) if mx else [1,2,3]
        ys = _parse_num_list(my.group(1)) if my else [2,3,5]
        pts = "\n".join([f"{_fmt(a)}\t{_fmt(b)}" for a,b in zip(xs,ys)])
        body = rf"""
\begin{{tikzpicture}}
\begin{{axis}}[axis lines=left,xlabel=$x$,ylabel=$y$,
