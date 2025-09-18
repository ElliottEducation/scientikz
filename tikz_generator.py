# tikz_generator.py
# Core generation logic for scientikz MVP with category + subcategory hints
# English-only comments and consistent formatting

import re
from dataclasses import dataclass
from typing import Literal, Tuple, Optional

Category = Literal["Math", "Physics", "Statistics", "Chemistry"]

@dataclass
class GenerationResult:
    latex: str
    category: Category
    summary: str

# ---------------------------
# Category detection (simple heuristics)
# ---------------------------
def detect_category(prompt: str) -> Category:
    p = prompt.lower()
    if any(k in p for k in ["molecule", "chemfig", "compound", "chemistry"]):
        return "Chemistry"
    if any(k in p for k in ["circuit", "inclined plane", "forces", "battery", "resistor", "electric"]):
        return "Physics"
    if any(k in p for k in ["scatter", "histogram", "normal", "regression", "statistics"]):
        return "Statistics"
    return "Math"

# ---------------------------
# Document wrapper
# ---------------------------
def wrap_as_document(body: str, needs_pgfplots: bool, needs_circuitikz: bool, needs_chemfig: bool) -> str:
    pkgs = [
        r"\usepackage{tikz}",
        r"\usepackage{siunitx}",
    ]
    if needs_pgfplots:
        pkgs.append(r"\usepackage{pgfplots}")
        pkgs.append(r"\pgfplotsset{compat=1.18}")
    if needs_circuitikz:
        pkgs.append(r"\usepackage[europeanresistors]{circuitikz}")
    if needs_chemfig:
        pkgs.append(r"\usepackage{chemfig}")

    pre = "\n".join(pkgs)
    return rf"""\documentclass[11pt]{{article}}
{pre}
\usepackage[margin=1in]{{geometry}}

\begin{{document}}

\section*{{scientikz output}}
{body}

\end{{document}}
"""

# ---------------------------
# Math generators
# ---------------------------
def gen_math_plot(prompt: str) -> Tuple[str, bool]:
    """
    Supports:
      - 'Plot y = sin(x) from -6.28 to 6.28'
      - 'Plot y = exp(-x^2) from -3 to 3'
      - 'Draw vector v = (3,2)'
    """
    expr = None
    x_min, x_max = -6.28, 6.28
    m = re.search(r"plot\s*y\s*=\s*([^;]+?)(?:\s+from\s+([-\d\.]+)\s+to\s+([-\d\.]+))?$", prompt, re.IGNORECASE)
    if m:
        expr = m.group(1).strip()
        if m.group(2) and m.group(3):
            try:
                x_min = float(m.group(2)); x_max = float(m.group(3))
            except:
                pass

    v = re.search(r"(draw\s+)?vector\s+\w*\s*=\s*\(\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*\)", prompt, re.IGNORECASE)
    if v and not expr:
        vx = v.group(2)
        vy = v.group(3)
        body = rf"""
\begin{{tikzpicture}}[scale=1.1]
  \draw[->] (-0.5,0) -- (4.5,0) node[right] {{$x$}};
  \draw[->] (0,-0.5) -- (0,4.5) node[above] {{$y$}};
  \draw[->, ultra thick] (0,0) -- ({vx},{vy}) node[above right] {{$\vec v$}};
  \draw[dashed] ({vx},0) -- ({vx},{vy});
  \draw[dashed] (0,{vy}) -- ({vx},{vy});
\end{{tikzpicture}}
"""
        return body, False

    if not expr:
        expr = "sin(deg(x))"

    expr_norm = (
        expr.replace("sin(x)", "sin(deg(x))")
            .replace("cos(x)", "cos(deg(x))")
            .replace("**", "^")  # keep caret
    )

    body = rf"""
\begin{{tikzpicture}}
\begin{{axis}}[
  axis lines=middle,
  xlabel=$x$,
  ylabel=$y$,
  domain={x_min}:{x_max},
  samples=300,
  grid=both
]
\addplot {{ {expr_norm} }};
\end{{axis}}
\end{{tikzpicture}}
"""
    return body, True

# ---------------------------
# Physics generators
# ---------------------------
def gen_inclined_plane(prompt: str) -> str:
    ang = 30
    m = re.search(r"angle\s*=\s*([-\d\.]+)", prompt, re.IGNORECASE)
    if m:
        try:
            ang = float(m.group(1))
        except:
            pass

    show_friction = "friction" in prompt.lower()
    body = rf"""
\begin{{tikzpicture}}[scale=1.1]
  \def\ang{{{ang}}}
  \draw[rotate=\ang] (0,0) -- (6,0);
  \draw[rotate=\ang] (0,0) -- (0,2.5);
  \filldraw[rotate=\ang, fill=gray!20] (2.5,0.2) rectangle (3.7,1.1);
  \draw[->, thick] (3.1,1.1) -- ++(0,-2) node[below] {{$mg$}};
  \draw[->, thick, rotate=\ang] (3.1,1.1) -- ++(0,1.6) node[above] {{$N$}};
  {"\\draw[->, thick, rotate=\\ang] (3.7,0.65) -- ++(0.9,0) node[right] {$f$};" if show_friction else ""}
  \draw (0,0) ++(0.9,0) arc (0:\ang:0.9);
  \node at ({0.9*0.5},{0.28*0.5}) {{$\\theta={ang}^\\circ$}};
\end{{tikzpicture}}
"""
    return body

def gen_dc_circuit(prompt: str) -> str:
    R = 100
    V = 9
    mR = re.search(r"R\s*=\s*([-\d\.]+)", prompt, re.IGNORECASE)
    mV = re.search(r"V\s*=\s*([-\d\.]+)", prompt, re.IGNORECASE)
    if mR:
        try:
            R = float(mR.group(1))
        except:
            pass
    if mV:
        try:
            V = float(mV.group(1))
        except:
            pass

    body = rf"""
\begin{{circuitikz}}[european]
  \draw (0,0)
    to[battery1,l_={V}\\,V] (0,3)
    to[R,l_={R}\\,$\\Omega$] (4,3)
    -- (4,0) -- (0,0);
\end{{circuitikz}}
"""
    return body

def gen_physics(prompt: str, subcategory: Optional[str]) -> Tuple[str, bool, bool]:
    p = prompt.lower()
    needs_pgfplots = False
    needs_circuitikz = False

    sub = (subcategory or "").lower()
    if "inclined plane" in p or "slope" in p or "classical" in sub:
        body = gen_inclined_plane(prompt)
    elif "circuit" in p or "battery" in p or "resistor" in p or "electromagnetism" in sub:
        body = gen_dc_circuit(prompt)
        needs_circuitikz = True
    else:
        # simple generic vector diagram (also used for quantum placeholder)
        body = r"""
\begin{tikzpicture}[scale=1.1]
  \draw[->] (-0.5,0) -- (4.5,0) node[right] {$x$};
  \draw[->] (0,-0.5) -- (0,4.5) node[above] {$y$};
  \draw[->, very thick] (0,0) -- (3,1.5) node[above right] {$\vec F$};
\end{tikzpicture}
"""
    return body, needs_pgfplots, needs_circuitikz

# ---------------------------
# Statistics generators
# ---------------------------
def parse_list_numbers(s: str):
    s = s.strip().strip("[]")
    items = [e.strip() for e in s.split(",")]
    out = []
    for it in items:
        try:
            out.append(float(it))
        except:
            pass
    return out

def gen_statistics(prompt: str, subcategory: Optional[str]) -> Tuple[str, bool]:
    p = prompt.lower()

    if "scatter" in p or (subcategory or "").lower() in ["descriptive", "regression"]:
        mx = re.search(r"x\s*=\s*(\[[^\]]+\])", prompt, re.IGNORECASE)
        my = re.search(r"y\s*=\s*(\[[^\]]+\])", prompt, re.IGNORECASE)
        xs = parse_list_numbers(mx.group(1)) if mx else [1, 2, 3]
        ys = parse_list_numbers(my.group(1)) if my else [2, 3, 5]
        pairs = "\n".join([f"{a}\t{b}" for a, b in zip(xs, ys)])

        # basic scatter; regression line could be added later as Pro feature
        body = rf"""
\begin{{tikzpicture}}
\begin{{axis}}[
  axis lines=left,
  xlabel=$x$,
  ylabel=$y$,
  grid=both
]
\addplot[
  only marks,
  mark=*,
]
table[row sep=crcr] {{
{pairs}
}};
\end{{axis}}
\end{{tikzpicture}}
"""
        return body, True

    # normal curve for inference
    m_mu = re.search(r"mu\s*=\s*([-\d\.]+)", prompt, re.IGNORECASE)
    m_si = re.search(r"sigma\s*=\s*([-\d\.]+)", prompt, re.IGNORECASE)
    m_rng = re.search(r"from\s+([-\d\.]+)\s+to\s+([-\d\.]+)", prompt, re.IGNORECASE)
    mu = float(m_mu.group(1)) if m_mu else 0.0
    si = float(m_si.group(1)) if m_si else 1.0
    x0, x1 = (-4.0, 4.0)
    if m_rng:
        try:
            x0 = float(m_rng.group(1)); x1 = float(m_rng.group(2))
        except:
            pass

    body = rf"""
\begin{{tikzpicture}}
\begin{{axis}}[
  axis lines=middle,
  xlabel=$x$,
  ylabel=$f(x)$,
  domain={x0}:{x1},
  samples=300,
  grid=both
]
\addplot {{
  1/({si}*sqrt(2*pi)) * exp(-0.5*((x-{mu})/{si})^2)
}};
\end{{axis}}
\end{{tikzpicture}}
"""
    return body, True

# ---------------------------
# Chemistry generators
# ---------------------------
def gen_chemistry(prompt: str) -> Tuple[str, bool]:
    m = re.search(r"molecule\s*:\s*([A-Za-z0-9\-\+\(\)]+)", prompt, re.IGNORECASE)
    mol = m.group(1) if m else "H2O"
    body = rf"""
\centering
\chemfig{{{mol}}}
"""
    return body, False

# ---------------------------
# Dispatcher
# ---------------------------
def generate_document(prompt: str, category_hint: str, subcategory_hint: Optional[str] = None) -> GenerationResult:
    """
    category_hint: "Math" | "Physics" | "Statistics" | "Chemistry" | "Auto"
    subcategory_hint: optional string; used as soft guidance for templates
    """
    cat = detect_category(prompt) if category_hint == "Auto" else category_hint
    needs_pgfplots = False
    needs_circuitikz = False
    needs_chemfig = False

    if cat == "Math":
        body, needs_pgfplots = gen_math_plot(prompt)
    elif cat == "Physics":
        body, needs_pgfplots, needs_circuitikz = gen_physics(prompt, subcategory_hint)
    elif cat == "Statistics":
        body, needs_pgfplots = gen_statistics(prompt, subcategory_hint)
    elif cat == "Chemistry":
        body, _ = gen_chemistry(prompt)
        needs_chemfig = True
    else:
        body, needs_pgfplots = gen_math_plot(prompt)

    doc = wrap_as_document(body, needs_pgfplots, needs_circuitikz, needs_chemfig)
    summary = "Generated LaTeX document with TikZ content."
    return GenerationResult(latex=doc, category=cat, summary=summary)
