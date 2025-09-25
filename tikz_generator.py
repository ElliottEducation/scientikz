# tikz_generator.py
# Core generator + English NL parsers for Math/Physics/Stats/Chemistry
import re
import math
from dataclasses import dataclass
from typing import Literal, Tuple, Optional, Dict, Any, List

Category = Literal["Math","Physics","Statistics","Chemistry"]

# ---------------- Errors ----------------
class ValidationError(Exception):
    pass

@dataclass
class GenerationResult:
    latex: str
    category: Category
    summary: str

# ---------- Utils ----------
def _fmt(x) -> str:
    try:
        xv = float(x)
    except:
        return str(x)
    s = f"{xv:.3f}".rstrip("0").rstrip(".")
    return "0" if s == "-0" else s

def _b(s: Optional[str]) -> bool:
    return bool(s) and s.strip().lower() in ["1","true","yes","y","on"]

def ensure_range(val: float, lo: float, hi: float, name: str):
    if not (lo <= val <= hi):
        raise ValidationError(f"[{name}] out of range: {val} not in [{lo},{hi}]")

def ensure_len_eq(xs: List[float], ys: List[float], ctx="pairs"):
    if len(xs) == 0 or len(xs) != len(ys):
        raise ValidationError(f"[{ctx}] x/y length mismatch or empty")

def preflight_latex(body: str):
    # simple environment balance
    pairs = [
        ("\\begin{tikzpicture}", "\\end{tikzpicture}"),
        ("\\begin{axis}", "\\end{axis}"),
        ("\\begin{circuitikz}", "\\end{circuitikz}")
    ]
    for b,e in pairs:
        if body.count(b) != body.count(e):
            raise ValidationError(f"LaTeX env not balanced: {b} vs {e}")
    if body.count("{") != body.count("}"):
        raise ValidationError("Curly braces not balanced")
    if len(body) > 200_000:
        raise ValidationError("LaTeX body too large (>200k chars)")

# ---------- Fallback detection (for directive path) ----------
def detect_category(prompt: str) -> Category:
    p = prompt.lower()
    if any(k in p for k in ["mode=scatter","mode=hist","mode=regression","normal curve"]):
        return "Statistics"
    if any(k in p for k in ["vectoranalysis:","mode=incline","mode=projectile","mode=circuit","inclined","circuit"]):
        return "Physics" if "mode=" in p and "circuit" in p or "projectile" in p or "incline" in p else "Math"
    if any(k in p for k in ["mode=benzene","mode=raw","chemfig","molecule"]):
        return "Chemistry"
    return "Math"

# ---------- Document wrapper ----------
def _wrap_document(body: str, needs_pgfplots: bool, needs_circuitikz: bool, needs_chemfig: bool) -> str:
    pre = ["\\documentclass{standalone}", "\\usepackage{tikz}"]
    if needs_pgfplots:
        pre += ["\\usepackage{pgfplots}", "\\pgfplotsset{compat=1.18}"]
    if needs_circuitikz:
        pre += ["\\usepackage[europeanresistors]{circuitikz}"]
    if needs_chemfig:
        pre += ["\\usepackage{chemfig}"]
    preamble = "\n".join(pre)
    full = preamble + "\n\\begin{document}\n" + body + "\n\\end{document}\n"
    return full

# ===================== Renderers (IR → TikZ) =====================

# ---- Math (function plot) ----
def render_math_function(ir: Dict[str, Any]) -> Tuple[str,bool]:
    expr = ir.get("expr", "sin(deg(x))")
    x0 = ir.get("x0", -6.28)
    x1 = ir.get("x1", 6.28)
    body = rf"""
\begin{{tikzpicture}}
\begin{{axis}}[axis lines=middle, xlabel=$x$, ylabel=$y$, domain={_fmt(x0)}:{_fmt(x1)}, samples=300, grid=both]
\addplot {{{expr}}};
\end{{axis}}
\end{{tikzpicture}}
"""
    preflight_latex(body)
    return body, True

# ---- Vector Analysis (kept as directive style) ----
def render_vector_analysis_from_directive(prompt: str) -> Tuple[str,bool]:
    # Minimal extraction; keep previous behavior for directives
    # (Omitted for brevity: full vector rendering present earlier version)
    # For NL MVP we focus on function plot; you can still use directive UI for vectors.
    return "% VectorAnalysis UI uses directive path; keep your previous version here.", False

# ---- Statistics ----
def _parse_num_list(txt: str):
    txt = txt.strip().strip("[]")
    if not txt:
        return []
    out = []
    for t in txt.split(","):
        t = t.strip()
        try:
            out.append(float(t))
        except:
            pass
    return out

def render_stats_scatter(ir: Dict[str, Any]) -> Tuple[str,bool]:
    xs, ys = ir["x"], ir["y"]
    ensure_len_eq(xs, ys, "scatter")
    pairs = "\n".join(f"{_fmt(a)}\t{_fmt(b)}" for a,b in zip(xs,ys))
    body = rf"""
\begin{{tikzpicture}}
\begin{{axis}}[axis lines=left, xlabel=$x$, ylabel=$y$, grid=both]
\addplot[only marks, mark=*] table[row sep=crcr]{{
{pairs}
}};
\end{{axis}}
\end{{tikzpicture}}
"""
    preflight_latex(body)
    return body, True

def render_stats_hist(ir: Dict[str, Any]) -> Tuple[str,bool]:
    data = ir["data"]
    bins = int(ir.get("bins", 10))
    ensure_range(bins, 1, 200, "bins")
    lines = "\n".join(_fmt(v) for v in data)
    body = rf"""
\begin{{tikzpicture}}
\begin{{axis}}[
  ybar interval,
  ymin=0,
  xlabel=Value,
  ylabel=Count,
  grid=both
]
\addplot+[
  hist={{bins={bins}}},
]
table[row sep=\\\\, y index=0]{{
{lines}
}};
\end{{axis}}
\end{{tikzpicture}}
"""
    preflight_latex(body)
    return body, True

def render_stats_regression(ir: Dict[str, Any]) -> Tuple[str,bool]:
    xs, ys = ir["x"], ir["y"]
    ensure_len_eq(xs, ys, "regression")
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x*x for x in xs)
    sxy = sum(x*y for x,y in zip(xs,ys))
    denom = n*sxx - sx*sx
    m = 0.0 if abs(denom) < 1e-12 else (n*sxy - sx*sy)/denom
    b = (sy - m*sx)/n
    ybar = sy/n
    ss_tot = sum((y - ybar)**2 for y in ys)
    ss_res = sum((y - (m*x + b))**2 for x,y in zip(xs,ys))
    r2 = 1.0 - (ss_res/ss_tot if ss_tot != 0 else 0.0)
    pairs = "\n".join(f"{_fmt(a)}\t{_fmt(b)}" for a,b in zip(xs,ys))
    body = rf"""
\begin{{tikzpicture}}
\begin{{axis}}[axis lines=left, xlabel=$x$, ylabel=$y$, grid=both]
\addplot[only marks, mark=*] table[row sep=crcr]{{
{pairs}
}};
\addplot[domain={_fmt(min(xs))}:{_fmt(max(xs))}, samples=2] {{ {_fmt(m)}*x + {_fmt(b)} }};
\node[anchor=south west] at (rel axis cs:0.02,0.98) {{${{y = {_fmt(m)}x + {_fmt(b)}}}\quad R^2={_fmt(r2)}$}};
\end{{axis}}
\end{{tikzpicture}}
"""
    preflight_latex(body)
    return body, True

def render_stats_normal(ir: Dict[str, Any]) -> Tuple[str,bool]:
    mu = float(ir.get("mu", 0.0))
    sigma = float(ir.get("sigma", 1.0))
    ensure_range(sigma, 1e-6, 1e6, "sigma")
    a = float(ir.get("a", -4.0)); b = float(ir.get("b", 4.0))
    if a >= b:
        raise ValidationError("range invalid: left bound must be < right bound")
    body = rf"""
\begin{{tikzpicture}}
\begin{{axis}}[axis lines=left, xlabel=$x$, ylabel=$f(x)$, domain={_fmt(a)}:{_fmt(b)}, samples=300, grid=both]
\addplot {{ 1/({_fmt(sigma)}*sqrt(2*pi)) * exp(-0.5*((x-{_fmt(mu)})/{_fmt(sigma)})^2) }};
\end{{axis}}
\end{{tikzpicture}}
"""
    preflight_latex(body)
    return body, True

# ---- Physics ----
def render_incline(ir: Dict[str, Any]) -> Tuple[str,bool,bool]:
    theta = float(ir.get("theta", 30.0))
    mass  = float(ir.get("mass", 2.0))
    mu    = float(ir.get("mu", 0.2))
    ensure_range(theta, 0, 85, "theta")
    ensure_range(mu, 0, 2, "mu")
    components = bool(ir.get("components", True))
    friction   = bool(ir.get("friction", True))
    tension    = bool(ir.get("tension", False))
    body = rf"""
\begin{{tikzpicture}}[scale=1.1, >=stealth]
  \def\ang{{{_fmt(theta)}}}
  \draw[rotate=\ang] (0,0) -- (6,0);
  \draw[rotate=\ang] (0,0) -- (0,2.5);
  \filldraw[rotate=\ang, fill=gray!20] (2.5,0.2) rectangle (3.7,1.1);
  \draw[->, thick] (3.1,1.1) -- ++(0,-2) node[below] {{$mg={_fmt(9.8*mass)}\,\mathrm{{N}}$}};
  {"\\draw[->, thick, rotate=\\ang] (3.1,1.1) -- ++(-1.5,0) node[below] {$mg\\sin\\theta$};" if components else ""}
  {"\\draw[->, thick, rotate=\\ang] (3.1,1.1) -- ++(0,1.6) node[above] {$N$};" if components else ""}
  {"\\draw[->, thick, rotate=\\ang] (3.7,0.65) -- ++(0.9,0) node[right] {$f=\\mu N$};" if friction else ""}
  {"\\draw[->, thick, rotate=\\ang, blue] (2.5,0.65) -- ++(-1.2,0) node[left] {$T$};" if tension else ""}
  \draw (0,0) ++(0.9,0) arc (0:\ang:0.9);
  \node at (1.0,-0.5) {{$\\theta={_fmt(theta)}^\\circ$}};
  \node at (1.8,-0.9) {{$\\mu={_fmt(mu)}$}};
\end{{tikzpicture}}
"""
    preflight_latex(body)
    return body, False, False

def render_projectile(ir: Dict[str, Any]) -> Tuple[str,bool,bool]:
    v0 = float(ir.get("v0", 20.0))
    alpha = float(ir.get("alpha", 45.0))
    g = float(ir.get("g", 9.8))
    ensure_range(alpha, 0, 89, "alpha")
    ensure_range(g, 0.1, 100, "g")
    show_marks = bool(ir.get("marks", True))

    rad = math.radians(alpha)
    R = v0**2 * math.sin(2*rad) / g
    H = (v0**2 * (math.sin(rad))**2) / (2*g)

    body = rf"""
\begin{{tikzpicture}}
\begin{{axis}}[
  axis lines=left, xlabel=$x$ (m), ylabel=$y$ (m),
  domain=0:{_fmt(R*1.05)}, samples=300, grid=both
]
\addplot {{
  {v0}*x*{math.sin(rad)} - 0.5*{g}*x^2/{(v0*math.cos(rad))**2}
}};
{"\\addplot[only marks] coordinates { (" + _fmt(R) + ",0) } node[pos=1, below] {Range};" if show_marks else ""}
{"\\addplot[only marks] coordinates { (" + _fmt(R/2) + "," + _fmt(H) + ") } node[pos=1, above] {Apex};" if show_marks else ""}
\end{{axis}}
\end{{tikzpicture}}
"""
    preflight_latex(body)
    return body, True, False

def render_circuit(ir: Dict[str, Any]) -> Tuple[str,bool,bool]:
    topo = str(ir.get("topology", "series")).lower()
    V = float(ir.get("V", 9.0))
    Rs = list(map(float, ir.get("R", [100,220,330])))
    labels = bool(ir.get("labels", True))
    if len(Rs) == 0:
        raise ValidationError("R list is empty")

    if topo == "series":
        x2 = 0.0
        segs = []
        segs.append(f"\\begin{{circuitikz}}[european]")
        segs.append(f"  \\draw (0,0) to[battery1,l_={_fmt(V)} V] (0,3)")
        for i, R in enumerate(Rs, start=1):
            x2 += 2.5
            lab = f",l_=R{i}={_fmt(R)}\\\\$\\Omega$" if labels else ""
            segs.append(f"    to[R{lab}] ({_fmt(x2)},3)")
        segs.append(f"    -- ({_fmt(x2)},0) -- (0,0);")
        segs.append("\\end{circuitikz}")
        body = "\n".join(segs) + "\n"
        preflight_latex(body)
        return body, False, True

    # parallel
    height = max(3.0, 1.5 + 1.2*len(Rs))
    segs = [f"\\begin{{circuitikz}}[european]",
            f"  \\draw (0,0) to[battery1,l_={_fmt(V)} V] (0,{_fmt(height)}) -- (5,{_fmt(height)});"]
    y = height - 1.0
    for i, R in enumerate(Rs, start=1):
        lab = f",l_=R{i}={_fmt(R)}\\\\$\\Omega$" if labels else ""
        segs.append(f"  \\draw (5,{_fmt(y)}) to[R{lab}] (5,{_fmt(y-1.0)}) -- (0,{_fmt(y-1.0)});")
        y -= 1.2
    segs.append("\\end{circuitikz}")
    body = "\n".join(segs) + "\n"
    preflight_latex(body)
    return body, False, True

# ---- Chemistry ----
def render_benzene(ir: Dict[str, Any]) -> Tuple[str,bool]:
    slots = ["","","","","",""]
    subs: List[str] = ir.get("subs", [])  # e.g., ["1-CH3","4-NO2"]
    for s in subs:
        mt = re.match(r"(\d+)\s*-\s*([A-Za-z0-9\+\-]+)", s)
        if mt:
            idx = max(1, min(6, int(mt.group(1)))) - 1
            slots[idx] = mt.group(2)
    segs = [("-"+s) if s else "-" for s in slots]
    ring = "*6(" + "".join(segs) + ")"
    body = rf"""
\centering
\chemfig{{{ring}}}
"""
    preflight_latex(body)
    return body, False

def render_molecule(ir: Dict[str, Any]) -> Tuple[str,bool]:
    mol = ir.get("mol", "H2O")
    if not re.match(r"^[A-Za-z0-9\-\+\(\)]+$", mol):
        raise ValidationError("invalid molecule token; use simple chemfig string like CH3-CH2-OH")
    body = rf"""
\centering
\chemfig{{{mol}}}
"""
    preflight_latex(body)
    return body, False

# ===================== English NL Parsers (NL → IR) =====================

# Math NL: function plot
def parse_math_en(text: str) -> Dict[str, Any]:
    t = text.strip()
    # e.g., "Plot y = sin(x) from -6.28 to 6.28" or "plot y=exp(-x^2) from -2 to 2"
    m = re.search(r"plot\s+y\s*=\s*([^;]+?)\s+from\s+([-\d\.]+)\s+to\s+([-\d\.]+)", t, re.IGNORECASE)
    if m:
        expr = m.group(1).strip()
        # normalize common functions for pgfplots
        expr = expr.replace("sin(x)", "sin(deg(x))").replace("cos(x)", "cos(deg(x))")
        x0 = float(m.group(2)); x1 = float(m.group(3))
        if x0 >= x1: raise ValidationError("range invalid: left bound must be < right bound")
        return {"task":"math_function", "expr":expr, "x0":x0, "x1":x1}
    raise ValidationError("Unrecognized math instruction. Try: 'Plot y = sin(x) from -6.28 to 6.28'")

# Statistics NL
def parse_stats_en(text: str) -> Dict[str, Any]:
    t = text.lower().strip()

    # histogram: "histogram with 10 bins for data 1, 1.2, 0.9"
    mh = re.search(r"(?:hist|histogram).*(?:with\s+([0-9]+)\s+bins?)?.*?(?:data|values?)\s*([\[\(]?[0-9\.\,\s\-]+[\]\)]?)", text, re.IGNORECASE)
    if mh:
        bins = int(mh.group(1)) if mh.group(1) else 10
        nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", mh.group(2))]
        return {"task":"hist","bins":bins,"data":nums}

    # scatter: "scatter x:[1,2,3] y:[2,3,5]"
    ms = re.search(r"scatter.*?x[:=]\s*[\[\(]([^\]\)]*)[\]\)].*?y[:=]\s*[\[\(]([^\]\)]*)[\]\)]", text, re.IGNORECASE)
    if ms:
        xs = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", ms.group(1))]
        ys = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", ms.group(2))]
        return {"task":"scatter","x":xs,"y":ys}

    # regression: "linear regression x:[...] y:[...]"
    mr = re.search(r"(?:linear\s+)?regression.*?x[:=]\s*[\[\(]([^\]\)]*)[\]\)].*?y[:=]\s*[\[\(]([^\]\)]*)[\]\)]", text, re.IGNORECASE)
    if mr:
        xs = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", mr.group(1))]
        ys = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", mr.group(2))]
        return {"task":"regression","x":xs,"y":ys}

    # normal curve: "normal curve mu=0 sigma=1 from -4 to 4"
    mn = re.search(r"normal\s+curve.*?mu\s*=\s*([-\d\.]+).*?sigma\s*=\s*([-\d\.]+).*?from\s+([-\d\.]+)\s+to\s+([-\d\.]+)", text, re.IGNORECASE)
    if mn:
        return {"task":"normal","mu":float(mn.group(1)),"sigma":float(mn.group(2)),"a":float(mn.group(3)),"b":float(mn.group(4))}

    raise ValidationError("Unrecognized statistics instruction. Examples: 'histogram with 10 bins for data 1,1.2,...' or 'scatter x:[...] y:[...]'")

# Physics NL
def parse_physics_en(text: str) -> Dict[str, Any]:
    t = text.lower().replace("°","").strip()

    # Inclined plane FBD
    # e.g., "draw an inclined plane FBD at 30 degrees for a 2 kg block with friction 0.2, show components and friction, no tension"
    if "inclined" in t or "fbd" in t or "incline" in t:
        theta = _first_num(t, r"(?:at|angle|theta)\s*([0-9]+(?:\.[0-9]+)?)") or 30.0
        mass  = _first_num(t, r"(?:mass|m)\s*([0-9]+(?:\.[0-9]+)?)") or 2.0
        mu    = _first_num(t, r"(?:friction|mu|coefficient)\s*([0-9]+(?:\.[0-9]+)?)") or 0.2
        components = "no components" not in t
        friction   = "no friction" not in t
        tension    = ("tension" in t) and ("no tension" not in t)
        return {"task":"incline_fbd","theta":theta,"mass":mass,"mu":mu,
                "components":components,"friction":friction,"tension":tension}

    # Projectile
    # e.g., "projectile with v0=20 and angle=45, g=9.8, mark range and apex"
    if "projectile" in t or "trajectory" in t:
        v0 = _first_num(text, r"v0\s*=\s*([-\d\.]+)") or _first_num(text, r"speed\s*([-\d\.]+)") or 20.0
        alpha = _first_num(text, r"(?:angle|alpha)\s*=?\s*([-\d\.]+)") or 45.0
        g = _first_num(text, r"\bg\s*=\s*([-\d\.]+)") or 9.8
        marks = "no mark" not in t and "no marks" not in t
        return {"task":"projectile","v0":v0,"alpha":alpha,"g":g,"marks":marks}

    # Circuit (series/parallel)
    # e.g., "series circuit at 9V with resistors 100,220,330 and labels"
    if "circuit" in t:
        topo = "parallel" if "parallel" in t else "series"
        V = _first_num(text, r"([-\d\.]+)\s*v") or _first_num(text, r"\bv\s*=\s*([-\d\.]+)") or 9.0
        Rs = [float(x) for x in re.findall(r"(?:resistors?|r)\s*(?:=|:)?\s*([0-9\.\,\s]+)", text, re.IGNORECASE)[0].split(",")] if re.search(r"(?:resistors?|r)\s*(?:=|:)?\s*([0-9\.\,\s]+)", text, re.IGNORECASE) else [100,220,330]
        labels = "no label" not in t and "no labels" not in t
        return {"task":"circuit","topology":topo,"V":V,"R":Rs,"labels":labels}

    raise ValidationError("Unrecognized physics instruction. Examples: 'inclined plane ...', 'projectile with v0=20 angle=45', 'series circuit 9V resistors 100,220'")

def _first_num(text: str, pat: str) -> Optional[float]:
    m = re.search(pat, text, re.IGNORECASE)
    return float(m.group(1)) if m else None

# Chemistry NL
def parse_chem_en(text: str) -> Dict[str, Any]:
    t = text.lower().strip()

    # benzene ring with CH3 at position 1 and NO2 at position 4
    if "benzene" in t or "ring" in t:
        subs = []
        # capture patterns like "CH3 at position 1", "NO2 at 4"
        for chem, pos in re.findall(r"([A-Za-z0-9\+\-]{1,10})\s+at\s+(?:position\s+)?([1-6])", text, re.IGNORECASE):
            subs.append(f"{pos}-{chem}")
        if not subs:
            # also allow "1-CH3,4-NO2"
            parts = re.findall(r"([1-6]\s*-\s*[A-Za-z0-9\+\-]{1,10})", text)
            subs = [p.replace(" ", "") for p in parts] or ["1-CH3"]
        return {"task":"benzene","subs":subs}

    # raw molecule CH3-CH2-OH
    if "molecule" in t or "chemfig" in t or re.search(r"[A-Za-z0-9]+\-[A-Za-z0-9\-]+", text):
        m = re.search(r"(?:molecule\s*)?([A-Za-z0-9\-\+\(\)]+)", text)
        mol = m.group(1) if m else "H2O"
        return {"task":"molecule","mol":mol}

    raise ValidationError("Unrecognized chemistry instruction. Examples: 'benzene ring with CH3 at position 1 and NO2 at position 4' or 'molecule CH3-CH2-OH'")

# ===================== Dispatcher =====================

def generate_document(prompt: str, category_hint: str, subcategory_hint: Optional[str] = None) -> GenerationResult:
    """
    Directive/Form path (no NL parsing): category_hint decides the route.
    """
    _map = {"Mathematics":"Math","Math":"Math","Statistics":"Statistics","Physics":"Physics","Chemistry":"Chemistry","Auto":"Auto"}
    hint = _map.get(category_hint, "Auto")
    cat = detect_category(prompt) if hint == "Auto" else hint

    needs_pgfplots = False
    needs_circuitikz = False
    needs_chemfig = False

    if cat == "Math":
        if prompt.strip().lower().startswith("vectoranalysis:"):
            body, needs_pgfplots = render_vector_analysis_from_directive(prompt)
        else:
            # try function directive "Plot y = ... from ... to ..."
            ir = parse_math_en(prompt)  # allow English directive phrase
            body, needs_pgfplots = render_math_function(ir)

    elif cat == "Statistics":
        p = prompt.lower()
        if "mode=scatter" in p:
            xs = _parse_num_list(_extract(prompt, r"x\s*=\s*\[([^\]]*)\]"))
            ys = _parse_num_list(_extract(prompt, r"y\s*=\s*\[([^\]]*)\]"))
            ir = {"task":"scatter","x":xs,"y":ys}
            body, needs_pgfplots = render_stats_scatter(ir)
        elif "mode=hist" in p:
            data = _parse_num_list(_extract(prompt, r"data\s*=\s*\[([^\]]*)\]"))
            bins = int(_extract(prompt, r"bins\s*=\s*([0-9]+)") or 10)
            ir = {"task":"hist","bins":bins,"data":data}
            body, needs_pgfplots = render_stats_hist(ir)
        elif "mode=regression" in p:
            xs = _parse_num_list(_extract(prompt, r"x\s*=\s*\[([^\]]*)\]"))
            ys = _parse_num_list(_extract(prompt, r"y\s*=\s*\[([^\]]*)\]"))
            ir = {"task":"regression","x":xs,"y":ys}
            body, needs_pgfplots = render_stats_regression(ir)
        elif "normal curve" in p:
            ir = {
                "task":"normal",
                "mu": float(_extract(prompt, r"mu\s*=\s*([-\d\.]+)") or 0.0),
                "sigma": float(_extract(prompt, r"sigma\s*=\s*([-\d\.]+)") or 1.0),
                "a": float(_extract(prompt, r"from\s+([-\d\.]+)") or -4.0),
                "b": float(_extract(prompt, r"to\s+([-\d\.]+)") or 4.0),
            }
            body, needs_pgfplots = render_stats_normal(ir)
        else:
            # try NL parser as fallback in stats tab
            ir = parse_stats_en(prompt)
            body, needs_pgfplots = _render_stats_by_ir(ir)

    elif cat == "Physics":
        p = prompt.lower()
        if "mode=incline" in p:
            ir = {
                "task":"incline_fbd",
                "theta": float(_extract(prompt, r"theta\s*=\s*([-\d\.]+)") or 30.0),
                "mass": float(_extract(prompt, r"\bm\s*=\s*([-\d\.]+)") or 2.0),
                "mu": float(_extract(prompt, r"\bmu\s*=\s*([-\d\.]+)") or 0.2),
                "components": _b(_extract(prompt, r"components\s*=\s*(\w+)") or "true"),
                "friction": _b(_extract(prompt, r"friction\s*=\s*(\w+)") or "true"),
                "tension": _b(_extract(prompt, r"tension\s*=\s*(\w+)") or "false"),
            }
            body, nplot, ncirc = render_incline(ir); needs_pgfplots|=nplot; needs_circuitikz|=ncirc
        elif "mode=projectile" in p:
            ir = {
                "task":"projectile",
                "v0": float(_extract(prompt, r"v0\s*=\s*([-\d\.]+)") or 20.0),
                "alpha": float(_extract(prompt, r"alpha\s*=\s*([-\d\.]+)") or 45.0),
                "g": float(_extract(prompt, r"\bg\s*=\s*([-\d\.]+)") or 9.8),
                "marks": _b(_extract(prompt, r"marks\s*=\s*(\w+)") or "true"),
            }
            body, nplot, ncirc = render_projectile(ir); needs_pgfplots|=nplot; needs_circuitikz|=ncirc
        elif "mode=circuit" in p:
            Rs = _parse_num_list(_extract(prompt, r"R\s*=\s*\[([^\]]*)\]") or "")
            ir = {
                "task":"circuit",
                "topology": (_extract(prompt, r"topology\s*=\s*(\w+)") or "series"),
                "V": float(_extract(prompt, r"\bV\s*=\s*([-\d\.]+)") or 9.0),
                "R": Rs if Rs else [100,220,330],
                "labels": _b(_extract(prompt, r"labels\s*=\s*(\w+)") or "true"),
            }
            body, nplot, ncirc = render_circuit(ir); needs_pgfplots|=nplot; needs_circuitikz|=ncirc
        else:
            # try NL
            ir = parse_physics_en(prompt)
            body, needs_pgfplots, needs_circuitikz = _render_physics_by_ir(ir)

    elif cat == "Chemistry":
        p = prompt.lower()
        if "mode=benzene" in p:
            subs_raw = _extract(prompt, r"subs\s*=\s*([^\n\r]+)") or "none"
            subs = [] if subs_raw.strip().lower()=="none" else [s.strip() for s in subs_raw.split(",")]
            ir = {"task":"benzene","subs":subs}
            body, needs_chemfig = render_benzene(ir)
        elif "mode=raw" in p:
            mol = _extract(prompt, r"mol\s*=\s*([A-Za-z0-9\-\+\(\)]+)") or "H2O"
            ir = {"task":"molecule","mol":mol}
            body, needs_chemfig = render_molecule(ir)
        else:
            ir = parse_chem_en(prompt)
            body, needs_chemfig = _render_chem_by_ir(ir)

    else:
        body = "% Unrecognized category"

    latex_full = _wrap_document(body, needs_pgfplots, needs_circuitikz, needs_chemfig)
    return GenerationResult(latex=latex_full, category=cat, summary="Generated successfully")

def _extract(text: str, pat: str) -> Optional[str]:
    m = re.search(pat, text, re.IGNORECASE)
    return m.group(1) if m else None

def _render_stats_by_ir(ir: Dict[str, Any]) -> Tuple[str,bool]:
    t = ir["task"]
    if t == "hist": return render_stats_hist(ir)
    if t == "scatter": return render_stats_scatter(ir)
    if t == "regression": return render_stats_regression(ir)
    if t == "normal": return render_stats_normal(ir)
    raise ValidationError(f"Unsupported stats IR task: {t}")

def _render_physics_by_ir(ir: Dict[str, Any]) -> Tuple[str,bool,bool]:
    t = ir["task"]
    if t == "incline_fbd": return render_incline(ir)
    if t == "projectile": return render_projectile(ir)
    if t == "circuit": return render_circuit(ir)
    raise ValidationError(f"Unsupported physics IR task: {t}")

def _render_chem_by_ir(ir: Dict[str, Any]) -> Tuple[str,bool]:
    t = ir["task"]
    if t == "benzene": return render_benzene(ir)
    if t == "molecule": return render_molecule(ir)
    raise ValidationError(f"Unsupported chemistry IR task: {t}")

# ---------- NL unified entry (for app NL mode) ----------
def nl_parse_and_render(text: str, category_hint: str, preview_only: bool=False) -> Any:
    """
    If preview_only=True: return IR only.
    Else: return GenerationResult with compiled LaTeX doc.
    """
    cat_map = {"Mathematics":"Math","Math":"Math","Statistics":"Statistics","Physics":"Physics","Chemistry":"Chemistry"}
    cat = cat_map.get(category_hint, "Math")

    if cat == "Math":
        ir = parse_math_en(text)
        if preview_only: return ir
        body, need_plot = render_math_function(ir)
        return GenerationResult(latex=_wrap_document(body, need_plot, False, False), category="Math", summary="NL OK")

    if cat == "Statistics":
        ir = parse_stats_en(text)
        if preview_only: return ir
        body, need_plot = _render_stats_by_ir(ir)
        return GenerationResult(latex=_wrap_document(body, need_plot, False, False), category="Statistics", summary="NL OK")

    if cat == "Physics":
        ir = parse_physics_en(text)
        if preview_only: return ir
        body, need_plot, need_circ = _render_physics_by_ir(ir)
        return GenerationResult(latex=_wrap_document(body, need_plot, need_circ, False), category="Physics", summary="NL OK")

    if cat == "Chemistry":
        ir = parse_chem_en(text)
        if preview_only: return ir
        body, need_chem = _render_chem_by_ir(ir)
        return GenerationResult(latex=_wrap_document(body, False, False, need_chem), category="Chemistry", summary="NL OK")

    raise ValidationError("Unsupported category in NL mode")
