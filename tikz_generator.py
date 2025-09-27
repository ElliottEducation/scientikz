from dataclasses import dataclass
import re
import math

# ===============================
# Data model
# ===============================

@dataclass
class GenerationResult:
    latex: str
    summary: str
    category: str

def _fmt(code: str) -> str:
    return code.strip().replace("\t", "    ")

# ===============================
# Small utilities
# ===============================

_NUM = r"-?\d+(?:\.\d+)?"

def _find_float_pair(text: str, label: str):
    """
    Match 'label=(a,b)' or 'label a,b' or 'label (a,b)'.
    Returns tuple(float,float) or None.
    """
    m = re.search(rf"{label}\s*=\s*\(\s*({_NUM})\s*,\s*({_NUM})\s*\)", text, flags=re.I)
    if not m:
        m = re.search(rf"{label}\s*\(?\s*({_NUM})\s*,\s*({_NUM})\s*\)?", text, flags=re.I)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None

def _find_float(text: str, *keys: str, default: float | None = None) -> float | None:
    for k in keys:
        m = re.search(rf"{k}\s*[:=]?\s*({_NUM})", text, flags=re.I)
        if m:
            return float(m.group(1))
    m2 = re.search(rf"({_NUM})\s*°", text)  # e.g., '30°'
    if m2 and any(k.lower() in text.lower() for k in keys):
        return float(m2.group(1))
    return default

def _has(text: str, *words: str) -> bool:
    return any(re.search(rf"\b{re.escape(w)}\b", text, flags=re.I) for w in words)

def _clip(n: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, n))

# ===============================
# Mathematics — Vector Analysis
# ===============================

def _gen_math_vector(prompt: str) -> GenerationResult:
    p = prompt.strip()

    v = _find_float_pair(p, r"v") or _find_float_pair(p, r"vector") or (3.0, 2.0)
    vx, vy = v

    origin = _find_float_pair(p, r"origin") or (0.0, 0.0)
    x0, y0 = origin

    want_components = _has(p, "components", "component", "decompose", "dashed", "resolution")
    want_unit = _has(p, "unit vector", "unit", "normalize", "normalise")
    want_grid = _has(p, "grid")

    L = max(abs(vx), abs(vy), abs(x0) + abs(vx), abs(y0) + abs(vy), 3.0)
    L = _clip(L, 3.0, 12.0)
    xlim = ylim = math.ceil(L + 1.0)

    mag = math.hypot(vx, vy)
    ux, uy = ((vx / mag, vy / mag) if mag > 1e-9 else (0.0, 0.0))

    grid_line = r"\draw[very thin,step=1cm,color=gray!30] (-{L},-{L}) grid ({L},{L});".replace("{L}", f"{xlim}")
    if not want_grid:
        grid_line = "% (no grid)"

    components_code = []
    if want_components:
        components_code.append(
            rf"\draw[dashed] ({x0+vx:.3f},{y0+vy:.3f}) -- ({x0+vx:.3f},{y0:.3f});"
        )
        components_code.append(
            rf"\draw[dashed] ({x0+vx:.3f},{y0+vy:.3f}) -- ({x0:.3f},{y0+vy:.3f});"
        )
        components_code.append(
            rf"\node[below] at ({x0+vx:.3f},{y0:.3f}) {{$v_x={vx:g}$}};"
        )
        components_code.append(
            rf"\node[left]  at ({x0:.3f},{y0+vy:.3f}) {{$v_y={vy:g}$}};"
        )

    unit_code = []
    if want_unit and mag > 1e-9:
        unit_code.append(
            rf"\draw[->,thick,green!60!black] ({x0:.3f},{y0:.3f}) -- ++({ux:.3f},{uy:.3f}) node[above right] {{$\hat v$}};"
        )
        unit_code.append(
            rf"\node at ({x0+ux*0.6:.3f},{y0+uy*0.6:.3f}) [green!60!black] {{$|v|={mag:.3f}$}};"
        )

    body = rf"""
\begin{{tikzpicture}}[>=stealth]
  % axes
  \draw[->] (-{xlim},0) -- ({xlim},0) node[right] {{$x$}};
  \draw[->] (0,-{ylim}) -- (0,{ylim}) node[above] {{$y$}};

  {grid_line}

  % vector v
  \draw[->,very thick,blue] ({x0:.3f},{y0:.3f}) -- ++({vx:.3f},{vy:.3f}) node[above right] {{$\vec v$}};

  % components
  {"".join(components_code)}

  % unit vector
  {"".join(unit_code)}
\end{{tikzpicture}}
"""
    summary = f"2D vector v=({vx:g},{vy:g}) from origin ({x0:g},{y0:g}); components={want_components}, unit={want_unit}, grid={want_grid}."
    return GenerationResult(latex=_fmt(body), summary=summary, category="Mathematics · Vector Analysis")

# ===============================
# Mathematics — Algebra
# ===============================

def _gen_math_algebra(prompt: str) -> GenerationResult:
    p = prompt.strip()
    m = re.search(r"y\s*=\s*([^;,\n]+)", p, flags=re.I)
    expr = m.group(1).strip() if m else "x^2"

    m2 = re.search(r"from\s*({_NUM})\s*(?:to|-)\s*({_NUM})", p, flags=re.I)
    a, b = (float(m2.group(1)), float(m2.group(2))) if m2 else (-3.0, 3.0)

    expr_pgfm = expr.replace("arctan", "atan")

    body = rf"""
\begin{{tikzpicture}}
  \draw[->] ({a:.3f},0) -- ({b:.3f},0) node[right] {{$x$}};
  \draw[->] (0,-3) -- (0,3) node[above] {{$y$}};
  \draw[domain={a:.3f}:{b:.3f},smooth,variable=\x,blue] plot (\x,{{{expr_pgfm}}});
\end{{tikzpicture}}
"""
    summary = f"Plot y={expr} over [{a:g},{b:g}]."
    return GenerationResult(latex=_fmt(body), summary=summary, category="Mathematics · Algebra")

# ===============================
# Mathematics — Calculus
# ===============================

def _gen_math_calculus(prompt: str) -> GenerationResult:
    """
    Supports two common tasks:
      1) Shade area under y = f(x) from a to b  (keywords: area, integral, shade)
      2) Draw tangent to y = f(x) at x = a      (keyword: tangent)
    """
    p = prompt.strip()

    # function
    m = re.search(r"y\s*=\s*([^;,\n]+)", p, flags=re.I)
    expr = (m.group(1).strip() if m else "x^2")

    # domain
    m2 = re.search(r"from\s*({_NUM})\s*(?:to|-)\s*({_NUM})", p, flags=re.I)
    a, b = (float(m2.group(1)), float(m2.group(2))) if m2 else (-1.0, 2.0)

    if _has(p, "tangent"):
        # x0
        x0 = _find_float(p, "x0", "x =", "at x", default=0.0)
        if x0 is None:
            x0 = (a + b) / 2.0

        # For demo, we approximate derivative symbolically only for simple cases
        # Otherwise show a generic line with slope approximated numerically
        def _safe_eval(x: float):
            local = {"x": x, "sin": math.sin, "cos": math.cos, "tan": math.tan,
                     "exp": math.exp, "log": math.log, "sqrt": math.sqrt, "pi": math.pi}
            try:
                return float(eval(expr, {"__builtins__": {}}, local))
            except Exception:
                return x*x  # fallback

        y0 = _safe_eval(x0)
        h = 1e-4
        slope = (_safe_eval(x0 + h) - _safe_eval(x0 - h)) / (2*h)

        # choose a visible segment around x0
        x1, x2 = x0 - (b - a) * 0.3, x0 + (b - a) * 0.3
        y1 = y0 + slope * (x1 - x0)
        y2 = y0 + slope * (x2 - x0)

        body = rf"""
\begin{{tikzpicture}}
  \draw[->] ({a:.3f},0) -- ({b:.3f},0) node[right] {{$x$}};
  \draw[->] (0,-3) -- (0,3) node[above] {{$y$}};
  \draw[domain={a:.3f}:{b:.3f},smooth,variable=\x,blue] plot (\x,{{{expr}}});
  % tangent
  \draw[red,thick] ({x1:.3f},{y1:.3f}) -- ({x2:.3f},{y2:.3f}) node[above right] {{$t$}};
  \fill[red] ({x0:.3f},{y0:.3f}) circle (1.2pt) node[above] {{$({x0:g},{y0:.2f})$}};
\end{{tikzpicture}}
"""
        summary = f"Tangent to y={expr} at x={x0:g}, slope≈{slope:.3f}."
        return GenerationResult(_fmt(body), summary, "Mathematics · Calculus")

    # Area (integral) mode (default if mentions area/integral/shade)
    if _has(p, "area", "integral", "shade"):
        body = rf"""
\begin{{tikzpicture}}
  \draw[->] ({a:.3f},0) -- ({b:.3f},0) node[right] {{$x$}};
  \draw[->] (0,-3) -- (0,3) node[above] {{$y$}};
  \draw[domain={a:.3f}:{b:.3f},smooth,variable=\x,blue] plot (\x,{{{expr}}});
  \addplot[name path=A,domain={a:.3f}:{b:.3f}]{{{expr}}};
\end{{tikzpicture}}
"""
        # 为兼容通用 LaTeX，直接用填充路径实现阴影
        body = rf"""
\begin{{tikzpicture}}
  \draw[->] ({a:.3f},0) -- ({b:.3f},0) node[right] {{$x$}};
  \draw[->] (0,-3) -- (0,3) node[above] {{$y$}};
  \draw[domain={a:.3f}:{b:.3f},smooth,variable=\x,blue] plot (\x,{{{expr}}});
  \begin{{scope}}
    \clip ({a:.3f},0) rectangle ({b:.3f},3.1);
    \fill[blue!20] plot[domain={a:.3f}:{b:.3f}] ({'{'}\x{'}'},{{{expr}}}) -- ({b:.3f},0) -- ({a:.3f},0) -- cycle;
  \end{{scope}}
\end{{tikzpicture}}
"""
        summary = f"Shaded area under y={expr} from x={a:g} to {b:g}."
        return GenerationResult(_fmt(body), summary, "Mathematics · Calculus")

    # fallback：如果没提及关键词，就画函数
    return _gen_math_algebra(prompt)

# ===============================
# Physics — Classical Mechanics (Inclined plane FBD)
# ===============================

def _gen_phys_classical(prompt: str) -> GenerationResult:
    p = prompt.strip()

    ang = _find_float(p, "angle", "theta", default=None)
    if ang is None:
        m2 = re.search(r"([0-9]+(?:\.\d+)?)\s*°", p)
        ang = float(m2.group(1)) if m2 else 30.0
    ang = _clip(ang, 0.0, 60.0)

    has_friction = _has(p, "friction", "mu", "coefficient of friction") and not _has(p, "no friction")
    show_components = _has(p, "components", "decompose", "resolution", "resolve", "parallel", "perpendicular")
    want_grid = _has(p, "grid")

    grid_line = r"\draw[very thin,step=1cm,color=gray!30] (-6,-4) grid (6,4);" if want_grid else "% (no grid)"

    body = rf"""
\begin{{tikzpicture}}[scale=1,>=stealth]
  % plane at angle {ang:.1f}°
  {grid_line}
  \begin{{scope}}[rotate={ang:.3f}]
    \draw[thick] (-6,-0.02) -- (6, -0.02); % incline line

    % block
    \coordinate (C) at (0,0.9);
    \draw[fill=gray!15] (-1,0) rectangle (1,1.8);

    % normal force
    \draw[->,thick,red] (C) -- ++(90:2.0) node[above] {{$N$}};

    % friction (along plane)
    {"\\draw[->,thick,orange] (C) -- ++(-0:2.0) node[above] {$f$};" if has_friction else "% (no friction)"}    

    % components of mg (optional)
    {"\\draw[dashed,thick] (C) -- ++(-2,0) node[below] {$mg\\sin\\theta$};" if show_components else "%"}
    {"\\draw[dashed,thick] (C) -- ++(0,-2) node[left] {$mg\\cos\\theta$};" if show_components else "%"}
    \draw (0,0) ++(-1.2,0) arc[start angle=180, end angle=180-{ang:.3f}, radius=1.2];
    \node at (-0.6,0.35) {{$\\theta={ang:.0f}^\\circ$}};
  \end{{scope}}

  % gravity mg (global vertical)
  \draw[->,thick,blue] (C) -- ++(0,-2.5) node[below] {{$mg$}};
\end{{tikzpicture}}
"""
    summary = f"Inclined-plane FBD with theta={ang:.0f}°, friction={has_friction}, components={show_components}."
    return GenerationResult(_fmt(body), summary, "Physics · Classical Mechanics")

# ===============================
# Physics — Electromagnetism
# ===============================

def _gen_phys_em(prompt: str) -> GenerationResult:
    p = prompt.strip()

    # Case A: series circuit
    if _has(p, "circuit", "series"):
        # parse component values if given
        R = _find_float(p, "R", "resistance", default=2.0)
        L = _find_float(p, "L", "inductance", default=3.0)
        C = _find_float(p, "C", "capacitance", default=4.0)
        V = _find_float(p, "V", "voltage", "E", default=None)

        vs = rf"to[V={{{V:g}<\volt>}}] " if (V is not None and V != 0) else ""

        body = rf"""
\begin{{circuitikz}}
  \draw (0,0) {vs}to[R={{{R:g}<\ohm>}}] (2,0)
        to[L={{{L:g}<\henry>}}] (4,0)
        to[C={{{C:g}<\micro\farad>}}] (6,0)
        to[short] (6,-2)
        to[short] (0,-2)
        to[short] (0,0);
\end{{circuitikz}}
"""
        summary = f"Series circuit with R={R:g}Ω, L={L:g}H, C={C:g}µF" + (f", V={V:g}V." if V is not None else ".")
        return GenerationResult(_fmt(body), summary, "Physics · Electromagnetism (circuit)")

    # Case B: electric field lines — dipole schematic
    if _has(p, "field", "dipole", "field lines", "electric field"):
        d = _find_float(p, "distance", "sep", "separation", default=3.0)
        d = _clip(d, 2.0, 6.0)
        body = rf"""
\begin{{tikzpicture}}[>=stealth]
  % charges
  \fill[red!70]   (-{d/2:.2f},0) circle (6pt) node[left] {{$+q$}};
  \fill[blue!70]  ({d/2:.2f},0)  circle (6pt) node[right] {{$-q$}};

  % suggested field lines (schematic)
  \foreach \a in {{-60,-30,-15,0,15,30,60}} {{
    \draw[->,thin]  (-{d/2:.2f},0) .. controls (-{d/4:.2f},\a/50+1) and ({d/4:.2f},\a/50+1) .. ({d/2:.2f},0);
  }}
  \foreach \a in {{-60,-30,-15,0,15,30,60}} {{
    \draw[->,thin]  (-{d/2:.2f},0) .. controls (-{d/4:.2f},\a/50-1) and ({d/4:.2f},\a/50-1) .. ({d/2:.2f},0);
  }}
\end{{tikzpicture}}
"""
        summary = f"Electric field lines of a dipole (schematic), separation≈{d:g}."
        return GenerationResult(_fmt(body), summary, "Physics · Electromagnetism (field lines)")

    # fallback: simple RLC
    return _gen_phys_em("series circuit R=2 L=3 C=4")

# ===============================
# Physics — Projectile Motion
# ===============================

def _gen_phys_projectile(prompt: str) -> GenerationResult:
    p = prompt.strip()
    ang = _find_float(p, "angle", "theta", default=45.0) or 45.0
    v0  = _find_float(p, "speed", "velocity", "v0", default=10.0) or 10.0
    g = 9.8
    ang_rad = math.radians(ang)
    # Range R = v0^2 sin(2θ)/g
    R = (v0 * v0 * math.sin(2 * ang_rad)) / g
    R = max(1.0, R)

    # y(x) = x tanθ - g x^2 / (2 v0^2 cos^2θ)
    expr = rf"\x*tan({ang_rad:.5f}) - {g:.3f}*\x*\x/(2*{v0*v0:.4f}*cos({ang_rad:.5f})*cos({ang_rad:.5f}))"

    body = rf"""
\begin{{tikzpicture}}
  \draw[->] (0,0) -- ({R*1.05:.3f},0) node[right] {{$x$}};
  \draw[->] (0,0) -- (0,{R*0.5:.3f}) node[above] {{$y$}};
  \draw[domain=0:{R:.3f},smooth,variable=\x,blue,thick] plot (\x,{{{expr}}});
  \node at (1.2,0.8) {{$\theta={ang:.0f}^\circ,~v_0={v0:g}$}};
\end{{tikzpicture}}
"""
    summary = f"Projectile motion with angle={ang:g}°, speed={v0:g}."
    return GenerationResult(_fmt(body), summary, "Physics · Projectile Motion")

# ===============================
# Chemistry — Molecules
# ===============================

def _gen_chem_molecules(prompt: str) -> GenerationResult:
    p = prompt.strip().lower()
    if "benzene" in p:
        body = r"""
\chemfig{*6(-=-=-=)}
"""
        return GenerationResult(_fmt(body), "Benzene ring.", "Chemistry · Molecules")
    if "ethanol" in p or "ch3-ch2-oh" in p or "ch3ch2oh" in p:
        body = r"""
\chemfig{CH_3-CH_2-OH}
"""
        return GenerationResult(_fmt(body), "Ethanol molecule.", "Chemistry · Molecules")
    if "water" in p or "h2o" in p:
        body = r"""
\chemfig{H-O-H}
"""
        return GenerationResult(_fmt(body), "Water molecule.", "Chemistry · Molecules")
    if "co2" in p:
        body = r"""
\chemfig{O=C=O}
"""
        return GenerationResult(_fmt(body), "Carbon dioxide.", "Chemistry · Molecules")

    # fallback: try raw formula like "CH3-CH2-OH"
    m = re.search(r"\b([A-Z][A-Za-z0-9\-\(\)]{1,})\b", prompt)
    if m:
        formula = m.group(1)
        body = rf"""
\chemfig{{{formula}}}
"""
        return GenerationResult(_fmt(body), f"Molecule: {formula}.", "Chemistry · Molecules")

    return GenerationResult("% Unsupported molecule", "Unknown molecule.", "Chemistry · Molecules")

# ===============================
# Chemistry — Reactions
# ===============================

def _gen_chem_reactions(prompt: str) -> GenerationResult:
    """
    Parse 'A + B -> C + D' style reaction.
    """
    p = prompt.strip()
    m = re.search(r"(.+?)\s*->\s*(.+)", p)
    if m:
        left = m.group(1).strip()
        right = m.group(2).strip()
        body = rf"""
\schemestart
  {left} \arrow{->} {right}
\schemestop
"""
        return GenerationResult(_fmt(body), f"Reaction: {left} -> {right}.", "Chemistry · Reactions")

    # examples
    if _has(p, "combustion", "burn"):
        body = r"""
\schemestart
  CH_4 + 2 O_2 \arrow{->} CO_2 + 2 H_2O
\schemestop
"""
        return GenerationResult(_fmt(body), "Combustion of methane.", "Chemistry · Reactions")

    return GenerationResult("% Unsupported reaction", "No reaction arrow found.", "Chemistry · Reactions")

# ===============================
# Chemistry — Lab Apparatus (Distillation)
# ===============================

def _gen_chem_apparatus(prompt: str) -> GenerationResult:
    if _has(prompt, "distillation", "distill"):
        body = r"""
\begin{tikzpicture}[scale=1]
  % heating flask
  \draw[fill=blue!10] (-4,-1) .. controls (-4.8,-1) and (-5,-0.2) .. (-5,0.6)
                      .. controls (-5,1.4) and (-4,1.4) .. (-4,0.6)
                      .. controls (-4,0) and (-3.5,-0.2) .. (-3.5,-1) -- cycle;
  \draw (-4,0.6) -- (-3,0.6) -- (-2.5,0.9);

  % condenser (tube)
  \draw[thick] (-2.5,0.9) -- (1,0.9);
  \draw[thick] (1,0.9) -- (2.2,0.3);
  \draw ( -2.2,0.9) -- (-2.2,1.2) node[above] {to condenser water};

  % receiving flask
  \draw[fill=blue!10] (2.6,0.2) arc[start angle=10,end angle=350,radius=0.8];

  % burner
  \draw[fill=orange!60] (-4.3,-1.4) -- (-3.7,-1.4) -- (-4,-1.0) -- cycle;
  \node at (-4,-1.8) {heat};
\end{tikzpicture}
"""
        return GenerationResult(_fmt(body), "Simple distillation setup (schematic).", "Chemistry · Lab Apparatus")

    return GenerationResult("% Unsupported apparatus", "Unknown apparatus.", "Chemistry · Lab Apparatus")

# ===============================
# Router
# ===============================

def generate_document(prompt: str, category_hint: str, subcategory_hint: str | None = None) -> GenerationResult:
    cat = (category_hint or "").strip().lower()
    sub = (subcategory_hint or "").strip().lower()

    # Mathematics
    if cat.startswith("math"):
        if sub.startswith("vector"):
            return _gen_math_vector(prompt)
        if sub.startswith("algebra"):
            return _gen_math_algebra(prompt)
        if sub.startswith("calculus"):
            return _gen_math_calculus(prompt)
        # fallback
        return _gen_math_algebra(prompt)

    # Physics
    if cat.startswith("phys"):
        if sub.startswith("classical"):
            return _gen_phys_classical(prompt)
        if sub.startswith("electromag"):
            return _gen_phys_em(prompt)
        if sub.startswith("projectile"):
            return _gen_phys_projectile(prompt)
        # fallback
        return _gen_phys_em("series circuit R=2 L=3 C=4")

    # Chemistry
    if cat.startswith("chem"):
        if sub.startswith("molecule"):
            return _gen_chem_molecules(prompt)
        if sub.startswith("reaction"):
            return _gen_chem_reactions(prompt)
        if sub.startswith("lab"):
            return _gen_chem_apparatus(prompt)
        # fallback
        return _gen_chem_molecules(prompt)

    # Fallback
    return GenerationResult("% Unsupported category", "Unsupported category.", category_hint or "Unknown")

# ===============================
# Backward compatibility shim
# ===============================

def nl_parse_and_render(prompt: str, category_hint: str = "Auto", subcategory_hint: str | None = None):
    res = generate_document(prompt, category_hint, subcategory_hint)
    meta = {"category": res.category, "summary": res.summary}
    return res.latex, meta
