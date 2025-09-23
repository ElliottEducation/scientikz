# tikz_generator.py
# Core generator (no subject prefixes required; UI passes the category hint)
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

# ---------- Category detection (fallback only) ----------
def detect_category(prompt: str) -> Category:
    p = prompt.lower()
    if any(k in p for k in ["mode=scatter","mode=hist","mode=regression","normal curve"]):
        return "Statistics"
    if any(k in p for k in ["mode=incline","mode=projectile","mode=circuit","inclined","circuit"]):
        return "Physics"
    if any(k in p for k in ["chem:","mode=benzene","mode=raw","chemfig","molecule"]):
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
    return preamble + "\n\\begin{document}\n" + body + "\n\\end{document}\n"

# ---------- Mathematics ----------
def gen_math_plot(prompt: str) -> Tuple[str,bool]:
    m = re.search(r"plot\s*y\s*=\s*([^;]+?)\s+from\s+([-\d\.]+)\s+to\s+([-\d\.]+)", prompt, re.IGNORECASE)
    if m:
        expr = m.group(1).strip().replace("**","^").replace("sin(x)","sin(deg(x))").replace("cos(x)","cos(deg(x))")
        x0, x1 = m.group(2), m.group(3)
    else:
        expr = "sin(deg(x))"
        x0, x1 = "-6.28","6.28"
    body = rf"""
\begin{{tikzpicture}}
\begin{{axis}}[axis lines=middle, xlabel=$x$, ylabel=$y$, domain={x0}:{x1}, samples=300, grid=both]
\addplot {{{expr}}};
\end{{axis}}
\end{{tikzpicture}}
"""
    return body, True

# Vector Analysis (directive-style: "VectorAnalysis: ...")
def gen_vector_analysis(prompt: str) -> Tuple[str,bool]:
    # mode
    mm = re.search(r"mode\s*=\s*(\w+)", prompt, re.IGNORECASE)
    mode = (mm.group(1).lower() if mm else "single")

    # axes and grid
    ax = re.search(r"axes\s*=\s*\(\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*\)", prompt, re.IGNORECASE)
    ax_lim = float(ax.group(1)) if ax else 5.0
    ay_lim = float(ax.group(2)) if ax else 5.0
    mg = re.search(r"grid\s*=\s*(\w+)", prompt, re.IGNORECASE)
    grid = _b(mg.group(1)) if mg else True

    header = rf"""
\begin{{tikzpicture}}[>=stealth, scale=1.0]
  \draw[->] (-{_fmt(ax_lim)},0) -- ({_fmt(ax_lim)},0) node[right] {{$x$}};
  \draw[->] (0,-{_fmt(ay_lim)}) -- (0,{_fmt(ay_lim)}) node[above] {{$y$}};
  {"\\draw[step=1cm, very thin, gray!30] (-" + _fmt(ax_lim) + ",-" + _fmt(ay_lim) + ") grid (" + _fmt(ax_lim) + "," + _fmt(ay_lim) + ");" if grid else ""}
"""

    footer = r"\end{tikzpicture}"

    if mode == "single":
        mv = re.search(r"v\s*=\s*\(\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*\)", prompt, re.IGNORECASE)
        vx = float(mv.group(1)) if mv else 3.0
        vy = float(mv.group(2)) if mv else 2.0
        mo = re.search(r"origin\s*=\s*\(\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*\)", prompt, re.IGNORECASE)
        x0 = float(mo.group(1)) if mo else 0.0
        y0 = float(mo.group(2)) if mo else 0.0
        components = _b(re.search(r"components\s*=\s*(\w+)", prompt, re.IGNORECASE).group(1)) if re.search(r"components\s*=\s*(\w+)", prompt, re.IGNORECASE) else True
        unit = _b(re.search(r"unit\s*=\s*(\w+)", prompt, re.IGNORECASE).group(1)) if re.search(r"unit\s*=\s*(\w+)", prompt, re.IGNORECASE) else False

        mag = math.hypot(vx, vy)
        unx = vx / mag if mag != 0 else 0.0
        uny = vy / mag if mag != 0 else 0.0

        body = header + rf"""
  \draw[->, very thick] ({_fmt(x0)},{_fmt(y0)}) -- ++({_fmt(vx)},{_fmt(vy)}) node[above right] {{$\vec v$}};
  {"\\draw[dashed] (" + _fmt(x0+vx) + ",0) -- (" + _fmt(x0+vx) + "," + _fmt(y0+vy) + ");" if components else ""}
  {"\\draw[dashed] (0," + _fmt(y0+vy) + ") -- (" + _fmt(x0+vx) + "," + _fmt(y0+vy) + ");" if components else ""}
  \\node[below right] at (0,-0.6) {{$|\\vec v| = " + _fmt(mag) + "$}};
  """ + (rf"""
  \draw[->, thick, blue] ({_fmt(x0)},{_fmt(y0)}) -- ++({_fmt(unx)},{_fmt(uny)}) node[above left] {{$\hat v$}};
  """ if unit else "") + footer

        return body, False

    # pair mode
    ma = re.search(r"a\s*=\s*\(\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*\)", prompt, re.IGNORECASE)
    mb = re.search(r"b\s*=\s*\(\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*\)", prompt, re.IGNORECASE)
    axv = float(ma.group(1)) if ma else 2.0
    ayv = float(ma.group(2)) if ma else 1.0
    bxv = float(mb.group(1)) if mb else 1.0
    byv = float(mb.group(2)) if mb else 3.0

    sum_on  = _b(re.search(r"sum\s*=\s*(\w+)", prompt, re.IGNORECASE).group(1)) if re.search(r"sum\s*=\s*(\w+)", prompt, re.IGNORECASE) else True
    diff_on = _b(re.search(r"diff\s*=\s*(\w+)", prompt, re.IGNORECASE).group(1)) if re.search(r"diff\s*=\s*(\w+)", prompt, re.IGNORECASE) else False
    para_on = _b(re.search(r"parallelogram\s*=\s*(\w+)", prompt, re.IGNORECASE).group(1)) if re.search(r"parallelogram\s*=\s*(\w+)", prompt, re.IGNORECASE) else True
    ang_on  = _b(re.search(r"angle\s*=\s*(\w+)", prompt, re.IGNORECASE).group(1)) if re.search(r"angle\s*=\s*(\w+)", prompt, re.IGNORECASE) else True
    proj_on = _b(re.search(r"projection_a_on_b\s*=\s*(\w+)", prompt, re.IGNORECASE).group(1)) if re.search(r"projection_a_on_b\s*=\s*(\w+)", prompt, re.IGNORECASE) else False
    dot_on  = _b(re.search(r"dot\s*=\s*(\w+)", prompt, re.IGNORECASE).group(1)) if re.search(r"dot\s*=\s*(\w+)", prompt, re.IGNORECASE) else False

    sumx, sumy = axv + bxv, ayv + byv
    diffx, diffy = axv - bxv, ayv - byv
    na = math.hypot(axv, ayv); nb = math.hypot(bxv, byv)
    dot = axv * bxv + ayv * byv
    cos_th = 0.0 if na == 0 or nb == 0 else max(-1.0, min(1.0, dot/(na*nb)))
    theta = math.degrees(math.acos(cos_th)) if na != 0 and nb != 0 else 0.0
    proj_scale = 0.0 if nb == 0 else dot/(nb*nb)
    projx, projy = proj_scale*bxv, proj_scale*byv

    body = header + rf"""
  \draw[->, thick] (0,0) -- ({_fmt(axv)},{_fmt(ayv)}) node[above right] {{$\vec a$}};
  \draw[->, thick, red] (0,0) -- ({_fmt(bxv)},{_fmt(byv)}) node[above left] {{$\vec b$}};
  {"\\draw[dashed, gray] (" + _fmt(axv) + "," + _fmt(ayv) + ") -- (" + _fmt(sumx) + "," + _fmt(sumy) + ");" if para_on else ""}
  {"\\draw[dashed, gray] (" + _fmt(bxv) + "," + _fmt(byv) + ") -- (" + _fmt(sumx) + "," + _fmt(sumy) + ");" if para_on else ""}
  """ + (rf"""
  \draw[->, very thick, blue] (0,0) -- ({_fmt(sumx)},{_fmt(sumy)}) node[above right] {{$\vec a + \vec b$}};
  """ if sum_on else "") + (rf"""
  \draw[->, thick, purple] (0,0) -- ({_fmt(diffx)},{_fmt(diffy)}) node[below right] {{$\vec a - \vec b$}};
  """ if diff_on else "") + (rf"""
  \draw (1,0) arc (0:{_fmt(theta)}:1);
  \node at (0.9,0.35) {{$\theta={_fmt(theta)}^\circ$}};
  """ if ang_on else "") + (rf"""
  \draw[->, thick, orange] (0,0) -- ({_fmt(projx)},{_fmt(projy)}) node[below] {{$\mathrm{{proj}}_{{\vec b}} \vec a$}};
  \draw[dashed, orange] ({_fmt(axv)},{_fmt(ayv)}) -- ({_fmt(projx)},{_fmt(projy)});
  """ if proj_on else "") + (rf"""
  \node[below right] at (0,-0.6) {{$\vec a\cdot \vec b = {_fmt(dot)} = |\vec a||\vec b|\cos\theta$}};
  """ if dot_on else "") + footer

    return body, False

# ---------- Statistics ----------
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

def gen_statistics_directive(prompt: str) -> Tuple[str,bool]:
    p = prompt.lower()

    if "mode=scatter" in p:
        mx = re.search(r"x\s*=\s*\[([^\]]*)\]", prompt, re.IGNORECASE)
        my = re.search(r"y\s*=\s*\[([^\]]*)\]", prompt, re.IGNORECASE)
        xs = _parse_num_list(mx.group(1)) if mx else [1,2,3]
        ys = _parse_num_list(my.group(1)) if my else [2,3,5]
        pairs = "\n".join([f"{_fmt(a)}\t{_fmt(b)}" for a,b in zip(xs,ys)])
        body = rf"""
\begin{{tikzpicture}}
\begin{{axis}}[axis lines=left, xlabel=$x$, ylabel=$y$, grid=both]
\addplot[only marks, mark=*] table[row sep=crcr]{{
{pairs}
}};
\end{{axis}}
\end{{tikzpicture}}
"""
        return body, True

    if "mode=hist" in p:
        md = re.search(r"data\s*=\s*\[([^\]]*)\]", prompt, re.IGNORECASE)
        data = _parse_num_list(md.group(1)) if md else [1,1.2,0.9,1.5,2.1,2.0,2.2,2.8,3.0,2.6,3.2]
        mb = re.search(r"bins\s*=\s*([0-9]+)", prompt, re.IGNORECASE)
        bins = int(mb.group(1)) if mb else 10
        data_lines = "\n".join(_fmt(d) for d in data)
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
{data_lines}
}};
\end{{axis}}
\end{{tikzpicture}}
"""
        return body, True

    if "mode=regression" in p:
        mx = re.search(r"x\s*=\s*\[([^\]]*)\]", prompt, re.IGNORECASE)
        my = re.search(r"y\s*=\s*\[([^\]]*)\]", prompt, re.IGNORECASE)
        xs = _parse_num_list(mx.group(1)) if mx else [1,2,3,4,5]
        ys = _parse_num_list(my.group(1)) if my else [1.8,2.2,2.9,3.8,5.1]
        n = max(1, min(len(xs), len(ys)))
        xs, ys = xs[:n], ys[:n]
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
        pairs = "\n".join([f"{_fmt(a)}\t{_fmt(b)}" for a,b in zip(xs,ys)])
        line_part = rf"""
\addplot[domain={_fmt(min(xs))}:{_fmt(max(xs))}, samples=2] {{ { _fmt(m) }*x + { _fmt(b) } }};
"""
        eq_part = rf"""
\node[anchor=south west] at (rel axis cs:0.02,0.98) {{${{y = { _fmt(m) }x + { _fmt(b) }}}\quad R^2={ _fmt(r2) }$}};
"""
        body = rf"""
\begin{{tikzpicture}}
\begin{{axis}}[axis lines=left, xlabel=$x$, ylabel=$y$, grid=both]
\addplot[only marks, mark=*] table[row sep=crcr]{{
{pairs}
}};
{line_part}
{eq_part}
\end{{axis}}
\end{{tikzpicture}}
"""
        return body, True

    # Normal curve (natural-language style kept)
    if "normal curve" in p:
        m_mu = re.search(r"mu\s*=\s*([-\d\.]+)", prompt, re.IGNORECASE)
        m_si = re.search(r"sigma\s*=\s*([-\d\.]+)", prompt, re.IGNORECASE)
        m_rng = re.search(r"from\s+([-\d\.]+)\s+to\s+([-\d\.]+)", prompt, re.IGNORECASE)
        mu = _fmt(m_mu.group(1)) if m_mu else "0"
        si = _fmt(m_si.group(1)) if m_si else "1"
        a, b = ("-4","4")
        if m_rng:
            a, b = m_rng.group(1), m_rng.group(2)
        body = rf"""
\begin{{tikzpicture}}
\begin{{axis}}[axis lines=left, xlabel=$x$, ylabel=$f(x)$, domain={a}:{b}, samples=300, grid=both]
\addplot {{ 1/({si}*sqrt(2*pi)) * exp(-0.5*((x-{mu})/{si})^2) }};
\end{{axis}}
\end{{tikzpicture}}
"""
        return body, True

    return "% Unrecognized statistics prompt", False

# ---------- Physics ----------
def gen_physics_directive(prompt: str) -> Tuple[str,bool,bool]:
    p = prompt.lower()
    needs_pgfplots = False
    needs_circuitikz = False

    # Inclined plane
    if "mode=incline" in p:
        mt = re.search(r"theta\s*=\s*([-\d\.]+)", prompt, re.IGNORECASE)
        mm = re.search(r"\bm\s*=\s*([-\d\.]+)", prompt, re.IGNORECASE)
        mu = re.search(r"\bmu\s*=\s*([-\d\.]+)", prompt, re.IGNORECASE)
        comp = re.search(r"components\s*=\s*(\w+)", prompt, re.IGNORECASE)
        fr   = re.search(r"friction\s*=\s*(\w+)", prompt, re.IGNORECASE)
        ten  = re.search(r"tension\s*=\s*(\w+)", prompt, re.IGNORECASE)

        theta = float(mt.group(1)) if mt else 30.0
        mass  = float(mm.group(1)) if mm else 2.0
        muv   = float(mu.group(1)) if mu else 0.2
        show_comp = _b(comp.group(1) if comp else "true")
        show_fric = _b(fr.group(1) if fr else "true")
        show_T    = _b(ten.group(1) if ten else "false")

        body = rf"""
\begin{{tikzpicture}}[scale=1.1, >=stealth]
  \def\ang{{{_fmt(theta)}}}
  \draw[rotate=\ang] (0,0) -- (6,0);
  \draw[rotate=\ang] (0,0) -- (0,2.5);

  \filldraw[rotate=\ang, fill=gray!20] (2.5,0.2) rectangle (3.7,1.1);

  \draw[->, thick] (3.1,1.1) -- ++(0,-2) node[below] {{$mg={_fmt(9.8*mass)}\,\mathrm{{N}}$}};
  {"\\draw[->, thick, rotate=\\ang] (3.1,1.1) -- ++(-1.5,0) node[below] {$mg\\sin\\theta$};" if show_comp else ""}
  {"\\draw[->, thick, rotate=\\ang] (3.1,1.1) -- ++(0,1.6) node[above] {$N$};" if show_comp else ""}
  {"\\draw[->, thick, rotate=\\ang] (3.7,0.65) -- ++(0.9,0) node[right] {$f=\\mu N$};" if show_fric else ""}
  {"\\draw[->, thick, rotate=\\ang, blue] (2.5,0.65) -- ++(-1.2,0) node[left] {$T$};" if show_T else ""}

  \draw (0,0) ++(0.9,0) arc (0:\ang:0.9);
  \node at (1.0,-0.5) {{$\\theta={_fmt(theta)}^\\circ$}};
  \node at (1.8,-0.9) {{$\\mu={_fmt(muv)}$}};
\end{{tikzpicture}}
"""
        return body, needs_pgfplots, needs_circuitikz

    # Projectile
    if "mode=projectile" in p:
        mv0 = re.search(r"v0\s*=\s*([-\d\.]+)", prompt, re.IGNORECASE)
        ma  = re.search(r"alpha\s*=\s*([-\d\.]+)", prompt, re.IGNORECASE)
        mgv = re.search(r"\bg\s*=\s*([-\d\.]+)", prompt, re.IGNORECASE)
        marks = re.search(r"marks\s*=\s*(\w+)", prompt, re.IGNORECASE)

        v0 = float(mv0.group(1)) if mv0 else 20.0
        alpha = float(ma.group(1)) if ma else 45.0
        g = float(mgv.group(1)) if mgv else 9.8
        show_marks = _b(marks.group(1) if marks else "true")

        rad = math.radians(alpha)
        R = v0**2 * math.sin(2*rad) / g
        H = (v0**2 * (math.sin(rad))**2) / (2*g)

        needs_pgfplots = True
        body = rf"""
\begin{{tikzpicture}}
\begin{{axis}}[
  axis lines=left,
  xlabel=$x$ (m),
  ylabel=$y$ (m),
  domain=0:{_fmt(R*1.05)},
  samples=300,
  grid=both
]
\addplot {{
  {v0}*x*{math.sin(rad)} - 0.5*{g}*x^2/{(v0*math.cos(rad))**2}
}};
{"\\addplot[only marks] coordinates { (" + _fmt(R) + ",0) } node[pos=1, below] {Range};" if show_marks else ""}
{"\\addplot[only marks] coordinates { (" + _fmt(R/2) + "," + _fmt(H) + ") } node[pos=1, above] {Apex};" if show_marks else ""}
\end{{axis}}
\end{{tikzpicture}}
"""
        return body, needs_pgfplots, needs_circuitikz

    # Electromagnetism circuits
    if "mode=circuit" in p:
        needs_circuitikz = True
        topo_m = re.search(r"topology\s*=\s*(\w+)", prompt, re.IGNORECASE)
        topo = (topo_m.group(1).lower() if topo_m else "series")
        mV = re.search(r"\bV\s*=\s*([-\d\.]+)", prompt, re.IGNORECASE)
        V = float(mV.group(1)) if mV else 9.0
        mR = re.search(r"R\s*=\s*\[([^\]]*)\]", prompt, re.IGNORECASE)
        Rs = _parse_num_list(mR.group(1)) if mR else [100, 220, 330]
        mL = re.search(r"labels\s*=\s*(\w+)", prompt, re.IGNORECASE)
        show_labels = _b(mL.group(1) if mL else "true")

        if topo == "series":
            x2 = 0.0
            segs = []
            segs.append(f"\\begin{{circuitikz}}[european]")
            segs.append(f"  \\draw (0,0) to[battery1,l_={_fmt(V)} V] (0,3)")
            for i, R in enumerate(Rs, start=1):
                x2 += 2.5
                lab = f",l_=R{i}={_fmt(R)}\\\\$\\Omega$" if show_labels else ""
                segs.append(f"    to[R{lab}] ({_fmt(x2)},3)")
            segs.append(f"    -- ({_fmt(x2)},0) -- (0,0);")
            segs.append("\\end{circuitikz}")
            body = "\n".join(segs) + "\n"
            return body, needs_pgfplots, needs_circuitikz

        # parallel
        height = max(3.0, 1.5 + 1.2*len(Rs))
        segs = [f"\\begin{{circuitikz}}[european]",
                f"  \\draw (0,0) to[battery1,l_={_fmt(V)} V] (0,{_fmt(height)}) -- (5,{_fmt(height)});"]
        y = height - 1.0
        for i, R in enumerate(Rs, start=1):
            lab = f",l_=R{i}={_fmt(R)}\\\\$\\Omega$" if show_labels else ""
            segs.append(f"  \\draw (5,{_fmt(y)}) to[R{lab}] (5,{_fmt(y-1.0)}) -- (0,{_fmt(y-1.0)});")
            y -= 1.2
        segs.append("\\end{circuitikz}")
        body = "\n".join(segs) + "\n"
        return body, needs_pgfplots, needs_circuitikz

    return "% Unrecognized physics prompt", False, False

# ---------- Chemistry ----------
def gen_chem_benzene_directive(prompt: str) -> Tuple[str,bool]:
    msubs = re.search(r"subs\s*=\s*([^\n\r]+)$", prompt, re.IGNORECASE)
    subs_str = (msubs.group(1).strip() if msubs else "none")
    slots = ["","","","","",""]
    if subs_str.lower() != "none":
        parts = [p.strip() for p in subs_str.split(",")]
        for p in parts:
            mt = re.match(r"(\d+)\s*-\s*([A-Za-z0-9\+\-]+)", p)
            if mt:
                idx = max(1, min(6, int(mt.group(1)))) - 1
                slots[idx] = mt.group(2)
    segs = [("-"+s) if s else "-" for s in slots]
    ring = "*6(" + "".join(segs) + ")"
    body = rf"""
\centering
\chemfig{{{ring}}}
"""
    return body, False

def gen_chem_directive(prompt: str) -> Tuple[str,bool]:
    m = re.search(r"mode\s*=\s*(\w+)", prompt, re.IGNORECASE)
    mode = (m.group(1).lower() if m else "raw")
    if mode == "benzene":
        return gen_chem_benzene_directive(prompt)
    mm = re.search(r"mol\s*=\s*([A-Za-z0-9\-\+\(\)]+)", prompt, re.IGNORECASE)
    mol = mm.group(1) if mm else "H2O"
    body = rf"""
\centering
\chemfig{{{mol}}}
"""
    return body, False

# ---------- Dispatcher ----------
def generate_document(prompt: str, category_hint: str, subcategory_hint: Optional[str] = None) -> GenerationResult:
    """
    category_hint: one of "Mathematics"/"Math", "Statistics", "Physics", "Chemistry", or "Auto"
    If not "Auto", we trust the hint and route directly (no prefixes needed in prompt).
    """
    _map = {
        "Mathematics": "Math",
        "Math": "Math",
        "Statistics": "Statistics",
        "Physics": "Physics",
        "Chemistry": "Chemistry",
        "Auto": "Auto",
    }
    hint = _map.get(category_hint, "Auto")
    cat = detect_category(prompt) if hint == "Auto" else hint

    needs_pgfplots = False
    needs_circuitikz = False
    needs_chemfig = False

    if cat == "Math":
        # Prefer VectorAnalysis if the directive is used, else fallback to function plot
        if prompt.strip().lower().startswith("vectoranalysis:"):
            body, needs_pgfplots = gen_vector_analysis(prompt)
        else:
            body, needs_pgfplots = gen_math_plot(prompt)

    elif cat == "Statistics":
        body, needs_pgfplots = gen_statistics_directive(prompt)

    elif cat == "Physics":
        body, needs_pgfplots, needs_circuitikz = gen_physics_directive(prompt)

    elif cat == "Chemistry":
        body, needs_chemfig = gen_chem_directive(prompt)

    else:
        body = "% Unrecognized category"

    latex_full = _wrap_document(body, needs_pgfplots, needs_circuitikz, needs_chemfig)
    return GenerationResult(latex=latex_full, category=cat, summary="Generated successfully")
