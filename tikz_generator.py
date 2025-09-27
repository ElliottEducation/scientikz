# tikz_generator.py — ScienTikZ backend (Physics/Statistics/Chemistry enhanced)
import re, math
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

Category = Literal["Math","Physics","Statistics","Chemistry"]

# ---------------- Errors ----------------
class ValidationError(Exception):
    pass

@dataclass
class GenerationResult:
    latex: str
    category: Category
    summary: str

# ---------------- Utils ----------------
def _fmt(x) -> str:
    try:
        xv = float(x)
    except:
        return str(x)
    s = f"{xv:.6f}".rstrip("0").rstrip(".")
    return "0" if s == "-0" else s

def _b(s: Optional[str]) -> bool:
    return bool(s) and s.strip().lower() in {"1","true","yes","y","on"}

def ensure_range(val: float, lo: float, hi: float, name: str):
    if not (lo <= val <= hi):
        raise ValidationError(f"[{name}] out of range: {val} not in [{lo},{hi}]")

def ensure_len_eq(xs: List[float], ys: List[float], ctx="pairs"):
    if len(xs) == 0 or len(xs) != len(ys):
        raise ValidationError(f"[{ctx}] x/y length mismatch or empty")

def preflight_latex(body: str):
    pairs = [
        ("\\begin{tikzpicture}", "\\end{tikzpicture}"),
        ("\\begin{axis}", "\\end{axis}"),
        ("\\begin{circuitikz}", "\\end{circuitikz}"),
        ("\\schemestart", "\\schemestop"),
    ]
    for b,e in pairs:
        if body.count(b) != body.count(e):
            raise ValidationError(f"LaTeX env not balanced: {b} vs {e}")
    if body.count("{") != body.count("}"):
        raise ValidationError("Curly braces not balanced")
    if len(body) > 250_000:
        raise ValidationError("LaTeX body too large")

def detect_category(prompt: str) -> Category:
    p = prompt.lower()
    if any(k in p for k in ["mode=scatter","mode=hist","mode=regression","normal curve","boxplot","qq","mode=kde","mode=ecdf","mode=bar_error"]):
        return "Statistics"
    if any(k in p for k in ["vectoranalysis:","plot y="]):
        return "Math"
    if any(k in p for k in ["mode=incline","mode=projectile","mode=circuit","incline","projectile","circuit","mode=thin_lens","mode=rc_transient","mode=two_charges","thin lens","rc"]):
        return "Physics"
    if any(k in p for k in ["mode=benzene","mode=raw","chemfig","molecule","mode=reaction","reaction"]):
        return "Chemistry"
    return "Math"

def _wrap_document(body: str, needs_pgfplots: bool, needs_circuitikz: bool, needs_chemfig: bool) -> str:
    pkgs = ["\\documentclass{standalone}", "\\usepackage{tikz}"]
    if needs_pgfplots:
        pkgs += ["\\usepackage{pgfplots}", "\\pgfplotsset{compat=1.18}"]
    if needs_circuitikz:
        pkgs += ["\\usepackage[europeanresistors]{circuitikz}"]
    if needs_chemfig:
        pkgs += ["\\usepackage{chemfig}"]
    pre = "\n".join(pkgs)
    return pre + "\n\\begin{document}\n" + body + "\n\\end{document}\n"

def _extract(text: str, pat: str) -> Optional[str]:
    m = re.search(pat, text, re.IGNORECASE)
    return m.group(1) if m else None

def _parse_list_numbers(txt: str) -> List[float]:
    if not txt: return []
    txt = txt.strip().strip("[]")
    if not txt: return []
    out = []
    for t in txt.split(","):
        t = t.strip()
        try: out.append(float(t))
        except: pass
    return out

# ===================== Math (kept minimal here; your app already has math module) =====================
def render_math_function(expr: str, x0: float, x1: float) -> Tuple[str,bool]:
    if x0 >= x1: raise ValidationError("range invalid: left bound must be < right bound")
    expr = expr.replace("sin(x)", "sin(deg(x))").replace("cos(x)", "cos(deg(x))")
    body = rf"""
\begin{{tikzpicture}}
\begin{{axis}}[axis lines=middle, xlabel=$x$, ylabel=$y$, domain={_fmt(x0)}:{_fmt(x1)}, samples=300, grid=both]
\addplot {{{expr}}};
\end{{axis}}
\end{{tikzpicture}}
"""
    preflight_latex(body); return body, True

def parse_math_en(text: str) -> Dict[str, Any]:
    t = text.strip()
    m = re.search(r"plot\s+y\s*=\s*([^;]+?)\s+from\s+([-\d\.]+)\s+to\s+([-\d\.]+)", t, re.IGNORECASE)
    if m:
        return {"task":"math_function","expr":m.group(1).strip(),"x0":float(m.group(2)),"x1":float(m.group(3))}
    raise ValidationError("Try: Plot y = sin(x) from -6.28 to 6.28")

# ===================== Statistics =====================
def render_stats_scatter(xs: List[float], ys: List[float]) -> Tuple[str,bool]:
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
    preflight_latex(body); return body, True

def render_stats_hist(data: List[float], bins: int=10) -> Tuple[str,bool]:
    ensure_range(bins, 1, 200, "bins")
    lines = "\n".join(_fmt(v) for v in data)
    body = rf"""
\begin{{tikzpicture}}
\begin{{axis}}[ybar interval, ymin=0, xlabel=Value, ylabel=Count, grid=both]
\addplot+[hist={{bins={bins}}}] table[row sep=\\\\, y index=0]{{
{lines}
}};
\end{{axis}}
\end{{tikzpicture}}
"""
    preflight_latex(body); return body, True

def render_stats_regression(xs: List[float], ys: List[float]) -> Tuple[str,bool]:
    ensure_len_eq(xs, ys, "regression")
    n = len(xs); sx=sum(xs); sy=sum(ys); sxx=sum(x*x for x in xs); sxy=sum(x*y for x,y in zip(xs,ys))
    denom = n*sxx - sx*sx
    m = 0.0 if abs(denom)<1e-12 else (n*sxy - sx*sy)/denom
    b = (sy - m*sx)/n
    ybar = sy/n; ss_tot=sum((y-ybar)**2 for y in ys); ss_res=sum((y-(m*x+b))**2 for x,y in zip(xs,ys))
    r2 = 1.0 - (ss_res/ss_tot if ss_tot!=0 else 0.0)
    pairs = "\n".join(f"{_fmt(a)}\t{_fmt(b)}" for a,b in zip(xs,ys))
    body = rf"""
\begin{{tikzpicture}}
\begin{{axis}}[axis lines=left, xlabel=$x$, ylabel=$y$, grid=both]
\addplot[only marks, mark=*] table[row sep=crcr]{{
{pairs}
}};
\addplot[domain={_fmt(min(xs))}:{_fmt(max(xs))}, samples=2] {{ {_fmt(m)}*x + {_fmt(b)} }};
\node[anchor=south west] at (rel axis cs:0.02,0.98) {{${{y = {_fmt(m)}x + {_fmt(b)}}}\;\;R^2={_fmt(r2)}$}};
\end{{axis}}
\end{{tikzpicture}}
"""
    preflight_latex(body); return body, True

def render_stats_normal(mu: float, sigma: float, a: float, b: float) -> Tuple[str,bool]:
    ensure_range(sigma, 1e-9, 1e9, "sigma"); 
    if a >= b: raise ValidationError("range invalid: left bound must be < right bound")
    body = rf"""
\begin{{tikzpicture}}
\begin{{axis}}[axis lines=left, xlabel=$x$, ylabel=$f(x)$, domain={_fmt(a)}:{_fmt(b)}, samples=300, grid=both]
\addplot {{ 1/({_fmt(sigma)}*sqrt(2*pi)) * exp(-0.5*((x-{_fmt(mu)})/{_fmt(sigma)})^2) }};
\end{{axis}}
\end{{tikzpicture}}
"""
    preflight_latex(body); return body, True

def render_stats_boxplot(data: List[float]) -> Tuple[str,bool]:
    if len(data) < 4: raise ValidationError("boxplot needs >=4 observations")
    data = sorted(data)
    def q(p):
        idx = (len(data)-1)*p; lo=int(math.floor(idx)); hi=int(math.ceil(idx))
        return data[lo] if lo==hi else data[lo]*(hi-idx)+data[hi]*(idx-lo)
    q1,q2,q3 = q(0.25), q(0.5), q(0.75); iqr=q3-q1
    lo = min(x for x in data if x>=q1-1.5*iqr); hi = max(x for x in data if x<=q3+1.5*iqr)
    body = rf"""
\begin{{tikzpicture}}
\begin{{axis}}[ytick=\empty, xmin={_fmt(min(data)-0.5)}, xmax={_fmt(max(data)+0.5)}, height=3.2cm, axis lines=left, xlabel=Value, grid=both]
\addplot+[boxplot prepared={{lower whisker={_fmt(lo)}, lower quartile={_fmt(q1)}, median={_fmt(q2)}, upper quartile={_fmt(q3)}, upper whisker={_fmt(hi)}}}] coordinates {{ }};
\end{{axis}}
\end{{tikzpicture}}
"""
    preflight_latex(body); return body, True

# ---- KDE & ECDF & Bar+Error ----
def _linspace(a: float, b: float, n: int) -> List[float]:
    if n<=1: return [a]
    step=(b-a)/(n-1); return [a+i*step for i in range(n)]

def render_stats_kde(data: List[float], bw: float=0.4, lo: Optional[float]=None, hi: Optional[float]=None) -> Tuple[str,bool]:
    if len(data) < 2: raise ValidationError("kde needs >=2 data points")
    if bw <= 0: raise ValidationError("bandwidth must be >0")
    lo = min(data) - 3*bw if lo is None else lo
    hi = max(data) + 3*bw if hi is None else hi
    xs = _linspace(lo, hi, 200)
    const = 1/(math.sqrt(2*math.pi)*bw*len(data))
    def kde(x):
        return const*sum(math.exp(-0.5*((x-xi)/bw)**2) for xi in data)
    pairs = "\n".join(f"{_fmt(x)}\t{_fmt(kde(x))}" for x in xs)
    body = rf"""
\begin{{tikzpicture}}
\begin{{axis}}[axis lines=left, grid=both, xlabel=$x$, ylabel=Density]
\addplot[smooth] table[row sep=crcr]{{
{pairs}
}};
\end{{axis}}
\end{{tikzpicture}}
"""
    preflight_latex(body); return body, True

def render_stats_ecdf(data: List[float]) -> Tuple[str,bool]:
    if len(data) < 2: raise ValidationError("ecdf needs >=2 observations")
    data = sorted(data)
    n = len(data)
    pairs = "\n".join(f"{_fmt(x)}\t{_fmt((i+1)/n)}" for i,x in enumerate(data))
    body = rf"""
\begin{{tikzpicture}}
\begin{{axis}}[axis lines=left, grid=both, xlabel=$x$, ylabel=ECDF]
\addplot[const plot] table[row sep=crcr]{{
{pairs}
}};
\end{{axis}}
\end{{tikzpicture}}
"""
    preflight_latex(body); return body, True

def render_stats_bar_error(means: List[float], errors: List[float], labels: Optional[List[str]]=None) -> Tuple[str,bool]:
    if len(means)==0 or len(means)!=len(errors):
        raise ValidationError("bar_error: means and errors length mismatch")
    xs = list(range(1, len(means)+1))
    labline = f"xtick={{ {', '.join(str(x) for x in xs)} }}, xticklabels={{ {', '.join(labels) if labels else ', '.join([str(i) for i in xs])} }}" 
    mean_pairs = "\n".join(f"{x}\t{_fmt(m)}" for x,m in zip(xs,means))
    err_pairs  = "\n".join(f"{x}\t{_fmt(e)}" for x,e in zip(xs,errors))
    body = rf"""
\begin{{tikzpicture}}
\begin{{axis}}[ymin=0, grid=both, xlabel=Group, ylabel=Value, {labline}]
\addplot[ybar, fill=gray!40] table[row sep=crcr]{{
{mean_pairs}
}};
\addplot+[only marks, error bars/.cd, y dir=both, y explicit] table[x index=0, y index=1, y error expr=\thisrowno{{2}}] {{
{'\n'.join(f'{x}\t{_fmt(m)}\t{_fmt(e)}' for x,m,e in zip(xs,means,errors))}
}};
\end{{axis}}
\end{{tikzpicture}}
"""
    preflight_latex(body); return body, True

def parse_stats_en(text: str) -> Dict[str, Any]:
    t = text
    # histogram
    mh = re.search(r"(?:hist|histogram).*(?:with\s+([0-9]+)\s+bins?)?.*?(?:data|values?)\s*([\[\(][^\]\)]*[\]\)])", t, re.IGNORECASE)
    if mh:
        bins = int(mh.group(1)) if mh.group(1) else 10
        nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", mh.group(2))]
        return {"task":"hist","bins":bins,"data":nums}
    # scatter
    ms = re.search(r"scatter.*?x[:=]\s*[\[\(]([^\]\)]*)[\]\)].*?y[:=]\s*[\[\(]([^\]\)]*)[\]\)]", t, re.IGNORECASE)
    if ms:
        xs = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", ms.group(1))]
        ys = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", ms.group(2))]
        return {"task":"scatter","x":xs,"y":ys}
    # regression
    mr = re.search(r"(?:linear\s+)?regression.*?x[:=]\s*[\[\(]([^\]\)]*)[\]\)].*?y[:=]\s*[\[\(]([^\]\)]*)[\]\)]", t, re.IGNORECASE)
    if mr:
        xs = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", mr.group(1))]
        ys = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", mr.group(2))]
        return {"task":"regression","x":xs,"y":ys}
    # normal curve
    mn = re.search(r"normal\s+curve.*?mu\s*=\s*([-\d\.]+).*?sigma\s*=\s*([-\d\.]+).*?from\s+([-\d\.]+)\s+to\s+([-\d\.]+)", t, re.IGNORECASE)
    if mn:
        return {"task":"normal","mu":float(mn.group(1)),"sigma":float(mn.group(2)),"a":float(mn.group(3)),"b":float(mn.group(4))}
    # boxplot
    if re.search(r"\bbox\s*plot|\bboxplot", t, re.IGNORECASE) or re.search(r"\bboxplot|box\s*plot|boxplot", t, re.IGNORECASE):
        nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", t)]
        return {"task":"boxplot","data":nums}
    # qq
    if re.search(r"(qq\s*plot|normal\s*probability\s*plot)", t, re.IGNORECASE):
        nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", t)]
        return {"task":"qq","data":nums}
    # kde
    mk = re.search(r"kde.*?data\s*[:=]\s*([\[\(][^\]\)]*[\]\)]).*(?:bandwidth\s*[:=]\s*([-\d\.]+))?", t, re.IGNORECASE)
    if mk:
        nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", mk.group(1))]
        bw = float(mk.group(2)) if mk.group(2) else 0.4
        return {"task":"kde","data":nums,"bw":bw}
    # ecdf
    me = re.search(r"ecdf.*?data\s*[:=]\s*([\[\(][^\]\)]*[\]\)])", t, re.IGNORECASE)
    if me:
        nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", me.group(1))]
        return {"task":"ecdf","data":nums}
    # bar+error
    mbe = re.search(r"bar.*?means\s*[:=]\s*([\[\(][^\]\)]*[\]\)]).*(?:errors\s*[:=]\s*([\[\(][^\]\)]*[\]\)]))", t, re.IGNORECASE)
    if mbe:
        means = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", mbe.group(1))]
        errors = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", mbe.group(2))]
        labels = re.findall(r"labels\s*[:=]\s*\[([^\]]*)\]", t, re.IGNORECASE)
        labs = [x.strip() for x in labels[0].split(",")] if labels else None
        return {"task":"bar_error","means":means,"errors":errors,"labels":labs}
    raise ValidationError("Unrecognized statistics instruction.")

# ===================== Physics =====================
def render_incline(theta: float, mass: float, mu: float, components: bool=True, friction: bool=True, tension: bool=False) -> Tuple[str,bool,bool]:
    ensure_range(theta, 0, 85, "theta"); ensure_range(mu, 0, 2, "mu")
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
    preflight_latex(body); return body, False, False

def render_projectile(v0: float, alpha: float, g: float=9.8, marks: bool=True) -> Tuple[str,bool,bool]:
    ensure_range(alpha, 0, 89, "alpha"); ensure_range(g, 0.1, 100, "g")
    rad=math.radians(alpha); R=v0**2*math.sin(2*rad)/g; H=(v0**2*(math.sin(rad))**2)/(2*g)
    body = rf"""
\begin{{tikzpicture}}
\begin{{axis}}[axis lines=left, xlabel=$x$ (m), ylabel=$y$ (m), domain=0:{_fmt(R*1.05)}, samples=300, grid=both]
\addplot {{ {v0}*x*{math.sin(rad)} - 0.5*{g}*x^2/{(v0*math.cos(rad))**2} }};
{"\\addplot[only marks] coordinates { (" + _fmt(R) + ",0) } node[pos=1, below] {Range};" if marks else ""}
{"\\addplot[only marks] coordinates { (" + _fmt(R/2) + "," + _fmt(H) + ") } node[pos=1, above] {Apex};" if marks else ""}
\end{{axis}}
\end{{tikzpicture}}
"""
    preflight_latex(body); return body, True, False

def render_circuit(topology: str, V: float, Rs: List[float], labels: bool=True, show_current: bool=True, voltmeter: bool=False, ammeter: bool=False) -> Tuple[str,bool,bool]:
    if not Rs: raise ValidationError("R list empty")
    if topology.lower()=="series":
        x2=0.0; segs=["\\begin{circuitikz}[european]", f"  \\draw (0,0) to[battery1,l_={_fmt(V)} V] (0,3)"]
        last="(0,3)"
        for i,R in enumerate(Rs,1):
            x2+=2.5; lab=f",l_=R{i}={_fmt(R)}\\\\$\\Omega$" if labels else ""; curr=f", i>^=I{i}" if show_current else ""
            segs.append(f"    to[R{lab}{curr}] ({_fmt(x2)},3)"); last=f"({_fmt(x2)},3)"
        if voltmeter: segs.append(f"    to[voltmeter] ({_fmt(x2+1.8)},3)"); last=f"({_fmt(x2+1.8)},3)"
        if ammeter:  segs.append(f"    to[ammeter] ({_fmt(x2+3.2)},3)");  last=f"({_fmt(x2+3.2)},3)"
        segs.append(f"    -- {last.replace(',3',',0')} -- (0,0);"); segs.append("\\end{circuitikz}")
        body="\n".join(segs)+"\n"; preflight_latex(body); return body, False, True
    # parallel
    h=max(3.0, 1.5+1.2*len(Rs)); segs=[f"\\begin{{circuitikz}}[european]", f"  \\draw (0,0) to[battery1,l_={_fmt(V)} V] (0,{_fmt(h)}) -- (5,{_fmt(h)});"]
    y=h-1.0
    for i,R in enumerate(Rs,1):
        lab=f",l_=R{i}={_fmt(R)}\\\\$\\Omega$" if labels else ""; curr=f", i>^=I{i}" if show_current else ""
        segs.append(f"  \\draw (5,{_fmt(y)}) to[R{lab}{curr}] (5,{_fmt(y-1.0)}) -- (0,{_fmt(y-1.0)});"); y-=1.2
    if voltmeter: segs.append(f"  \\draw (5,{_fmt(h)}) to[voltmeter] (7,{_fmt(h)});")
    if ammeter:  segs.append(f"  \\draw (0,{_fmt(h)}) to[ammeter] (-2,{_fmt(h)});")
    segs.append("\\end{circuitikz}"); body="\n".join(segs)+"\n"; preflight_latex(body); return body, False, True

# ---- New: Thin lens ray diagram ----
def render_thin_lens(f: float, u: float, show_labels: bool=True) -> Tuple[str,bool,bool]:
    if f==0: raise ValidationError("f must not be 0")
    v = 1/(1/f - 1/u)  # lens formula 1/f = 1/u + 1/v
    m = -v/u
    # choose scale so that drawing fits nicely
    s = 1.0
    body = rf"""
\begin{{tikzpicture}}[>=stealth, scale={_fmt(s)}]
  \draw[->] (-8,0)--(8,0) node[right]{{Optical axis}};
  \draw (0,-3)--(0,3); \node at (0,3.3) {{Lens}};
  \draw[<->] ({_fmt(-f)},0.15)--({-_fmt(-f)},0.15) node[midway, above]{{$2f$}};
  \fill ( {_fmt(f)},0) circle (1.2pt) node[below]{{$F$}};
  \fill ( {-_fmt(f)},0) circle (1.2pt) node[below]{{$F'$}};
  % object
  \draw[thick] ({-_fmt(u)},0) -- ++(0,2) node[above]{{$O$}};
  % rays
  \draw[->,red] ({-_fmt(u)},2) -- (0,2) -- ({_fmt(v)}, { _fmt(-2*m) });
  \draw[->,red] ({-_fmt(u)},2) -- (0,0);
  \draw[red, dashed] (0,0) -- ({_fmt(v)}, { _fmt(-2*m) });
  % image
  \draw[thick, blue] ({_fmt(v)},0) -- ++(0,{ _fmt(-2*m) }) node[below]{{$I$}};
  {"\\node at (4.6,2.6) {$v=" + _fmt(v) + ",\\; m=" + _fmt(m) + "$};" if show_labels else ""}
\end{{tikzpicture}}
"""
    preflight_latex(body); return body, False, False

# ---- New: RC transient ----
def render_rc_transient(R: float, C: float, V0: float, tmax: float, annotate: bool=True) -> Tuple[str,bool,bool]:
    if R<=0 or C<=0: raise ValidationError("R and C must be >0")
    tau = R*C
    xs = _linspace(0.0, tmax, 200)
    pairs = "\n".join(f"{_fmt(t)}\t{_fmt(V0*(1-math.exp(-t/tau)))}" for t in xs)
    body = rf"""
\begin{{tikzpicture}}
\begin{{axis}}[axis lines=left, grid=both, xlabel=$t$ (s), ylabel=$v_c(t)$ (V)]
\addplot[smooth] table[row sep=crcr]{{
{pairs}
}};
{"\\addplot[only marks] coordinates { (" + _fmt(tau) + "," + _fmt(V0*(1-math.exp(-1))) + ") } node[pos=1, above] {$\\tau$};" if annotate else ""}
\end{{axis}}
\end{{tikzpicture}}
"""
    preflight_latex(body); return body, True, False

# ---- New: two point charges (+q and -q) simple field sketch ----
def render_two_charges(q: float=1.0, d: float=2.0) -> Tuple[str,bool,bool]:
    # schematic (not rigorous field integration) to illustrate lines
    a = d/2
    body = rf"""
\begin{{tikzpicture}}[scale=1.0]
  \fill[red] ({-_fmt(a)},0) circle (2pt) node[below] {{+$q$}};
  \fill[blue] ({_fmt(a)},0) circle (2pt) node[below] {{$-q$}};
  % a few hand-drawn streamlines
  \foreach \y in {{-2.5,-1.5,-0.8,0.8,1.5,2.5}} {{
     \draw[->, gray] ({-_fmt(a)},0) .. controls ({-_fmt(a+0.8)},\y) and ({_fmt(a-0.8)},\y) .. ({_fmt(a)},0);
  }}
  \draw[->] ({-_fmt(a)},0) --++(-2,0);
  \draw[->] ({_fmt(a)},0) --++(2,0);
\end{{tikzpicture}}
"""
    preflight_latex(body); return body, False, False

def parse_physics_en(text: str) -> Dict[str, Any]:
    t = text.lower().replace("°","").strip()
    if "incline" in t or "inclined" in t or "fbd" in t:
        th = _first_num(text, r"(?:angle|alpha|theta)\s*=?\s*([-\d\.]+)") or 30.0
        mass= _first_num(text, r"(?:mass|m)\s*=?\s*([-\d\.]+)") or 2.0
        mu  = _first_num(text, r"(?:friction|mu)\s*=?\s*([-\d\.]+)") or 0.2
        return {"task":"incline","theta":th,"mass":mass,"mu":mu,"components":True,"friction":True,"tension":("tension" in t)}
    if "projectile" in t or "trajectory" in t:
        v0 = _first_num(text, r"v0\s*=\s*([-\d\.]+)") or _first_num(text, r"speed\s*([-\d\.]+)") or 20.0
        alpha = _first_num(text, r"(?:angle|alpha)\s*=?\s*([-\d\.]+)") or 45.0
        g = _first_num(text, r"\bg\s*=\s*([-\d\.]+)") or 9.8
        return {"task":"projectile","v0":v0,"alpha":alpha,"g":g,"marks":True}
    if "circuit" in t:
        topo = "parallel" if "parallel" in t else "series"
        V = _first_num(text, r"([-\d\.]+)\s*v") or 9.0
        Rs_match = re.search(r"(?:resistors?|r)\s*(?:=|:)?\s*([0-9\.\,\s]+)", text, re.IGNORECASE)
        Rs = [float(x) for x in Rs_match.group(1).split(",")] if Rs_match else [100,220,330]
        return {"task":"circuit","topology":topo,"V":V,"R":Rs,"labels":True,"show_current":True,"voltmeter":("voltmeter" in t),"ammeter":("ammeter" in t)}
    if "thin lens" in t or "lens" in t:
        f = _first_num(text, r"f\s*=\s*([-\d\.]+)") or 2.0
        u = _first_num(text, r"(?:u|object)\s*=?\s*([-\d\.]+)") or 5.0
        return {"task":"thin_lens","f":f,"u":u,"show_labels":True}
    if "rc" in t:
        R = _first_num(text, r"r\s*=\s*([-\d\.]+)") or 1000.0
        C = _first_num(text, r"c\s*=\s*([-\d\.]+)") or 0.001
        V0= _first_num(text, r"v0\s*=\s*([-\d\.]+)") or 5.0
        tmax = _first_num(text, r"tmax\s*=\s*([-\d\.]+)") or (5*R*C)
        return {"task":"rc","R":R,"C":C,"V0":V0,"tmax":tmax,"annotate":True}
    if "two charges" in t or "dipole" in t:
        q = _first_num(text, r"q\s*=\s*([-\d\.]+)") or 1.0
        d = _first_num(text, r"d\s*=\s*([-\d\.]+)") or 2.0
        return {"task":"two_charges","q":q,"d":d}
    raise ValidationError("Unrecognized physics instruction.")

def _first_num(text: str, pat: str) -> Optional[float]:
    m = re.search(pat, text, re.IGNORECASE); 
    return float(m.group(1)) if m else None

# ===================== Chemistry =====================
def render_benzene(subs: List[str]) -> Tuple[str,bool]:
    slots = ["","","","","",""]
    for s in subs:
        m = re.match(r"(\d+)\s*-\s*([A-Za-z0-9\+\-]+)", s)
        if m:
            idx = max(1, min(6, int(m.group(1))))-1
            slots[idx] = m.group(2)
    ring = "*6(" + "".join(("-"+s) if s else "-" for s in slots) + ")"
    body = rf"\chemfig{{{ring}}}"
    preflight_latex(body); return body, False

def render_molecule(mol: str) -> Tuple[str,bool]:
    if not re.match(r"^[A-Za-z0-9\-\+\(\)]+$", mol):
        raise ValidationError("Invalid chemfig token")
    body = rf"\chemfig{{{mol}}}"
    preflight_latex(body); return body, False

# ---- New: reaction scheme (chemfig) ----
def render_reaction(left: str, right: str, above: str="", below: str="") -> Tuple[str,bool]:
    body = rf"""
\schemestart
\chemfig{{{left}}}
\arrow{{->[{above}][{below}]}}
\chemfig{{{right}}}
\schemestop
"""
    preflight_latex(body); return body, True  # needs chemfig; pgfplots not required but safe

def parse_chem_en(text: str) -> Dict[str, Any]:
    t = text.lower()
    if "benzene" in t or "ring" in t:
        subs=[]
        for chem,pos in re.findall(r"([A-Za-z0-9\+\-]{1,10})\s+at\s+(?:position\s+)?([1-6])", text, re.IGNORECASE):
            subs.append(f"{pos}-{chem}")
        if not subs:
            parts = re.findall(r"([1-6]\s*-\s*[A-Za-z0-9\+\-]{1,10})", text); subs=[p.replace(" ","") for p in parts] or ["1-CH3"]
        return {"task":"benzene","subs":subs}
    if "reaction" in t or "->" in t:
        mleft = _extract(text, r"reaction\s*:\s*([^-\n\r]+?)\s*->") or _extract(text, r"^\s*([^-\n\r]+?)\s*->")
        mright= _extract(text, r"->\s*([^\n\r]+)")
        above = _extract(text, r"above\s*[:=]\s*([^\n\r]+)") or ""
        below = _extract(text, r"below\s*[:=]\s*([^\n\r]+)") or ""
        if mleft and mright: return {"task":"reaction","left":mleft.strip(), "right":mright.strip(), "above":above.strip(), "below":below.strip()}
    # molecule
    m = re.search(r"(?:molecule\s*)?([A-Za-z0-9\-\+\(\)]+)", text)
    if m: return {"task":"molecule","mol":m.group(1)}
    raise ValidationError("Unrecognized chemistry instruction.")

# ===================== Dispatcher (Directive/NL) =====================
def _render_stats_by_ir(ir: Dict[str, Any]) -> Tuple[str,bool]:
    t = ir["task"]
    if t=="hist": return render_stats_hist(ir["data"], ir.get("bins",10))
    if t=="scatter": return render_stats_scatter(ir["x"], ir["y"])
    if t=="regression": return render_stats_regression(ir["x"], ir["y"])
    if t=="normal": return render_stats_normal(ir["mu"], ir["sigma"], ir["a"], ir["b"])
    if t=="boxplot": return render_stats_boxplot(ir["data"])
    if t=="qq":
        # reuse ECDF x vs theor quantiles? (already available in previous version)
        data = sorted(ir["data"])
        n=len(data)
        def inv_norm_cdf(p): return math.sqrt(2)*_inv_erf(2*p-1)
        xs=[inv_norm_cdf((i-0.5)/n) for i in range(1,n+1)]
        pairs="\n".join(f"{_fmt(a)}\t{_fmt(b)}" for a,b in zip(xs,data))
        body = rf"""
\begin{{tikzpicture}}
\begin{{axis}}[axis equal image, grid=both, xlabel=Theoretical $N(0,1)$, ylabel=Sample]
\addplot[only marks, mark=*] table[row sep=crcr]{{
{pairs}
}};
\addplot[domain={_fmt(min(xs))}:{_fmt(max(xs))}, samples=2] {{ x }};
\end{{axis}}
\end{{tikzpicture}}
"""
        preflight_latex(body); return body, True
    if t=="kde": return render_stats_kde(ir["data"], ir.get("bw",0.4), ir.get("lo"), ir.get("hi"))
    if t=="ecdf": return render_stats_ecdf(ir["data"])
    if t=="bar_error": return render_stats_bar_error(ir["means"], ir["errors"], ir.get("labels"))
    raise ValidationError(f"Unsupported stats IR: {t}")

def _inv_erf(x):
    a=0.147; s=1 if x>=0 else -1; ln=math.log(1-x*x); first=2/(math.pi*a)+ln/2
    return s*math.sqrt(math.sqrt(first*first - ln/a) - first)

def _render_physics_by_ir(ir: Dict[str, Any]) -> Tuple[str,bool,bool]:
    t=ir["task"]
    if t=="incline": return render_incline(ir["theta"],ir["mass"],ir["mu"], ir.get("components",True), ir.get("friction",True), ir.get("tension",False))
    if t=="projectile": return render_projectile(ir["v0"],ir["alpha"],ir.get("g",9.8), ir.get("marks",True))
    if t=="circuit": return render_circuit(ir["topology"], ir["V"], ir["R"], ir.get("labels",True), ir.get("show_current",True), ir.get("voltmeter",False), ir.get("ammeter",False))
    if t=="thin_lens": return render_thin_lens(ir["f"], ir["u"], ir.get("show_labels",True))
    if t=="rc": return render_rc_transient(ir["R"], ir["C"], ir["V0"], ir["tmax"], ir.get("annotate",True))
    if t=="two_charges": return render_two_charges(ir.get("q",1.0), ir.get("d",2.0))
    raise ValidationError(f"Unsupported physics IR: {t}")

def _render_chem_by_ir(ir: Dict[str, Any]) -> Tuple[str,bool]:
    t=ir["task"]
    if t=="benzene": return render_benzene(ir["subs"])
    if t=="molecule": return render_molecule(ir["mol"])
    if t=="reaction": return render_reaction(ir["left"], ir["right"], ir.get("above",""), ir.get("below",""))
    raise ValidationError(f"Unsupported chemistry IR: {t}")

# -------- Main entry used by app (directive path) --------
def generate_document(prompt: str, category_hint: str, subcategory_hint: Optional[str]=None) -> GenerationResult:
    hint = {"Mathematics":"Math","Math":"Math","Statistics":"Statistics","Physics":"Physics","Chemistry":"Chemistry","Auto":"Auto"}.get(category_hint,"Auto")
    cat = detect_category(prompt) if hint=="Auto" else hint

    needs_pgf=False; needs_circ=False; needs_chem=False

    if cat=="Math":
        if prompt.strip().lower().startswith("plot y"):
            m=parse_math_en(prompt); body,needs_pgf = render_math_function(m["expr"], m["x0"], m["x1"])
        else:
            m=parse_math_en(prompt); body,needs_pgf = render_math_function(m["expr"], m["x0"], m["x1"])

    elif cat=="Statistics":
        p=prompt.lower()
        if "mode=scatter" in p:
            xs=_parse_list_numbers(_extract(prompt, r"x\s*=\s*\[([^\]]*)\]") or ""); ys=_parse_list_numbers(_extract(prompt, r"y\s*=\s*\[([^\]]*)\]") or "")
            body,needs_pgf = render_stats_scatter(xs,ys)
        elif "mode=hist" in p:
            data=_parse_list_numbers(_extract(prompt, r"data\s*=\s*\[([^\]]*)\]") or ""); bins=int(_extract(prompt, r"bins\s*=\s*([0-9]+)") or 10)
            body,needs_pgf = render_stats_hist(data,bins)
        elif "mode=regression" in p:
            xs=_parse_list_numbers(_extract(prompt, r"x\s*=\s*\[([^\]]*)\]") or ""); ys=_parse_list_numbers(_extract(prompt, r"y\s*=\s*\[([^\]]*)\]") or "")
            body,needs_pgf = render_stats_regression(xs,ys)
        elif "boxplot" in p:
            data=_parse_list_numbers(_extract(prompt, r"data\s*=\s*\[([^\]]*)\]") or "")
            body,needs_pgf = render_stats_boxplot(data)
        elif "qq" in p:
            data=_parse_list_numbers(_extract(prompt, r"data\s*=\s*\[([^\]]*)\]") or "")
            body,needs_pgf = _render_stats_by_ir({"task":"qq","data":data})
        elif "mode=kde" in p:
            data=_parse_list_numbers(_extract(prompt, r"data\s*=\s*\[([^\]]*)\]") or ""); bw=float(_extract(prompt, r"bandwidth\s*=\s*([-\d\.]+)") or 0.4)
            lo = _extract(prompt, r"range\s*=\s*\[\s*([-\d\.]+)") ; hi=_extract(prompt, r"range\s*=\s*\[\s*[-\d\.]+\s*,\s*([-\d\.]+)\s*\]")
            body,needs_pgf = render_stats_kde(data,bw, float(lo) if lo else None, float(hi) if hi else None)
        elif "mode=ecdf" in p:
            data=_parse_list_numbers(_extract(prompt, r"data\s*=\s*\[([^\]]*)\]") or "")
            body,needs_pgf = render_stats_ecdf(data)
        elif "mode=bar_error" in p:
            means=_parse_list_numbers(_extract(prompt, r"means\s*=\s*\[([^\]]*)\]") or "")
            errors=_parse_list_numbers(_extract(prompt, r"errors\s*=\s*\[([^\]]*)\]") or "")
            labs_raw=_extract(prompt, r"labels\s*=\s*\[([^\]]*)\]")
            labels=[s.strip() for s in labs_raw.split(",")] if labs_raw else None
            body,needs_pgf = render_stats_bar_error(means,errors,labels)
        else:
            ir=parse_stats_en(prompt); body,needs_pgf=_render_stats_by_ir(ir)

    elif cat=="Physics":
        p=prompt.lower()
        if "mode=incline" in p:
            body,needs_pgf,needs_circ = render_incline(float(_extract(prompt, r"theta\s*=\s*([-\d\.]+)") or 30.0),
                                                       float(_extract(prompt, r"mass\s*=\s*([-\d\.]+)") or 2.0),
                                                       float(_extract(prompt, r"mu\s*=\s*([-\d\.]+)") or 0.2),
                                                       _b(_extract(prompt, r"components\s*=\s*(\w+)") or "true"),
                                                       _b(_extract(prompt, r"friction\s*=\s*(\w+)") or "true"),
                                                       _b(_extract(prompt, r"tension\s*=\s*(\w+)") or "false"))
        elif "mode=projectile" in p:
            body,needs_pgf,needs_circ = render_projectile(float(_extract(prompt, r"v0\s*=\s*([-\d\.]+)") or 20.0),
                                                          float(_extract(prompt, r"alpha\s*=\s*([-\d\.]+)") or 45.0),
                                                          float(_extract(prompt, r"g\s*=\s*([-\d\.]+)") or 9.8),
                                                          _b(_extract(prompt, r"marks\s*=\s*(\w+)") or "true"))
        elif "mode=circuit" in p:
            Rs=_parse_list_numbers(_extract(prompt, r"R\s*=\s*\[([^\]]*)\]") or "")
            body,needs_pgf,needs_circ = render_circuit((_extract(prompt, r"topology\s*=\s*(\w+)") or "series"),
                                                        float(_extract(prompt, r"V\s*=\s*([-\d\.]+)") or 9.0),
                                                        Rs if Rs else [100,220,330],
                                                        _b(_extract(prompt, r"labels\s*=\s*(\w+)") or "true"),
                                                        _b(_extract(prompt, r"show_current\s*=\s*(\w+)") or "true"),
                                                        _b(_extract(prompt, r"voltmeter\s*=\s*(\w+)") or "false"),
                                                        _b(_extract(prompt, r"ammeter\s*=\s*(\w+)") or "false"))
        elif "mode=thin_lens" in p:
            body,needs_pgf,needs_circ = render_thin_lens(float(_extract(prompt, r"f\s*=\s*([-\d\.]+)") or 2.0),
                                                         float(_extract(prompt, r"u\s*=\s*([-\d\.]+)") or 5.0),
                                                         _b(_extract(prompt, r"show_labels\s*=\s*(\w+)") or "true"))
        elif "mode=rc_transient" in p:
            body,needs_pgf,needs_circ = render_rc_transient(float(_extract(prompt, r"R\s*=\s*([-\d\.]+)") or 1000.0),
                                                            float(_extract(prompt, r"C\s*=\s*([-\d\.]+)") or 0.001),
                                                            float(_extract(prompt, r"V0\s*=\s*([-\d\.]+)") or 5.0),
                                                            float(_extract(prompt, r"tmax\s*=\s*([-\d\.]+)") or 0.02),
                                                            _b(_extract(prompt, r"annotate\s*=\s*(\w+)") or "true"))
        elif "mode=two_charges" in p:
            body,needs_pgf,needs_circ = render_two_charges(float(_extract(prompt, r"q\s*=\s*([-\d\.]+)") or 1.0),
                                                           float(_extract(prompt, r"d\s*=\s*([-\d\.]+)") or 2.0))
        else:
            ir=parse_physics_en(prompt); body,needs_pgf,needs_circ=_render_physics_by_ir(ir)

    elif cat=="Chemistry":
        p=prompt.lower()
        if "mode=benzene" in p:
            subs_raw=_extract(prompt, r"subs\s*=\s*([^\n\r]+)") or "none"
            subs=[] if subs_raw.strip().lower()=="none" else [s.strip() for s in subs_raw.split(",")]
            body,needs_chem = render_benzene(subs)
        elif "mode=raw" in p:
            mol=_extract(prompt, r"mol\s*=\s*([A-Za-z0-9\-\+\(\)]+)") or "H2O"
            body,needs_chem = render_molecule(mol)
        elif "mode=reaction" in p:
            left = _extract(prompt, r"left\s*=\s*([^\n\r,]+)") or "CH3-CH2-OH"
            right= _extract(prompt, r"right\s*=\s*([^\n\r,]+)") or "CH2=CH2"
            above= _extract(prompt, r"above\s*=\s*([^\n\r,]+)") or ""
            below= _extract(prompt, r"below\s*=\s*([^\n\r,]+)") or ""
            body,needs_chem = render_reaction(left,right,above,below)
        else:
            ir=parse_chem_en(prompt); 
            if ir["task"]=="reaction": body,needs_chem = render_reaction(ir["left"],ir["right"],ir.get("above",""),ir.get("below",""))
            elif ir["task"]=="benzene": body,needs_chem = render_benzene(ir["subs"])
            else: body,needs_chem = render_molecule(ir["mol"])
    else:
        body = "% unsupported"

    latex_full = _wrap_document(body, needs_pgf, needs_circ, needs_chem)
    return GenerationResult(latex=latex_full, category=cat, summary="OK")
