"""ScienTikZ core: safe expressions + axis/plot builders.

User formulas never reach TeX as raw text. They are parsed into a Python AST,
checked against a whitelist, then re-emitted as pgfmath. The same tree is
evaluated in Python to pick ranges, split at singularities and compute values.
"""
from __future__ import annotations

import ast
import math
import re

# name -> (python function, pgfmath name)
FUNCS = {
    "sin": (math.sin, "sin"), "cos": (math.cos, "cos"), "tan": (math.tan, "tan"),
    "exp": (math.exp, "exp"), "ln": (math.log, "ln"), "log": (math.log, "ln"),
    "lg": (math.log10, "log10"), "sqrt": (math.sqrt, "sqrt"), "abs": (abs, "abs"),
}
CONSTS = {"pi": (math.pi, "pi"), "e": (math.e, "e")}
TRIG = {"sin", "cos", "tan"}
PGF_MAX = 16000.0  # pgfmath overflows around 16383.99

_BIN = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Pow: "^"}
_PREC = {ast.Add: 1, ast.Sub: 1, ast.Mult: 2, ast.Div: 2, ast.Pow: 4}


class ExprError(ValueError):
    """Formula the user typed cannot be used (message is shown to the user)."""


def _normalise(s: str) -> str:
    s = s.strip().lower()
    for a, b in {"×": "*", "·": "*", "÷": "/", "−": "-", "–": "-", "²": "^2", "³": "^3",
                 "π": "pi", "（": "(", "）": ")", "，": ","}.items():
        s = s.replace(a, b)
    s = re.sub(r"^\s*(y|f\(x\))\s*=", "", s)
    s = re.sub(r"(\d)\s*([a-z(])", r"\1*\2", s)          # 2x, 3(x+1), 2pi
    s = re.sub(r"\)\s*([a-z0-9(])", r")*\1", s)          # (x+1)(x-1), (x)2
    s = re.sub(r"\b(x|pi|e)\b\s*(?=[a-z(])", r"\1*", s)  # x sin(x), pi x, x(x+1)
    return s.replace("^", "**").strip()


class Expr:
    """A whitelisted formula in x. `Expr('2x^2 - sin(x)')`."""

    def __init__(self, text: str, allow_x: bool = True):
        if not text or not text.strip():
            raise ExprError("请输入表达式 / Expression is empty")
        if len(text) > 120:
            raise ExprError("表达式过长（最多 120 字符）/ Expression too long")
        self.text = text.strip()
        try:
            self.tree = ast.parse(_normalise(text), mode="eval").body
        except SyntaxError:
            raise ExprError(f"无法识别：{text}（函数请带括号，如 sin(x)）/ Cannot parse")
        self.allow_x = allow_x
        self.uses_trig = False
        self._check(self.tree, 0)

    def _check(self, n, depth):
        if depth > 30:
            raise ExprError("表达式嵌套过深 / Too deeply nested")
        if isinstance(n, ast.BinOp) and type(n.op) in _BIN:
            self._check(n.left, depth + 1); self._check(n.right, depth + 1)
        elif isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            self._check(n.operand, depth + 1)
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in FUNCS:
            if len(n.args) != 1 or n.keywords:
                raise ExprError(f"{n.func.id}() 只接受一个参数 / takes one argument")
            self.uses_trig |= n.func.id in TRIG
            self._check(n.args[0], depth + 1)
        elif isinstance(n, ast.Name) and (n.id in CONSTS or (n.id == "x" and self.allow_x)):
            pass
        elif isinstance(n, ast.Constant) and type(n.value) in (int, float):
            if abs(n.value) > PGF_MAX:
                raise ExprError("数值过大 / Number too large")
        else:
            bad = getattr(n, "id", None) or type(n).__name__
            raise ExprError(f"不支持：{bad}。可用：x, pi, e, + - * / ^, "
                            + ", ".join(sorted(FUNCS)) + " / Not supported")

    # ---------- evaluation (also enforces pgfmath's numeric range) ----------
    def __call__(self, x: float = 0.0) -> float:
        try:
            v = self._ev(self.tree, x)
        except (ValueError, ZeroDivisionError, OverflowError, TypeError):
            return math.nan
        return v if math.isfinite(v) else math.nan

    def _ev(self, n, x):
        if isinstance(n, ast.Constant):
            return float(n.value)
        if isinstance(n, ast.Name):
            return x if n.id == "x" else CONSTS[n.id][0]
        if isinstance(n, ast.UnaryOp):
            v = self._ev(n.operand, x)
            return -v if isinstance(n.op, ast.USub) else v
        if isinstance(n, ast.Call):
            v = FUNCS[n.func.id][0](self._ev(n.args[0], x))
        else:
            a, b = self._ev(n.left, x), self._ev(n.right, x)
            op = type(n.op)
            if op is ast.Pow:
                if a < 0 and b != int(b):
                    raise ValueError
                v = a ** b
            else:
                v = {ast.Add: a + b, ast.Sub: a - b, ast.Mult: a * b}.get(op) if op is not ast.Div else a / b
        if isinstance(v, complex) or abs(v) > PGF_MAX:   # pgfmath would overflow here
            raise OverflowError
        return v

    def value(self) -> float:
        """Numeric value of a constant expression such as '-2pi'."""
        v = self(0.0)
        if math.isnan(v):
            raise ExprError(f"不是有效数值：{self.text} / Not a valid number")
        return v

    # ---------- pgfmath emission ----------
    def pgf(self) -> str:
        return self._pgf(self.tree)[0]

    def _pgf(self, n):
        if isinstance(n, ast.Constant):
            return num(n.value), 5
        if isinstance(n, ast.Name):
            return ("\\x" if n.id == "x" else CONSTS[n.id][1]), 5
        if isinstance(n, ast.Call):
            return f"{FUNCS[n.func.id][1]}({self._pgf(n.args[0])[0]})", 5
        if isinstance(n, ast.UnaryOp):
            s, p = self._pgf(n.operand)
            if isinstance(n.op, ast.UAdd):
                return s, p
            return f"-{s if p == 5 else '(' + s + ')'}", 3
        op = type(n.op)
        p = _PREC[op]
        (ls, lp), (rs, rp) = self._pgf(n.left), self._pgf(n.right)
        # pgf substitutes \x textually: \x^2 at x=-1.5 becomes -1.5^2 = -2.25, so bracket power bases
        if lp < p or (op is ast.Pow and (lp < 5 or ls == "\\x")):
            ls = f"({ls})"
        if rp < p or (rp == p and op in (ast.Sub, ast.Div, ast.Pow)) or rp == 3 or (op is ast.Pow and (rp < 5 or rs == "\\x")):
            rs = f"({rs})"
        sep = " " if p == 1 else ""
        return f"{ls}{sep}{_BIN[op]}{sep}{rs}", p

    def tex(self) -> str:
        """Readable LaTeX for labels, e.g. x^{2}, \\sin(x)."""
        return self._tex(self.tree)[0]

    def _tex(self, n):
        if isinstance(n, ast.Constant):
            return num(n.value), 5
        if isinstance(n, ast.Name):
            return {"x": "x", "pi": "\\pi", "e": "e"}[n.id], 5
        if isinstance(n, ast.Call):
            f, inner = n.func.id, self._tex(n.args[0])[0]
            if f == "sqrt":
                return f"\\sqrt{{{inner}}}", 5
            if f == "abs":
                return f"\\left|{inner}\\right|", 5
            name = {"log": "ln", "lg": "lg"}.get(f, f)
            return f"\\{name}\\left({inner}\\right)" if name != "lg" else f"\\lg\\left({inner}\\right)", 5
        if isinstance(n, ast.UnaryOp):
            s, p = self._tex(n.operand)
            return (s, p) if isinstance(n.op, ast.UAdd) else (f"-{s if p >= 3 else '(' + s + ')'}", 3)
        op = type(n.op)
        p = _PREC[op]
        (ls, lp), (rs, rp) = self._tex(n.left), self._tex(n.right)
        if op is ast.Div:
            return f"\\frac{{{ls}}}{{{rs}}}", 5
        if op is ast.Pow:
            return f"{ls if lp == 5 and not isinstance(n.left, ast.Call) else '(' + ls + ')'}^{{{rs}}}", 4
        if lp < p:
            ls = f"({ls})"
        if rp < p or (rp == p and op is ast.Sub) or rp == 3:
            rs = f"({rs})"
        if op is ast.Mult:
            return (f"{ls}\\cdot {rs}" if rs[0].isdigit() or rp == 3 else f"{ls}{rs}"), 2
        return f"{ls} {_BIN[op]} {rs}", 1


def number(text, name="value") -> float:
    """Parse a number field that may contain pi, e.g. '-2pi', '3/2'."""
    if isinstance(text, (int, float)):
        return float(text)
    try:
        return Expr(str(text), allow_x=False).value()
    except ExprError as err:
        raise ExprError(f"{name}: {err}")


def num(v: float, nd: int = 4) -> str:
    """Compact number for TikZ code: 2.0 -> 2, 0.333333 -> 0.3333."""
    s = f"{round(float(v), nd):.{nd}f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


# ---------------------------------------------------------------- ranges
def samples(f: Expr, a: float, b: float, n: int = 1200):
    return [(a + (b - a) * i / n, f(a + (b - a) * i / n)) for i in range(n + 1)]


def auto_yrange(pts, include_zero=True):
    ys = sorted(y for _, y in pts if not math.isnan(y))
    if not ys:
        raise ExprError("函数在该区间内无定义 / Undefined on this interval")
    lo, hi = ys[int(0.02 * (len(ys) - 1))], ys[int(0.98 * (len(ys) - 1))]
    if hi - lo < 1e-9:
        lo, hi = lo - 1, hi + 1
    if include_zero and lo > 0 and lo < (hi - lo):
        lo = 0.0
    if include_zero and hi < 0 and -hi < (hi - lo):
        hi = 0.0
    pad = 0.1 * (hi - lo)
    return lo - pad, hi + pad


def segments(pts, ylo, yhi):
    """Maximal x-intervals where f is defined and stays near the view window."""
    cap = 0.75 * (yhi - ylo)
    out, start, prev = [], None, None
    for x, y in pts:
        ok = not math.isnan(y) and ylo - cap <= y <= yhi + cap
        if ok and prev is not None and abs(y - prev[1]) > 2 * (yhi - ylo):
            ok_jump = False
        else:
            ok_jump = True
        if ok and ok_jump:
            start = x if start is None else start
        else:
            if start is not None and prev is not None and prev[0] > start:
                out.append((start, prev[0]))
            start = x if ok else None
        prev = (x, y) if ok else None
    if start is not None and prev is not None and prev[0] > start:
        out.append((start, prev[0]))
    return out


def nice_step(span: float, target: int = 6) -> float:
    raw = span / target
    k = 10 ** math.floor(math.log10(raw))
    return next(m * k for m in (1, 2, 2.5, 5, 10) if m * k >= raw)


def ticks(lo: float, hi: float, pi_mode: bool = False):
    """[(value, label)] inside (lo, hi), excluding 0."""
    if pi_mode and hi - lo >= math.pi:
        step = math.pi / 2 if hi - lo <= 4 * math.pi else math.pi
        out = []
        for k in range(math.ceil(lo / step), math.floor(hi / step) + 1):
            frac = k * step / math.pi          # multiple of pi, in halves
            if k and lo < k * step < hi:
                out.append((k * step, f"${pi_label(frac)}$"))
        return out
    step = nice_step(hi - lo)
    first = math.ceil(lo / step)
    vals = [i * step for i in range(first, math.floor(hi / step) + 1)]
    return [(v, f"${num(v)}$") for v in vals if abs(v) > step * 1e-6 and lo < v < hi]


def pi_label(m: float) -> str:
    """m * pi as LaTeX, for m a multiple of 1/2."""
    sign, m = ("-" if m < 0 else ""), abs(m)
    if abs(m - round(m)) < 1e-9:
        m = int(round(m))
        return f"{sign}{'' if m == 1 else m}\\pi"
    n = int(round(2 * m))
    return f"{sign}\\frac{{{'' if n == 1 else n}\\pi}}{{2}}"


# ---------------------------------------------------------------- builders
def picture(body: list[str], opts: str = "") -> str:
    head = "\\begin{tikzpicture}[>=Stealth" + (", " + opts if opts else "") + "]"
    return "\n".join([head, *("  " + line for line in body), "\\end{tikzpicture}"])


def axes(xr, yr, xt=None, yt=None, grid=False, xlabel="x", ylabel="y"):
    """Axis lines through the origin (or the window edge), arrows, tick labels."""
    (x0, x1), (y0, y1) = xr, yr
    ay = 0 if y0 <= 0 <= y1 else y0
    ax = 0 if x0 <= 0 <= x1 else x0
    out = []
    if grid:
        out.append(f"\\draw[gray!20, very thin] ({num(x0)},{num(y0)}) rectangle ({num(x1)},{num(y1)});")
        if xt:
            out.append(f"\\foreach \\t in {{{', '.join(num(v) for v, _ in xt)}}} "
                       f"\\draw[gray!20, very thin] (\\t,{num(y0)}) -- (\\t,{num(y1)});")
        if yt:
            out.append(f"\\foreach \\t in {{{', '.join(num(v) for v, _ in yt)}}} "
                       f"\\draw[gray!20, very thin] ({num(x0)},\\t) -- ({num(x1)},\\t);")
    out.append(f"\\draw[->] ({num(x0)},{num(ay)}) -- ({num(x1)},{num(ay)}) node[right] {{${xlabel}$}};")
    out.append(f"\\draw[->] ({num(ax)},{num(y0)}) -- ({num(ax)},{num(y1)}) node[above] {{${ylabel}$}};")
    if xt:
        items = ", ".join(f"{num(v)}/{{{l}}}" for v, l in xt)
        out.append(f"\\foreach \\t/\\l in {{{items}}}")
        out.append(f"  \\draw (\\t,{num(ay)}) -- ++(0,-2pt) node[below, font=\\small] {{\\l}};")
    if yt:
        items = ", ".join(f"{num(v)}/{{{l}}}" for v, l in yt)
        out.append(f"\\foreach \\t/\\l in {{{items}}}")
        out.append(f"  \\draw ({num(ax)},\\t) -- ++(-2pt,0) node[left, font=\\small] {{\\l}};")
    if ax == 0 and ay == 0:
        out.append("\\node[below left, font=\\small] at (0,0) {$O$};")
    return out


def units(xr, yr, width=9.0, height=6.0, equal=False):
    """tikzpicture options so the window fills width x height cm."""
    xu, yu = width / (xr[1] - xr[0]), height / (yr[1] - yr[0])
    if equal:
        xu = yu = min(xu, yu)
    return f"x={num(xu, 3)}cm, y={num(yu, 3)}cm"


def plot_cmds(f: Expr, segs, style="blue, thick"):
    trig = ", trig format=rad" if f.uses_trig else ""
    span = sum(b - a for a, b in segs) or 1
    return [f"\\draw[{style}, domain={num(a)}:{num(b)}, samples={max(40, int(240 * (b - a) / span))}, smooth{trig}] "
            f"plot (\\x, {{{f.pgf()}}});" for a, b in segs]
