"""ScienTikZ diagram library: every diagram is parameters -> TikZ code.

To add a diagram: write a build function and register it with @diagram(...).
tests/test_diagrams.py compiles every diagram automatically.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable

from engine import (Expr, ExprError, auto_yrange, axes, number, num, picture,
                    plot_cmds, samples, segments, ticks, units)

GROUPS = ("数学 Math", "物理 Physics")


@dataclass(frozen=True)
class P:
    key: str
    label: str
    default: object
    kind: str = "float"          # float | int | num (text, may use pi) | expr (formula in x) | bool | choice
    lo: float | None = None
    hi: float | None = None
    step: float | None = None
    options: tuple = ()


@dataclass
class Diagram:
    id: str
    zh: str
    en: str
    group: str
    keywords: str
    params: list[P]
    build: Callable[[dict], str]
    libs: tuple = ()
    free: bool = False           # code + .tex/.pdf/.svg downloadable without a licence

    def defaults(self) -> dict:
        return {p.key: p.default for p in self.params}

    def code(self, values: dict | None = None) -> str:
        v = self.defaults() | (values or {})
        for p in self.params:                      # numeric range guard (UI already enforces it)
            if p.kind in ("float", "int"):
                x = float(v[p.key])
                if (p.lo is not None and x < p.lo) or (p.hi is not None and x > p.hi):
                    raise ExprError(f"{p.label} 超出范围 [{p.lo}, {p.hi}]")
        return self.build(v)

    def document(self, values: dict | None = None) -> str:
        libs = ",".join(("arrows.meta",) + self.libs)
        return ("\\documentclass[tikz,border=4mm]{standalone}\n\\usepackage{amsmath}\n"
                f"\\usetikzlibrary{{{libs}}}\n\\begin{{document}}\n{self.code(values)}\n\\end{{document}}\n")


DIAGRAMS: dict[str, Diagram] = {}


def diagram(id, zh, en, group, keywords, params, libs=(), free=False):
    def reg(fn):
        DIAGRAMS[id] = Diagram(id, zh, en, group, keywords, params, fn, libs, free)
        return fn
    return reg


def _range(a, b, what="x"):
    a, b = number(a, f"{what} min"), number(b, f"{what} max")
    if b <= a:
        raise ExprError(f"{what} 范围需要 min < max / min must be less than max")
    return a, b


def label_box(x, y, text, anchor="north east", color="black"):
    return (f"\\node[anchor={anchor}, {color}, font=\\small, fill=white, fill opacity=0.85, "
            f"text opacity=1, inner sep=2pt] at ({num(x)},{num(y)}) {{{text}}};")


# ================================================================ Math
@diagram("function", "函数图像", "Function graph", GROUPS[0],
         "函数 图像 曲线 抛物线 二次函数 指数 对数 幂函数 function graph plot curve parabola",
         [P("f", "f(x) =", "sin(x)", "expr"), P("xmin", "x 最小 / min", "-2pi", "num"),
          P("xmax", "x 最大 / max", "2pi", "num"), P("grid", "网格 / Grid", True, "bool")], free=True)
def build_function(v):
    f = Expr(v["f"])
    a, b = _range(v["xmin"], v["xmax"])
    pts = samples(f, a, b)
    y0, y1 = auto_yrange(pts)
    m = 0.04 * (b - a)
    xr, yr = (a - m, b + m), (y0, y1)
    xt, yt = ticks(*xr, pi_mode=f.uses_trig), ticks(*yr)
    body = axes(xr, yr, xt, yt, grid=v["grid"])
    body += [f"\\begin{{scope}}", f"  \\clip ({num(xr[0])},{num(y0)}) rectangle ({num(xr[1])},{num(y1)});"]
    body += ["  " + c for c in plot_cmds(f, segments(pts, y0, y1))]
    body += ["\\end{scope}", label_box(xr[1], y1, f"$y = {f.tex()}$", color="blue")]
    return picture(body, units(xr, yr))


@diagram("integral", "定积分面积", "Area under a curve", GROUPS[0],
         "积分 定积分 面积 微积分 黎曼 integral area under curve calculus riemann",
         [P("f", "f(x) =", "x^2", "expr"), P("a", "下限 a / lower", "0", "num"), P("b", "上限 b / upper", "2", "num")])
def build_integral(v):
    f = Expr(v["f"])
    a, b = _range(v["a"], v["b"], "积分 a, b")
    inner = samples(f, a, b, 1000)
    if any(math.isnan(y) for _, y in inner):
        raise ExprError("f(x) 在 [a, b] 内有无定义的点，无法积分 / f is undefined somewhere in [a, b]")
    h = (b - a) / 1000                                     # Simpson's rule
    area = h / 3 * sum(y * (1 if i in (0, 1000) else 4 if i % 2 else 2) for i, (_, y) in enumerate(inner))
    lo, hi = min(a, 0), max(b, 0)
    xr = (lo - 0.2 * (hi - lo), hi + 0.25 * (hi - lo))
    pts = samples(f, *xr)
    y0, y1 = auto_yrange(pts)
    y0 = min(y0, 0, min(y for _, y in inner)); y1 = max(y1, 0, max(y for _, y in inner))
    pad = 0.12 * (y1 - y0); y0, y1 = (y0 - pad if y0 < 0 else y0), y1 + pad
    trig = ", trig format=rad" if f.uses_trig else ""
    body = axes(xr, (y0, y1), [(t, f"${num(t)}$") for t in (a, b) if abs(t) > 1e-9], ticks(y0, y1))
    body += [f"\\fill[blue!15] ({num(a)},0) -- plot[domain={num(a)}:{num(b)}, samples=120, smooth{trig}] "
             f"(\\x, {{{f.pgf()}}}) -- ({num(b)},0) -- cycle;",
             "\\begin{scope}", f"  \\clip ({num(xr[0])},{num(y0)}) rectangle ({num(xr[1])},{num(y1)});"]
    body += ["  " + c for c in plot_cmds(f, segments(pts, y0, y1))]
    body += ["\\end{scope}",
             label_box((a + b) / 2, y1, f"$\\displaystyle\\int_{{{num(a)}}}^{{{num(b)}}} {f.tex()}\\,"
                                        f"\\mathrm{{d}}x \\approx {num(area, 4)}$", anchor="north")]
    return picture(body, units(xr, (y0, y1)))


@diagram("tangent", "切线与导数", "Tangent line", GROUPS[0],
         "切线 导数 斜率 微分 tangent line derivative slope differentiation",
         [P("f", "f(x) =", "0.5x^2", "expr"), P("x0", "切点 x₀ / point", "1", "num"),
          P("xmin", "x 最小 / min", "-3", "num"), P("xmax", "x 最大 / max", "3", "num")])
def build_tangent(v):
    f = Expr(v["f"])
    a, b = _range(v["xmin"], v["xmax"])
    x0 = number(v["x0"], "x0")
    if not a < x0 < b:
        raise ExprError("切点 x₀ 需在 x 范围内 / x0 must lie inside the x range")
    y0f, hh = f(x0), 1e-5
    m = (f(x0 + hh) - f(x0 - hh)) / (2 * hh)
    if math.isnan(y0f) or math.isnan(m):
        raise ExprError("该点处函数不可导 / f is not differentiable at x0")
    c = y0f - m * x0
    pts = samples(f, a, b)
    ylo, yhi = auto_yrange(pts)
    ylo, yhi = min(ylo, y0f - 0.3 * (yhi - ylo)), max(yhi, y0f + 0.3 * (yhi - ylo))
    mg = 0.04 * (b - a); xr = (a - mg, b + mg)
    d = 0.3 * (b - a)
    mx = {"1": "", "-1": "-", "0": ""}.get(num(m, 3), num(m, 3)) + ("x" if num(m, 3) != "0" else "")
    cs = num(c, 3) if not mx else (f" - {num(-c, 3)}" if c < 0 else f" + {num(c, 3)}") if abs(c) > 5e-4 else ""
    eq = f"y = {mx}{cs}" if mx or cs else "y = 0"
    body = axes(xr, (ylo, yhi), ticks(*xr, pi_mode=f.uses_trig), ticks(ylo, yhi))
    body += ["\\begin{scope}", f"  \\clip ({num(xr[0])},{num(ylo)}) rectangle ({num(xr[1])},{num(yhi)});"]
    body += ["  " + c_ for c_ in plot_cmds(f, segments(pts, ylo, yhi))]
    body += [f"  \\draw[red, thick] ({num(x0 - d)},{num(y0f - m * d)}) -- ({num(x0 + d)},{num(y0f + m * d)});",
             "\\end{scope}",
             f"\\draw[dashed, gray] ({num(x0)},0) -- ({num(x0)},{num(y0f)});",
             f"\\fill[red] ({num(x0)},{num(y0f)}) circle (1.8pt) node[above left, font=\\small] "
             f"{{$({num(x0, 3)},\\ {num(y0f, 3)})$}};",
             label_box(xr[1], yhi, f"$y = {f.tex()}$", color="blue"),
             label_box(xr[1], yhi - 0.1 * (yhi - ylo), f"tangent: ${eq}$", color="red")]
    return picture(body, units(xr, (ylo, yhi)))


@diagram("unit_circle", "单位圆与三角函数", "Unit circle", GROUPS[0],
         "单位圆 三角函数 正弦 余弦 角度 弧度 unit circle trigonometry sine cosine angle",
         [P("theta", "角度 θ (°)", 40.0, "float", 0, 360, 1.0)], free=True)
def build_unit_circle(v):
    t = float(v["theta"]); c, s = math.cos(math.radians(t)), math.sin(math.radians(t))
    body = ["\\draw[->] (-1.35,0) -- (1.45,0) node[right] {$x$};",
            "\\draw[->] (0,-1.35) -- (0,1.45) node[above] {$y$};",
            "\\draw[thick] (0,0) circle (1);",
            "\\foreach \\p/\\l in {1/below right, -1/below left} \\node[\\l, font=\\small] at (\\p,0) {$\\p$};",
            "\\foreach \\p/\\l in {1/above left, -1/below left} \\node[\\l, font=\\small] at (0,\\p) {$\\p$};",
            f"\\coordinate (P) at ({num(t)}:1);",
            "\\draw[thick] (0,0) -- (P);",
            f"\\draw[red, very thick] (0,0) -- ({num(c)},0) node[midway, {'below' if s >= 0 else 'above'}] {{$\\cos\\theta$}};",
            f"\\draw[blue, very thick] ({num(c)},0) -- (P) node[midway, {'right' if c >= 0 else 'left'}] {{$\\sin\\theta$}};",
            f"\\draw (0.28,0) arc[start angle=0, end angle={num(t)}, radius=0.28];",
            f"\\node[font=\\small] at ({num(t / 2)}:0.42) {{$\\theta$}};",
            f"\\node[circle, fill, inner sep=1.3pt, label={{[font=\\small]{num(t)}:{{$P({num(c, 3)},\\ {num(s, 3)})$}}}}] at (P) {{}};",
            f"\\node[anchor=north west, font=\\small] at (-1.35,1.45) {{$\\theta = {num(t)}^\\circ$}};"]
    return picture(body, "scale=2.6")


@diagram("vectors", "向量加法", "Vector addition", GROUPS[0],
         "向量 矢量 加法 合成 平行四边形 三角形法则 vector addition resultant parallelogram",
         [P("ux", "u 的 x 分量", 3.0, "float", -10, 10, 0.5), P("uy", "u 的 y 分量", 1.0, "float", -10, 10, 0.5),
          P("vx", "v 的 x 分量", 1.0, "float", -10, 10, 0.5), P("vy", "v 的 y 分量", 3.0, "float", -10, 10, 0.5),
          P("parallelogram", "平行四边形 / Parallelogram", True, "bool")], free=True)
def build_vectors(v):
    ux, uy, vx, vy = (float(v[k]) for k in ("ux", "uy", "vx", "vy"))
    sx, sy = ux + vx, uy + vy
    xs, ys = [0, ux, vx, sx], [0, uy, vy, sy]
    xr = (min(xs) - 1, max(xs) + 1); yr = (min(ys) - 1, max(ys) + 1)
    body = axes(xr, yr, ticks(*xr), ticks(*yr), grid=True)
    if v["parallelogram"]:
        body += [f"\\draw[dashed, gray] ({num(ux)},{num(uy)}) -- ({num(sx)},{num(sy)}) -- ({num(vx)},{num(vy)});"]
    body += [f"\\draw[->, very thick, blue] (0,0) -- ({num(ux)},{num(uy)}) node[midway, sloped, above] {{$\\vec{{u}}$}};",
             f"\\draw[->, very thick, red] (0,0) -- ({num(vx)},{num(vy)}) node[midway, sloped, above] {{$\\vec{{v}}$}};",
             f"\\draw[->, very thick, violet] (0,0) -- ({num(sx)},{num(sy)}) node[pos=0.62, sloped, below] "
             f"{{$\\vec{{u}}+\\vec{{v}} = ({num(sx)},\\ {num(sy)})$}};"]
    return picture(body, units(xr, yr, 8, 6, equal=True))


@diagram("roots_of_unity", "复数单位根", "Roots of unity", GROUPS[0],
         "单位根 复数 复平面 棣莫弗 极坐标 roots of unity complex plane de moivre polar",
         [P("n", "n（zⁿ = 1）", 6, "int", 2, 24, 1), P("polygon", "连成正多边形 / Polygon", True, "bool")])
def build_roots(v):
    n = int(v["n"])
    body = ["\\draw[->] (-1.4,0) -- (1.5,0) node[right] {$\\mathrm{Re}$};",
            "\\draw[->] (0,-1.4) -- (0,1.5) node[above] {$\\mathrm{Im}$};",
            "\\draw[dashed, gray] (0,0) circle (1);"]
    if v["polygon"]:
        body.append(f"\\draw[orange!60] (0:1) \\foreach \\k in {{1,...,{n - 1}}} {{ -- ({{360*\\k/{n}}}:1) }} -- cycle;")
    body += [f"\\foreach \\k in {{0,...,{n - 1}}} {{",
             f"  \\draw[->, orange, thick] (0,0) -- ({{360*\\k/{n}}}:1);",
             f"  \\fill[orange] ({{360*\\k/{n}}}:1) circle (0.9pt);"]
    if n <= 12:
        body.append(f"  \\node[font=\\small] at ({{360*\\k/{n}}}:1.2) {{$\\omega^{{\\k}}$}};")
    body += ["}", f"\\node[anchor=north west, font=\\small] at (-1.4,1.5) "
                  f"{{$z^{{{n}}} = 1,\\quad \\omega = e^{{2\\pi i/{n}}}$}};"]
    return picture(body, "scale=2.4")


@diagram("normal", "正态分布", "Normal distribution", GROUPS[0],
         "正态分布 高斯 概率 统计 钟形曲线 normal distribution gaussian bell curve probability statistics",
         [P("mu", "均值 μ", "0", "num"), P("sigma", "标准差 σ", "1", "num"),
          P("a", "区间 a", "-1", "num"), P("b", "区间 b", "1", "num")])
def build_normal(v):
    mu, sd = number(v["mu"], "μ"), number(v["sigma"], "σ")
    if not 1e-3 <= sd <= 1000 or abs(mu) > 5000:
        raise ExprError("需 0.001 ≤ σ ≤ 1000，|μ| ≤ 5000")
    a, b = _range(v["a"], v["b"], "区间")
    lo, hi = mu - 4 * sd, mu + 4 * sd
    a2, b2 = max(a, lo), min(b, hi)
    peak = 1 / (sd * math.sqrt(2 * math.pi))
    prob = 0.5 * (math.erf((b - mu) / (sd * math.sqrt(2))) - math.erf((a - mu) / (sd * math.sqrt(2))))
    z = f"exp(-((\\x - {num(mu)})/{num(sd)})^2/2)"         # height in units of the peak density
    xt = [(mu + k * sd, f"${num(mu + k * sd, 3)}$") for k in range(-3, 4) if abs(mu + k * sd) > 1e-9]
    yt = [(fr, f"${num(fr * peak, 3)}$") for fr in (0.5, 1.0)]
    body = [f"% y is measured in units of the peak density 1/(sigma*sqrt(2*pi)) = {num(peak, 5)}"]
    body += axes((lo - 0.3 * sd, hi + 0.5 * sd), (0, 1.25), xt, yt, ylabel="f(x)")
    if b2 > a2:
        body.append(f"\\fill[blue!20] ({num(a2)},0) -- plot[domain={num(a2)}:{num(b2)}, samples=80, smooth] "
                    f"(\\x, {{{z}}}) -- ({num(b2)},0) -- cycle;")
    body += [f"\\draw[blue, thick, domain={num(lo)}:{num(hi)}, samples=160, smooth] plot (\\x, {{{z}}});",
             f"\\draw[dashed, gray] ({num(mu)},0) -- ({num(mu)},1);",
             label_box(hi + 0.5 * sd, 1.25, f"$P({num(a, 3)} < X < {num(b, 3)}) \\approx {num(prob, 4)}$"),
             label_box(lo - 0.3 * sd, 1.25, f"$X \\sim N({num(mu, 3)},\\ {num(sd, 3)}^2)$", anchor="north west")]
    return picture(body, units((lo - 0.3 * sd, hi + 0.5 * sd), (0, 1.25), 9, 5))


# ================================================================ Physics
@diagram("incline", "斜面受力分析", "Inclined plane (free-body)", GROUPS[1],
         "斜面 受力分析 摩擦力 支持力 重力 牛顿 inclined plane slope free body diagram force friction normal",
         [P("theta", "倾角 θ (°)", 30.0, "float", 5, 60, 1.0), P("friction", "静摩擦力 / Static friction", True, "bool"),
          P("components", "分解重力 / Resolve mg", False, "bool")], free=True)
def build_incline(v):
    t = float(v["theta"]); r = math.radians(t)
    S, s, w, h, F = 6.0, 3.6, 1.3, 0.85, 1.9          # slope length, block position/size, |mg| arrow
    body = [f"\\draw[thick] (0,0) -- ({num(S * math.cos(r))},0) -- ({num(S * math.cos(r))},{num(S * math.sin(r))}) -- cycle;",
            f"\\draw (1,0) arc[start angle=0, end angle={num(t)}, radius=1];",
            f"\\node[font=\\small] at ({num(t / 2)}:1.35) {{$\\theta$}};",
            f"\\begin{{scope}}[rotate={num(t)}]",
            f"  \\draw[thick, fill=blue!10] ({num(s - w / 2)},0) rectangle ({num(s + w / 2)},{num(h)});",
            f"  \\coordinate (C) at ({num(s)},{num(h / 2)});",
            "\\end{scope}",
            f"\\draw[->, very thick, red] (C) -- ++(0,{num(-F)}) node[right] {{$m\\vec{{g}}$}};",
            f"\\draw[->, very thick, green!50!black] (C) -- ++({num(t + 90)}:{num(F * math.cos(r))}) node[above] {{$\\vec{{N}}$}};"]
    if v["friction"]:     # at rest: friction balances mg sin(theta) and points up the slope
        body.append(f"\\draw[->, very thick, orange] (C) -- ++({num(t)}:{num(F * math.sin(r))}) node[above right] {{$\\vec{{f}}$}};")
    if v["components"]:
        body += [f"\\draw[->, dashed, red!70] (C) -- ++({num(t + 180)}:{num(F * math.sin(r))}) node[above left, font=\\small] {{$mg\\sin\\theta$}};",
                 f"\\draw[->, dashed, red!70] (C) -- ++({num(t - 90)}:{num(F * math.cos(r))}) node[right, font=\\small] {{$mg\\cos\\theta$}};"]
    body.append(f"\\node[anchor=north west, font=\\small] at (0,{num(S * math.sin(r) + 0.6)}) {{$\\theta = {num(t)}^\\circ$}};")
    return picture(body)


@diagram("projectile", "抛体运动", "Projectile motion", GROUPS[1],
         "抛体 斜抛 射程 轨迹 运动学 projectile motion trajectory range kinematics",
         [P("v0", "初速度 v₀ (m/s)", 20.0, "float", 1, 100, 0.5), P("theta", "抛射角 θ (°)", 45.0, "float", 5, 85, 1.0),
          P("g", "重力加速度 g (m/s²)", 9.8, "float", 1, 30, 0.1)])
def build_projectile(v):
    v0, t, g = float(v["v0"]), float(v["theta"]), float(v["g"])
    r = math.radians(t)
    vx, vy = v0 * math.cos(r), v0 * math.sin(r)
    T, R, H = 2 * vy / g, v0 ** 2 * math.sin(2 * r) / g, vy ** 2 / (2 * g)
    xr, yr = (-0.08 * R, 1.32 * R), (-0.12 * max(H, R / 4), 1.3 * H)
    body = axes(xr, yr, ticks(*xr), ticks(0, yr[1]), xlabel="x\\,(\\mathrm{m})", ylabel="y\\,(\\mathrm{m})")
    body += [f"\\draw[blue, thick, variable=\\t, domain=0:{num(T)}, samples=100, smooth] "
             f"plot ({{{num(vx)}*\\t}}, {{{num(vy)}*\\t - {num(g / 2)}*\\t^2}});",
             f"\\draw[->, very thick, red] (0,0) -- ++({num(t)}:1.4cm) node[above, font=\\small] {{$\\vec{{v}}_0$}};",
             f"\\draw (0.7cm,0) arc[start angle=0, end angle={num(t)}, radius=0.7cm];",
             f"\\node[font=\\small] at ({num(t / 2)}:0.95cm) {{$\\theta$}};",
             f"\\draw[dashed, gray] ({num(R / 2)},0) -- ({num(R / 2)},{num(H)});",
             f"\\node[right, font=\\small] at ({num(R / 2)},{num(H / 2)}) {{$H = {num(H, 2)}\\,\\mathrm{{m}}$}};",
             f"\\fill ({num(R)},0) circle (1.5pt) node[above right, font=\\small] {{$R = {num(R, 2)}\\,\\mathrm{{m}}$}};",
             label_box(xr[1], yr[1], f"$v_0 = {num(v0)}\\,\\mathrm{{m/s}},\\ \\theta = {num(t)}^\\circ,\\ "
                                     f"T = {num(T, 2)}\\,\\mathrm{{s}}$")]
    return picture(body, units(xr, yr, 10, 6, equal=True))


@diagram("wave", "简谐波", "Sinusoidal wave", GROUPS[1],
         "波 横波 简谐波 波长 振幅 波速 wave wavelength amplitude transverse sinusoidal",
         [P("A", "振幅 A", 1.0, "float", 0.1, 10, 0.1), P("lam", "波长 λ", 4.0, "float", 0.5, 50, 0.5),
          P("direction", "传播方向 / Direction", "+x", "choice", options=("+x", "-x"))])
def build_wave(v):
    A, L = float(v["A"]), float(v["lam"])
    xr, yr = (-0.1 * L, 2.25 * L), (-1.5 * A, 1.6 * A)
    xt = [(k * L / 2, f"${num(k * L / 2, 3)}$") for k in range(1, 5)]
    sgn = 1 if v["direction"] == "+x" else -1
    body = axes(xr, yr, xt, [(A, f"${num(A, 3)}$"), (-A, f"${num(-A, 3)}$")])
    body += [f"\\draw[blue, thick, domain=0:{num(2 * L)}, samples=200, smooth, trig format=rad] "
             f"plot (\\x, {{{num(A)}*sin(2*pi*\\x/{num(L)})}});",
             f"\\draw[<->] ({num(L / 4)},0) -- ({num(L / 4)},{num(A)}) node[midway, right] {{$A$}};",
             f"\\draw[dashed, gray] ({num(L / 4)},{num(A)}) -- ({num(L / 4)},{num(1.3 * A)}) "
             f"({num(5 * L / 4)},{num(A)}) -- ({num(5 * L / 4)},{num(1.3 * A)});",
             f"\\draw[<->] ({num(L / 4)},{num(1.25 * A)}) -- ({num(5 * L / 4)},{num(1.25 * A)}) node[midway, fill=white] {{$\\lambda$}};",
             f"\\draw[->, very thick, red] ({num(1.75 * L if sgn > 0 else 2 * L)},{num(1.25 * A)}) -- ++({sgn * 1}cm,0) "
             f"node[{'right' if sgn > 0 else 'left'}] {{$\\vec{{v}}$}};"]
    return picture(body, units(xr, yr, 10, 4.5))


@diagram("capacitor", "平行板电容器电场", "Parallel-plate capacitor", GROUPS[1],
         "电容器 平行板 电场线 电场 匀强电场 静电 capacitor parallel plate electric field lines electrostatics",
         [P("lines", "电场线条数 / Field lines", 7, "int", 3, 15, 1), P("d", "板间距 d (cm)", 3.0, "float", 1, 6, 0.5),
          P("length", "板长 (cm)", 4.0, "float", 2, 8, 0.5)], libs=("decorations.markings",))
def build_capacitor(v):
    n, d, L = int(v["lines"]), float(v["d"]), float(v["length"])
    ys = ", ".join(num(-L / 2 + L * (i + 0.5) / n) for i in range(n))
    body = ["\\tikzset{mid arrow/.style={postaction={decorate}, decoration={markings, "
            "mark=at position 0.55 with {\\arrow{Stealth}}}}}",
            f"\\foreach \\y in {{{ys}}} \\draw[blue, mid arrow] (0.08,\\y) -- ({num(d - 0.08)},\\y);",
            f"\\draw[line width=2.2pt, red] (0,{num(-L / 2)}) -- (0,{num(L / 2)}) node[above, black] {{$+Q$}};",
            f"\\draw[line width=2.2pt, blue!60!black] ({num(d)},{num(-L / 2)}) -- ({num(d)},{num(L / 2)}) node[above, black] {{$-Q$}};",
            f"\\foreach \\y in {{{ys}}} {{ \\node[red, font=\\small] at (-0.25,\\y) {{$+$}}; "
            f"\\node[blue!60!black, font=\\small] at ({num(d + 0.25)},\\y) {{$-$}}; }}",
            f"\\draw[<->] (0,{num(-L / 2 - 0.45)}) -- ({num(d)},{num(-L / 2 - 0.45)}) node[midway, below] {{$d$}};",
            f"\\node[fill=white, inner sep=1.5pt] at ({num(d / 2)},{num(L / 2 + 0.35)}) {{$\\vec{{E}}$}};",
            f"\\node[anchor=west, font=\\small] at ({num(d + 0.7)},0) {{$E = \\dfrac{{V}}{{d}}$}};"]
    return picture(body)


@diagram("infinite_well", "一维无限深势阱", "Infinite square well", GROUPS[1],
         "无限深势阱 量子力学 波函数 能级 薛定谔 infinite square well particle in a box quantum wavefunction energy levels",
         [P("levels", "能级数 / Levels", 3, "int", 1, 5, 1),
          P("mode", "绘制 / Plot", "ψ", "choice", options=("ψ", "|ψ|²")),
          P("to_scale", "能级按比例 / Levels to scale", False, "bool")], libs=("patterns",))
def build_well(v):
    N, W, top = int(v["levels"]), 6.0, 6.0
    sq = v["mode"] != "ψ"
    if v["to_scale"]:                                   # E_n = n^2 E_1
        E = [top * n * n / (N * N + 0.6) for n in range(N + 1)]
    else:                                               # evenly spaced for readability
        E = [top * n / (N + 0.6) for n in range(N + 1)]
    ceiling = top + 0.5
    body = [f"\\fill[pattern=north east lines] (-0.35,0) rectangle (0,{num(ceiling)});",
            f"\\fill[pattern=north east lines] ({num(W)},0) rectangle ({num(W + 0.35)},{num(ceiling)});",
            f"\\draw[thick] (0,{num(ceiling)}) -- (0,0) -- ({num(W)},0) -- ({num(W)},{num(ceiling)});",
            "\\node[below] at (0,0) {$0$};", f"\\node[below] at ({num(W)},0) {{$L$}};",
            f"\\node[above] at (-0.18,{num(ceiling)}) {{$V=\\infty$}};",
            f"\\node[above] at ({num(W + 0.18)},{num(ceiling)}) {{$V=\\infty$}};"]
    for n in range(1, N + 1):
        below = E[n] - E[n - 1]
        above = (E[n + 1] if n < N else ceiling) - E[n]
        amp = 0.8 * above if sq else 0.42 * min(below, above)
        f = f"{num(E[n])} + {num(amp)}*sin({n}*pi*\\x/{num(W)})" + ("^2" if sq else "")
        psi = f"|\\psi_{{{n}}}|^2" if sq else f"\\psi_{{{n}}}"
        energy = f"E_{{{n}}} = {n * n}E_1" if n > 1 else "E_1"
        body += [f"\\draw[gray, dashed] (0,{num(E[n])}) -- ({num(W)},{num(E[n])}) "
                 f"node[right, black, font=\\small, xshift=0.35cm] {{${psi}:\\ {energy}$}};",
                 f"\\draw[blue, thick, domain=0:{num(W)}, samples=120, smooth, trig format=rad] plot (\\x, {{{f}}});"]
    if not v["to_scale"]:
        body.append(f"\\node[anchor=north, font=\\footnotesize, gray] at ({num(W / 2)},-0.35) {{(energy levels not to scale)}};")
    return picture(body)


@diagram("kepler_orbit", "开普勒椭圆轨道", "Kepler orbit", GROUPS[1],
         "开普勒 轨道 椭圆 行星 天文 焦点 近日点 远日点 偏心率 kepler orbit ellipse planet astronomy perihelion aphelion eccentricity",
         [P("e", "偏心率 e", 0.5, "float", 0, 0.9, 0.05), P("nu", "行星位置角 ν (°)", 60.0, "float", 0, 360, 5.0)],
         libs=("shapes.misc",))
def build_kepler(v):
    e, nu = float(v["e"]), math.radians(float(v["nu"]))
    a = 3.0; b = a * math.sqrt(1 - e * e); c = a * e
    r = a * (1 - e * e) / (1 + e * math.cos(nu))
    px, py = c + r * math.cos(nu), r * math.sin(nu)
    body = [f"\\draw[thick] (0,0) ellipse[x radius={num(a)}, y radius={num(b)}];",
            f"\\draw[dashed, gray] ({num(-a)},0) -- ({num(a)},0);",
            f"\\fill[orange] ({num(c)},0) circle (5pt);",
            f"\\node[below, font=\\small] at ({num(c)},-0.15) {{Sun}};"]
    if e > 0:
        body.append(f"\\draw ({num(-c)},0) node[cross out, draw, inner sep=1.5pt] {{}};")
    body += [f"\\draw[->, thick, red] ({num(c)},0) -- ({num(px)},{num(py)}) node[pos=0.5, sloped, above] {{$r$}};",
             f"\\fill[blue] ({num(px)},{num(py)}) circle (3pt);",
             f"\\fill ({num(a)},0) circle (1.3pt) node[right, font=\\small, align=left] {{perihelion\\\\$r_p = a(1-e)$}};",
             f"\\fill ({num(-a)},0) circle (1.3pt) node[left, font=\\small, align=right] {{aphelion\\\\$r_a = a(1+e)$}};",
             f"\\draw[<->] ({num(-a)},{num(-b - 0.45)}) -- ({num(a)},{num(-b - 0.45)}) node[midway, below] {{$2a$}};",
             f"\\node[anchor=south, font=\\small] at (0,{num(b + 0.2)}) "
             f"{{$e = {num(e, 2)},\\quad r = \\dfrac{{a(1-e^2)}}{{1+e\\cos\\nu}}$}};"]
    return picture(body)


# ================================================================ search
def find(query: str):
    """Pick a diagram and pre-filled values from free text: '斜面 30度', 'y = x^2', 'n=8 单位根'."""
    q = query.strip().lower()
    if not q:
        return None, {}
    best, score = None, 0
    for d in DIAGRAMS.values():
        words = (d.keywords + " " + d.zh + " " + d.en).lower().split()
        sc = sum(len(w) for w in set(words) if len(w) > 1 and w in q)
        if sc > score:
            best, score = d, sc
    values = {}
    if best is None:                                   # looks like a formula? -> function graph
        try:
            Expr(query)
            return DIAGRAMS["function"], {"f": query.strip()}
        except ExprError:
            return None, {}
    keys = {p.key.lower(): p for p in best.params}
    for k, val in re.findall(r"([a-zμσθν]+\d?)\s*=\s*(-?\d+(?:\.\d+)?)", q):
        k = {"μ": "mu", "σ": "sigma", "θ": "theta", "ν": "nu", "λ": "lam"}.get(k, k)
        if k in keys:
            values[keys[k].key] = _cast(keys[k], val)
    deg = re.search(r"(-?\d+(?:\.\d+)?)\s*(°|度|deg)", q)
    if deg and "theta" in keys:
        values.setdefault("theta", _cast(keys["theta"], deg.group(1)))
    if "f" in keys:                                    # e.g. '积分 x^3 from 0 to 1' / 'x^3 的切线'
        m = re.search(r"((?:y\s*=|f\(x\)\s*=)?[\sx0-9+\-*/^().a-z]*x[\sx0-9+\-*/^().a-z]*)", q)
        if m:
            cand = re.sub(r"\b(from|to|area|under|integral|tangent|at|of|the|graph|plot)\b", " ", m.group(1)).strip()
            try:
                Expr(cand); values["f"] = cand
            except ExprError:
                pass
    return best, values


def _cast(p: P, s: str):
    x = float(s)
    if p.kind == "int":
        x = int(x)
    if p.kind == "num":
        return s
    if p.lo is not None:
        x = max(p.lo, min(p.hi, x))
    return x
