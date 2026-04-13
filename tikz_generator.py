# tikz_generator.py
from dataclasses import dataclass
import re
import math
from typing import List, Dict, Any, Optional

@dataclass
class GraphSpec:
    type: str
    category: str
    title: str = ""
    domain: Optional[List[float]] = None
    axes: Dict[str, bool] = None
    plots: List[Dict] = None
    vectors: List[Dict] = None
    labels: List[Dict] = None
    grid: bool = True
    theme: str = "academic"
    extra: Dict[str, Any] = None

    def __post_init__(self):
        if self.axes is None: self.axes = {"x": True, "y": True, "grid": self.grid}
        if self.plots is None: self.plots = []
        if self.vectors is None: self.vectors = []
        if self.labels is None: self.labels = []
        if self.extra is None: self.extra = {}

@dataclass
class GenerationResult:
    latex: str
    summary: str
    category: str
    spec: GraphSpec = None

def _fmt(code: str) -> str:
    return code.strip().replace("\t", "    ")

# ====================== 改进后的 Parser（支持空格、中文） ======================
def parse_to_spec(prompt: str, category: str = "Mathematics") -> GraphSpec:
    p = prompt.strip().lower()
    orig = prompt.strip()

    # ==================== 数学类 ====================
    if re.search(r'y\s*=', p) or "函数" in p or "plot" in p or "sin" in p or "cos" in p:
        m = re.search(r'y\s*=\s*([^;,\n]+)', orig, re.I)
        expr = m.group(1).strip() if m else "x^2"
        domain = (-5, 5)
        m2 = re.search(r'from\s*([-\d.]+)\s*to\s*([-\d.]+)', orig, re.I)
        if m2:
            domain = (float(m2.group(1)), float(m2.group(2)))
        return GraphSpec(
            type="function_plot",
            category="Mathematics · 函数图像",
            domain=list(domain),
            plots=[{"expr": expr, "style": "blue,thick"}],
            grid=True
        )

    elif "vector" in p or "矢量" in p or "v=" in p or "力" in p:
        # 支持 (3,2) 或 v=(3,2)
        m = re.search(r'[v力]\s*[=（(]\s*([-\d.]+)\s*[,，]\s*([-\d.]+)', orig, re.I)
        if m:
            vx, vy = float(m.group(1)), float(m.group(2))
        else:
            vx, vy = 3.0, 2.0
        return GraphSpec(
            type="vector",
            category="Mathematics · 矢量分析",
            vectors=[{"start": (0,0), "end": (vx, vy), "label": r"\vec{v}"}],
            grid=True
        )

    # ==================== 物理类（重点新增） ====================
    elif "平抛" in p or "抛体" in p or "projectile" in p or "free fall" in p:
        return GraphSpec(
            type="projectile_motion",
            category="Physics · 抛体运动",
            extra={"v0": 10, "angle": 45, "g": 9.8},
            grid=True
        )

    elif "受力" in p or "自由体" in p or "free body" in p or "力图" in p:
        return GraphSpec(
            type="force_diagram",
            category="Physics · 受力分析",
            vectors=[
                {"start": (0,0), "end": (0,5), "label": "mg"},
                {"start": (0,0), "end": (4,0), "label": "N"},
                {"start": (0,0), "end": (2,3), "label": "F"}
            ],
            grid=False
        )

    # ==================== 化学 / 统计（占位，后面再完善） ====================
    elif "化学" in p or "轨道" in p or "分子" in p:
        return GraphSpec(type="chemistry", category="Chemistry · 分子轨道")
    elif "统计" in p or "分布" in p or "normal" in p:
        return GraphSpec(type="statistics", category="Statistics · 正态分布")

    # 默认 fallback（数学）
    return GraphSpec(type="function_plot", category="Mathematics · 默认函数", domain=[-5,5], plots=[{"expr":"x^2"}])

# ====================== 改进后的 Renderer ======================
def render_spec(spec: GraphSpec) -> str:
    if spec.type == "function_plot":
        a, b = spec.domain
        expr = spec.plots[0]["expr"].replace("^", "**").replace("arctan", "atan")
        return rf"""
\begin{{tikzpicture}}
  \draw[->] ({a:.1f},0) -- ({b:.1f},0) node[right] {{$x$}};
  \draw[->] (0,-3) -- (0,3) node[above] {{$y$}};
  \draw[very thin,color=gray!30] ({a:.1f},-3) grid ({b:.1f},3);
  \draw[domain={a:.1f}:{b:.1f},smooth,variable=\x,blue,thick] plot (\x,{{{expr}}});
\end{{tikzpicture}}
"""

    elif spec.type == "vector":
        v = spec.vectors[0]
        ex, ey = v["end"]
        L = math.ceil(max(abs(ex), abs(ey), 5))
        return rf"""
\begin{{tikzpicture}}[>=stealth]
  \draw[->] (-{L},0) -- ({L},0) node[right] {{$x$}};
  \draw[->] (0,-{L}) -- (0,{L}) node[above] {{$y$}};
  \draw[very thin,color=gray!30] (-{L},-{L}) grid ({L},{L});
  \draw[->,very thick,blue] (0,0) -- ({ex:.2f},{ey:.2f}) node[above right] {{{v.get("label", r"\vec v")}}};
\end{{tikzpicture}}
"""

    elif spec.type == "projectile_motion":
        return r"""
\begin{tikzpicture}[>=stealth]
  \draw[->] (-1,0) -- (11,0) node[right] {$x$};
  \draw[->] (0,0) -- (0,6) node[above] {$y$};
  \draw[very thin,color=gray!30] (0,0) grid (10,6);
  \draw[domain=0:10,smooth,variable=\x,red,thick] plot (\x,{-0.05*\x^2 + 7});
  \draw[dashed,blue] (0,0) -- (5,7) node[midway,above] {初速度};
  \node[below] at (10,0) {落地};
\end{tikzpicture}
"""

    elif spec.type == "force_diagram":
        return r"""
\begin{tikzpicture}[>=stealth]
  \draw[thick] (0,0) rectangle (4,3);
  \draw[->,very thick,red] (2,1.5) -- (2,4) node[above] {mg};
  \draw[->,very thick,blue] (2,1.5) -- (6,1.5) node[right] {N};
  \draw[->,very thick,green] (2,1.5) -- (4.5,3.5) node[above right] {F};
  \node at (2,1.5) [circle,draw,fill=white] {物体};
\end{tikzpicture}
"""

    # 占位
    return r"\begin{tikzpicture}\node[red] {该类型渲染正在开发中}; \end{tikzpicture}"

def generate_document(prompt: str, category: str = "Mathematics") -> GenerationResult:
    spec = parse_to_spec(prompt, category)
    latex = render_spec(spec)
    summary = f"Generated {spec.type} in {spec.category}"
    return GenerationResult(latex=_fmt(latex), summary=summary, category=spec.category, spec=spec)
