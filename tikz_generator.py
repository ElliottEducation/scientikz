# tikz_generator.py
from dataclasses import dataclass, asdict
import re
import math
from typing import Optional, List, Dict, Any
import json

# ===============================
# GraphSpec 数据模型（核心！）
# ===============================
@dataclass
class GraphSpec:
    type: str                          # "vector", "function_plot", "calculus_area", ...
    category: str                      # "Mathematics · Vector Analysis"
    title: str = ""
    domain: Optional[List[float]] = None
    axes: Dict[str, bool] = None
    plots: List[Dict] = None
    vectors: List[Dict] = None
    labels: List[Dict] = None
    grid: bool = True
    theme: str = "academic"
    extra: Dict[str, Any] = None       # 额外参数

    def __post_init__(self):
        if self.axes is None:
            self.axes = {"x": True, "y": True, "grid": self.grid}
        if self.plots is None:
            self.plots = []
        if self.vectors is None:
            self.vectors = []
        if self.labels is None:
            self.labels = []
        if self.extra is None:
            self.extra = {}

# ===============================
# 输出结果
# ===============================
@dataclass
class GenerationResult:
    latex: str
    summary: str
    category: str
    spec: GraphSpec = None   # 新增：方便调试

def _fmt(code: str) -> str:
    return code.strip().replace("\t", "    ")

# ===============================
# 通用工具函数（保持不变）
# ===============================
_NUM = r"-?\d+(?:\.\d+)?"

def _find_float_pair(text: str, label: str):
    m = re.search(rf"{label}\s*=\s*\(\s*({_NUM})\s*,\s*({_NUM})\s*\)", text, flags=re.I)
    if not m:
        m = re.search(rf"{label}\s*\(?\s*({_NUM})\s*,\s*({_NUM})\s*\)?", text, flags=re.I)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None

def _find_float(text: str, *keys: str, default: Optional[float] = None) -> Optional[float]:
    for k in keys:
        m = re.search(rf"{k}\s*[:=]?\s*({_NUM})", text, flags=re.I)
        if m:
            return float(m.group(1))
    return default

def _has(text: str, *words: str) -> bool:
    return any(re.search(rf"\b{re.escape(w)}\b", text, flags=re.I) for w in words)

# ===============================
# Parser → GraphSpec（新增核心解析层）
# ===============================
def parse_to_spec(prompt: str, category: str = "Mathematics") -> GraphSpec:
    p = prompt.strip().lower()
    
    if "vector" in p or "矢量" in p or "v=" in p:
        v = _find_float_pair(prompt, r"v") or _find_float_pair(prompt, r"vector") or (3.0, 2.0)
        origin = _find_float_pair(prompt, r"origin") or (0.0, 0.0)
        return GraphSpec(
            type="vector",
            category=f"{category} · Vector Analysis",
            vectors=[{"start": origin, "end": (origin[0]+v[0], origin[1]+v[1]), "label": r"\vec v"}],
            grid=_has(p, "grid")
        )
    
    elif "y=" in p or "函数" in p or "plot" in p:
        m = re.search(r"y\s*=\s*([^;,\n]+)", prompt, flags=re.I)
        expr = m.group(1).strip() if m else "x^2"
        m2 = re.search(r"from\s*({_NUM})\s*(?:to|-)\s*({_NUM})".format(_NUM=_NUM), prompt, flags=re.I)
        domain = (float(m2.group(1)), float(m2.group(2))) if m2 else (-3.0, 3.0)
        return GraphSpec(
            type="function_plot",
            category=f"{category} · Algebra",
            domain=list(domain),
            plots=[{"expr": expr, "style": "blue"}],
            grid=True
        )
    
    # 未来可轻松扩展 calculus / physics / chemistry
    else:
        # 默认 fallback
        return GraphSpec(type="unknown", category=category)

# ===============================
# GraphSpec → TikZ 渲染（统一入口）
# ===============================
def render_spec(spec: GraphSpec) -> str:
    if spec.type == "vector":
        v = spec.vectors[0]
        sx, sy = v["start"]
        ex, ey = v["end"]
        L = math.ceil(max(abs(ex), abs(ey), 4.0))
        
        body = rf"""
\begin{{tikzpicture}}[>=stealth]
  \draw[->] (-{L},0) -- ({L},0) node[right] {{$x$}};
  \draw[->] (0,-{L}) -- (0,{L}) node[above] {{$y$}};
  {"\\draw[very thin,step=1cm,color=gray!30] (-{L},-{L}) grid ({L},{L});" if spec.grid else "% no grid"}
  \draw[->,very thick,blue] ({sx:.2f},{sy:.2f}) -- ({ex:.2f},{ey:.2f}) node[above right] {{{v.get("label", r"\vec v")}}};
\end{{tikzpicture}}
"""
        return body.strip()
    
    elif spec.type == "function_plot":
        a, b = spec.domain
        expr = spec.plots[0]["expr"].replace("arctan", "atan")
        body = rf"""
\begin{{tikzpicture}}
  \draw[->] ({a:.2f},0) -- ({b:.2f},0) node[right] {{$x$}};
  \draw[->] (0,-3) -- (0,3) node[above] {{$y$}};
  \draw[domain={a:.2f}:{b:.2f},smooth,variable=\x,blue] plot (\x,{{{expr}}});
\end{{tikzpicture}}
"""
        return body.strip()
    
    return "% GraphSpec rendering not implemented yet"

# ===============================
# 主生成函数（对外唯一入口）
# ===============================
def generate_document(prompt: str, category: str = "Mathematics") -> GenerationResult:
    spec = parse_to_spec(prompt, category)
    latex = render_spec(spec)
    summary = f"Generated {spec.type} in {spec.category}"
    
    return GenerationResult(
        latex=_fmt(latex),
        summary=summary,
        category=spec.category,
        spec=spec
    )
