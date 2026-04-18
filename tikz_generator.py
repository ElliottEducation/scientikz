import re
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from jinja2 import Template

@dataclass
class GraphSpec:
    type: str
    category: str
    sub_category: str
    domain: Optional[List[float]] = None
    plots: List[Dict] = None
    vectors: List[Dict] = None
    extra_params: Dict[str, Any] = None

    def __post_init__(self):
        if self.domain is None: self.domain = [-5, 5]
        if self.plots is None: self.plots = []
        if self.vectors is None: self.vectors = []
        if self.extra_params is None: self.extra_params = {"scale": 1.0}

# ====================== Day 1: Advanced Math Parser ======================
def parse_prompt_to_spec(prompt: str, domain: str, sub_cat: str, scale: float = 1.0) -> GraphSpec:
    p = prompt.lower()
    spec = GraphSpec(type="generic", category=domain, sub_category=sub_cat)
    spec.extra_params["scale"] = scale

    # 1. Calculus: Integral Area (定积分面积阴影)
    if any(kw in p for kw in ["integral", "area", "积分", "面积"]):
        spec.type = "calculus_area"
        spec.extra_params["expr"] = "0.5*x*x" if "x^2" in p else "sin(x r)"
        spec.extra_params["range"] = [0, 2] # Default integration range
        
    # 2. Complex Plane: N-th Roots of Unity (单位根)
    elif any(kw in p for kw in ["roots of unity", "单位根"]):
        spec.type = "complex_roots"
        match = re.search(r'(\d+)\s*(th|次|st|nd|rd)|n\s*=\s*(\d+)', p)
        spec.extra_params["n"] = int(match.group(1) or match.group(3)) if match else 5
        
    # 3. Trigonometry: Sine/Cosine Wave (三角函数)
    elif any(kw in p for kw in ["trig", "sin", "cos", "正弦", "余弦"]):
        spec.type = "trig_wave"
        expr = "cos(\\x r)" if "cos" in p or "余弦" in p else "sin(\\x r)"
        spec.plots = [{"expr": expr, "style": "blue, thick"}]
        
    # 4. Vectors: Addition & Parallelogram (向量合成)
    elif any(kw in p for kw in ["vector", "向量", "addition", "加法"]):
        spec.type = "vector_addition"
        spec.vectors = [
            {"end": [3, 1], "label": "u", "color": "blue"},
            {"end": [1, 3], "label": "v", "color": "red"}
        ]

    # 5. Physics: Inclined Plane (修正版重力方向)
    elif "inclined" in p or "斜面" in p:
        spec.type = "physics_incline"
        match = re.search(r'(\d+)\s*(degree|度)', p)
        spec.extra_params["angle"] = int(match.group(1)) if match else 30

    return spec

# ====================== Specialist Template Library ======================
TIKZ_TEMPLATES = {
    "calculus_area": """
\\begin{tikzpicture}[>=stealth, scale={{extra_params.scale}}]
  \\draw[->] (-1,0) -- (4,0) node[right] {$x$};
  \\draw[->] (0,-1) -- (0,3) node[above] {$y$};
  % Shaded Area
  \\fill[blue!20] ({{extra_params.range[0]}},0) -- plot[domain={{extra_params.range[0]}}:{{extra_params.range[1]}}] (\\x, { {{extra_params.expr}} }) -- ({{extra_params.range[1]}},0) -- cycle;
  % Function Curve
  \\draw[domain=-0.5:3.5, smooth, thick, blue] plot (\\x, { {{extra_params.expr}} }) node[right] {$f(x)$};
  \\node[below] at ({{extra_params.range[0]}},0) {$a$};
  \\node[below] at ({{extra_params.range[1]}},0) {$b$};
\\end{tikzpicture}
""",
    "vector_addition": """
\\begin{tikzpicture}[>=stealth, scale={{extra_params.scale}}]
  \\draw[very thin, gray!20] (-1,-1) grid (5,5);
  \\draw[->] (-0.5,0) -- (5,0) node[right] {$x$};
  \\draw[->] (0,-0.5) -- (0,5) node[above] {$y$};
  % Vectors
  \\draw[->, thick, {{vectors[0].color}}] (0,0) -- ({{vectors[0].end[0]}}, {{vectors[0].end[1]}}) node[midway, above left] {$\\vec{ {{vectors[0].label}} }$};
  \\draw[->, thick, {{vectors[1].color}}] (0,0) -- ({{vectors[1].end[0]}}, {{vectors[1].end[1]}}) node[midway, below right] {$\\vec{ {{vectors[1].label}} }$};
  % Resultant
  \\draw[dashed, gray] ({{vectors[0].end[0]}}, {{vectors[0].end[1]}}) -- ({ {{vectors[0].end[0]}}+{{vectors[1].end[0]}} }, { {{vectors[0].end[1]}}+{{vectors[1].end[1]}} });
  \\draw[dashed, gray] ({{vectors[1].end[0]}}, {{vectors[1].end[1]}}) -- ({ {{vectors[0].end[0]}}+{{vectors[1].end[0]}} }, { {{vectors[0].end[1]}}+{{vectors[1].end[1]}} });
  \\draw[->, ultra thick, purple] (0,0) -- ({ {{vectors[0].end[0]}}+{{vectors[1].end[0]}} }, { {{vectors[0].end[1]}}+{{vectors[1].end[1]}} }) node[above right] {$\\vec{u}+\\vec{v}$};
\\end{tikzpicture}
""",
    "complex_roots": """
\\begin{tikzpicture}[>=stealth, scale={{extra_params.scale}}]
  \\draw[very thin, gray!20] (-2.2,-2.2) grid (2.2,2.2);
  \\draw[->] (-2.5,0) -- (2.5,0) node[right] {$\\text{Re}$};
  \\draw[->] (0,-2.5) -- (0,2.5) node[above] {$\\text{Im}$};
  \\draw[dashed, gray] (0,0) circle (2);
  \\foreach \\i in {1,...,{{extra_params.n}}} {
    \\pgfmathsetmacro{\\a}{(\\i-1)*360/{{extra_params.n}}}
    \\draw[->, orange, thick] (0,0) -- (\\a:2);
    \\fill[orange] (\\a:2) circle (1.5pt);
  }
  \\node[orange, xshift=10pt, yshift=5pt] at (0:2) {$\\omega_0$};
\\end{tikzpicture}
""",
    "physics_incline": """
\\begin{tikzpicture}[>=stealth, scale={{extra_params.scale}}]
  \\draw[thick] (0,0) -- (5,0) -- (5,{5*tan({{extra_params.angle}})}) -- cycle;
  \\begin{scope}[rotate={{extra_params.angle}}]
    \\draw[thick, fill=blue!10] (2,0) rectangle (3.2,0.8);
    \\draw[->, green!60!black, thick] (2.6,0.4) -- ++(0,1.5) node[above] {$\\vec{N}$};
    \\draw[->, orange, thick] (2.6,0.4) -- ++(-1.2,0) node[above] {$\\vec{f}$};
    % 重力方向修正逻辑
    \\draw[->, red, thick] (2.6,0.4) -- ++({270-{{extra_params.angle}}}:2) node[right] {$m\\vec{g}$};
  \\end{scope}
\\end{tikzpicture}
""",
    "trig_wave": """
\\begin{tikzpicture}[>=stealth, scale={{extra_params.scale}}]
  \\draw[->] (-0.5,0) -- (6.8,0) node[right] {$x$};
  \\draw[->] (0,-1.5) -- (0,1.5) node[above] {$y$};
  \\draw[domain=0:6.28, samples=100, blue, thick] plot (\\x, { {{plots[0].expr}} });
  \\node[below] at (3.14,0) {$\\pi$}; \\node[below] at (6.28,0) {$2\\pi$};
\\end{tikzpicture}
"""
}

def render_spec(spec: GraphSpec) -> str:
    tmpl = Template(TIKZ_TEMPLATES.get(spec.type, "\\node{Scene placeholder for " + spec.sub_category + "};"))
    return tmpl.render(asdict(spec)).strip()

def generate_document(prompt, domain, sub_cat, scale=1.0):
    spec = parse_prompt_to_spec(prompt, domain, sub_cat, scale)
    return type('Res', (), {'latex': render_spec(spec), 'summary': spec.type})
