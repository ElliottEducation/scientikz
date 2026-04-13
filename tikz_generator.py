import re
import json
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

def parse_prompt_to_spec(prompt: str, domain: str, sub_cat: str, scale: float = 1.0) -> GraphSpec:
    p = prompt.lower()
    spec = GraphSpec(type="generic", category=domain, sub_category=sub_cat)
    spec.extra_params["scale"] = scale

    if "roots of unity" in p or "单位根" in p:
        spec.type = "complex_roots"
        match = re.search(r'(\d+)\s*(th|次|st|nd|rd)|n\s*=\s*(\d+)', p)
        spec.extra_params["n"] = int(match.group(1) or match.group(3)) if match else 5
        
    elif any(kw in p for kw in ["trig", "sin", "cos", "正弦", "余弦", "波"]):
        spec.type = "trig_wave"
        expr = "cos(\\x r)" if "cos" in p or "余弦" in p else "sin(\\x r)"
        spec.plots = [{"expr": expr, "style": "blue, thick, smooth"}]
        
    elif "inclined" in p or "斜面" in p:
        spec.type = "physics_incline"
        match = re.search(r'(\d+)\s*(degree|度)', p)
        spec.extra_params["angle"] = int(match.group(1)) if match else 30

    elif any(kw in p for kw in ["vector", "向量", "矢量", "arrow"]):
        spec.type = "math_vector"
        spec.vectors = [{"start": [0,0], "end": [3,2], "label": "v"}, {"start": [0,0], "end": [-1,3], "label": "u"}]
    
    return spec

TIKZ_TEMPLATES = {
    "trig_wave": """
\\begin{tikzpicture}[>=stealth, scale={{extra_params.scale}}]
  \\draw[->] (-0.5,0) -- (6.5,0) node[right] {$x$};
  \\draw[->] (0,-1.5) -- (0,1.5) node[above] {$f(x)$};
  \\draw[very thin, gray!30, dashed] (0,-1) grid (6,1);
  {% for plot in plots %}
  \\draw[domain=0:6.28, samples=100, {{plot.style}}] plot (\\x, { {{plot.expr}} });
  {% endfor %}
  \\node[below] at (1.57,0) {$\\frac{\\pi}{2}$};
  \\node[below] at (3.14,0) {$\\pi$};
  \\node[below] at (4.71,0) {$\\frac{3\\pi}{2}$};
  \\node[below] at (6.28,0) {$2\\pi$};
\\end{tikzpicture}
""",
    "complex_roots": """
\\begin{tikzpicture}[>=stealth, scale={{extra_params.scale}}]
  \\draw[very thin, gray!20, step=0.5] (-2.2,-2.2) grid (2.2,2.2);
  \\draw[->] (-2.5,0) -- (2.5,0) node[right] {$\\text{Re}$};
  \\draw[->] (0,-2.5) -- (0,2.5) node[above] {$\\text{Im}$};
  \\draw[dashed, gray!80] (0,0) circle (2);
  \\pgfmathsetmacro{\\angleStep}{360/{{extra_params.n}}}
  \\foreach \\i in {1,...,{{extra_params.n}}} {
    \\pgfmathsetmacro{\\a}{(\\i-1)*\\angleStep}
    \\draw[->, orange, thick] (0,0) -- (\\a:2);
    \\fill[orange!80!black] (\\a:2) circle (2pt);
  }
  \\node[orange!80!black, above right] at (0:2) {$\\omega_0 = 1$};
\\end{tikzpicture}
""",
    "math_vector": """
\\begin{tikzpicture}[>=stealth, scale={{extra_params.scale}}]
  \\draw[very thin, gray!15] (-2,-1) grid (4,4);
  \\draw[->] (-2.5,0) -- (4.5,0) node[right] {$x$};
  \\draw[->] (0,-1.5) -- (0,4.5) node[above] {$y$};
  \\node[below left] at (0,0) {$O$};
  {% for v in vectors %}
  \\draw[->, ultra thick, blue!80!black] ({{v.start[0]}}, {{v.start[1]}}) -- ({{v.end[0]}}, {{v.end[1]}}) 
    node[midway, sloped, above] {$\\vec{ {{v.label}} }$};
  {% endfor %}
\\end{tikzpicture}
""",
    "physics_incline": """
\\begin{tikzpicture}[>=stealth, scale={{extra_params.scale}}]
  \\draw[thick] (0,0) -- (5,0) -- (5,{5*tan({{extra_params.angle}})}) -- cycle;
  \\begin{scope}[rotate={{extra_params.angle}}]
    \\draw[thick, fill=blue!10, rounded corners=1pt] (2,0) rectangle (3.2,0.8);
    \\draw[->, red, thick] (2.6,0.4) -- ++(0,-2) node[right] {$m\\vec{g}$};
    \\draw[->, green!60!black, thick] (2.6,0.4) -- ++(0,1.5) node[above] {$\\vec{N}$};
    \\draw[->, orange, thick] (2.6,0.4) -- ++(-1.2,0) node[above] {$\\vec{f}$};
  \\end{scope}
  \\draw (1,0) arc (0:{{extra_params.angle}}:1) node[midway, right, xshift=2pt] {$\\theta$};
\\end{tikzpicture}
"""
}

def render_spec(spec: GraphSpec) -> str:
    tmpl_str = TIKZ_TEMPLATES.get(spec.type, "\\node{Waiting for specific scene...};")
    return Template(tmpl_str).render(asdict(spec)).strip()

def generate_document(prompt: str, domain: str, sub_cat: str, scale: float = 1.0):
    spec = parse_prompt_to_spec(prompt, domain, sub_cat, scale)
    latex = render_spec(spec)
    return type('Res', (), {'latex': latex, 'summary': spec.type})
