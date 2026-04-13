import os
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from jinja2 import Template

@dataclass
class GraphSpec:
    type: str  # function_plot, vector, force_diagram
    category: str
    title: str = ""
    domain: Optional[List[float]] = None
    plots: List[Dict] = None  # [{'expr': 'sin(\\x r)', 'style': 'blue'}]
    vectors: List[Dict] = None  # [{'start': [0,0], 'end': [2,2], 'label': 'v'}]
    extra_configs: Dict[str, Any] = None

    def __post_init__(self):
        if self.domain is None: self.domain = [-5, 5]
        if self.plots is None: self.plots = []
        if self.vectors is None: self.vectors = []
        if self.extra_configs is None: self.extra_configs = {}

# ====================== LLM 解析接口 (模拟) ======================
def call_llm_for_spec(prompt: str, category: str) -> Dict:
    """
    未来在此处接入 Gemini 或 GPT-4 API。
    目前根据关键词返回结构化数据。
    """
    p = prompt.lower()
    if "sin" in p or "正弦" in p:
        return {
            "type": "function_plot",
            "category": "Mathematics",
            "domain": [-6.28, 6.28],
            "plots": [{"expr": "sin(\\x r)", "style": "blue, thick"}]
        }
    elif any(kw in p for kw in ["force", "受力", "gravity", "重力"]):
        return {
            "type": "force_diagram",
            "category": "Physics",
            "vectors": [
                {"start": [0,0], "end": [0, -2], "label": "mg"},
                {"start": [0,0], "end": [0, 2], "label": "N"},
                {"start": [0,0], "end": [1.5, 0], "label": "f"}
            ]
        }
    # 默认返回矢量示例
    return {
        "type": "vector",
        "category": category,
        "vectors": [{"start": [0,0], "end": [3, 2], "label": "v"}]
    }

# ====================== Jinja2 绘图模板 ======================
# 包含自动矢量箭头规范与水印
TIKZ_TEMPLATES = {
    "function_plot": """
\\begin{tikzpicture}[cap=round, line join=round, >=stealth]
  \\draw[->] ({{domain[0]-0.5}}, 0) -- ({{domain[1]+0.5}}, 0) node[right] {$x$};
  \\draw[->] (0, -3.5) -- (0, 3.5) node[above] {$y$};
  \\draw[very thin, lightgray!30] ({{domain[0]}}, -3) grid ({{domain[1]}}, 3);
  {% for plot in plots %}
  \\draw[domain={{domain[0]}}:{{domain[1]}}, smooth, variable=\\x, {{plot.style}}] plot (\\x, { {{plot.expr}} });
  {% endfor %}
  % 水印标注
  \\node[gray, opacity=0.5, font=\\tiny] at ({{domain[1]-1}}, -3.2) {Elliott Math Coaching};
  \\node[gray, opacity=0.5, font=\\tiny] at ({{domain[1]-1}}, -3.5) {曹老师数学辅导};
\\end{tikzpicture}
""",
    "vector": """
\\begin{tikzpicture}[>=stealth]
  \\draw[very thin, gray!20] (-1,-1) grid (5,5);
  \\draw[->] (-0.5,0) -- (5.5,0) node[right] {$x$};
  \\draw[->] (0,-0.5) -- (0,5.5) node[above] {$y$};
  {% for v in vectors %}
  \\draw[->, ultra thick, blue] ({{v.start[0]}}, {{v.start[1]}}) -- ({{v.end[0]}}, {{v.end[1]}}) 
    node[midway, above right] {$\\vec{ {{v.label}} }$};
  {% endfor %}
  \\node[gray, opacity=0.3, font=\\tiny] at (4.5, -0.3) {Elliott Math Coaching};
\\end{tikzpicture}
""",
    "force_diagram": """
\\begin{tikzpicture}[>=stealth]
  \\node (obj) [circle, fill=gray!10, draw, thick, minimum size=1.2cm] {m};
  {% for v in vectors %}
  \\draw[->, ultra thick, red] (obj.center) -- ++({{v.end[0]}}, {{v.end[1]}}) 
    node[at end, right] {${{v.label}}$};
  {% endfor %}
  \\node[gray, opacity=0.3, font=\\tiny] at (2, -2) {曹老师数学辅导};
\\end{tikzpicture}
"""
}

def render_spec(spec: GraphSpec) -> str:
    template_str = TIKZ_TEMPLATES.get(spec.type, "\\node{Unknown Type};")
    tmpl = Template(template_str)
    return tmpl.render(asdict(spec)).strip()

def generate_document(prompt: str, category: str = "Mathematics"):
    spec_data = call_llm_for_spec(prompt, category)
    spec = GraphSpec(**spec_data)
    latex = render_spec(spec)
    # 使用 type 构造一个简单的返回对象
    return type('Result', (), {'latex': latex, 'summary': f"Generated {spec.type}", 'category': category})
