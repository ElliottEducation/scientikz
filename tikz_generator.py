from dataclasses import dataclass

# -------------------------------
# Dataclass to hold generation results
# -------------------------------
@dataclass
class GenerationResult:
    latex: str
    summary: str
    category: str

# -------------------------------
# Formatting helper
# -------------------------------
def _fmt(code: str) -> str:
    return code.strip()

# -------------------------------
# Main generator
# -------------------------------
def generate_document(prompt: str, category_hint: str, subcategory_hint: str = None) -> GenerationResult:
    """
    Core function to generate TikZ/LaTeX code based on the prompt.
    """
    # Placeholder example logic — replace with real model calls
    if category_hint == "Mathematics":
        latex = rf"""
\\begin{{tikzpicture}}
    \\draw[->] (-3,0) -- (3,0) node[right] {{x}};
    \\draw[->] (0,-3) -- (0,3) node[above] {{y}};
    \\draw[domain=-3:3,smooth,variable=\\x,blue] plot (\\x,{{\\x*\\x}});
\\end{{tikzpicture}}
"""
        summary = "Generated a quadratic function plot y=x^2."
    elif category_hint == "Statistics":
        latex = rf"""
\\begin{{tikzpicture}}
    \\draw[->] (0,0) -- (5,0) node[right] {{Value}};
    \\draw[->] (0,0) -- (0,4) node[above] {{Frequency}};
    \\draw[fill=blue!30] (0.5,0) rectangle (1.5,2);
    \\draw[fill=blue!30] (2.0,0) rectangle (3.0,3);
    \\draw[fill=blue!30] (3.5,0) rectangle (4.5,1);
\\end{{tikzpicture}}
"""
        summary = "Generated a basic histogram."
    elif category_hint == "Physics":
        latex = rf"""
\\begin{{circuitikz}}
    \\draw (0,0) to[R=2<\\ohm>] (2,0)
          to[L=3<\\henry>] (4,0)
          to[C=4<\\micro\\farad>] (6,0)
          to[short] (6,-2)
          to[short] (0,-2)
          to[short] (0,0);
\\end{{circuitikz}}
"""
        summary = "Generated a simple RLC circuit."
    elif category_hint == "Chemistry":
        latex = rf"""
\\chemfig{{C(-[2]H)(-[6]H)-C(-[2]H)(-[6]H)-OH}}
"""
        summary = "Generated ethanol molecule structure."
    else:
        latex = "% Unknown category"
        summary = "No valid generation logic found."

    return GenerationResult(latex=_fmt(latex), summary=summary, category=category_hint)

# -------------------------------
# Backward compatibility shim
# -------------------------------
def nl_parse_and_render(prompt: str, category_hint: str = "Auto", subcategory_hint: str = None):
    """
    Legacy wrapper to support old app.py versions.
    Returns (latex, meta_dict)
    """
    result = generate_document(prompt, category_hint, subcategory_hint)
    meta = {
        "category": result.category,
        "summary": result.summary,
    }
    return result.latex, meta
