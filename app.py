# app.py
# scientikz · Streamlit UI (Tabs decide subject) + English NL parsing (optional)
import os
import io
import json
from typing import Optional, Dict, Tuple, List

import streamlit as st
from streamlit.components.v1 import html

from tikz_generator import (
    generate_document,
    GenerationResult,
    _fmt,
    nl_parse_and_render,   # NEW: NL parsing entry
    ValidationError,       # NEW: typed error
)

APP_NAME = os.getenv("SCIENTIKZ_APP_NAME", "scientikz")
st.set_page_config(page_title=f"{APP_NAME} · NLP → TikZ/LaTeX", layout="wide")

# --------- CSS ----------
st.markdown("""
<style>
.stCodeBlock { font-size: 0.92rem; }
div[data-baseweb="tab-list"] { gap: 6px; }
div[data-baseweb="tab"] { border-radius: 10px; }
.pill { padding: 6px 10px; border: 1px solid #d0d7de; border-radius: 999px; }
.subtle { color:#6a737d; }
pre code { white-space: pre-wrap; }
</style>
""", unsafe_allow_html=True)

# --------- Header ----------
def header_bar():
    left, right = st.columns([3, 2])
    with left:
        st.title(APP_NAME)
        st.caption("Natural language → TikZ/PGFPlots/Circuitikz/Chemfig · with English NL parsing (beta)")
    with right:
        st.markdown('<span class="pill">Offline mode</span> <span class="subtle">(no auth in this MVP)</span>', unsafe_allow_html=True)

def copy_button(code_text: str, key: str = "copy_btn"):
    payload = json.dumps(code_text)
    html(
        f"""
        <div style="display:flex;gap:8px;align-items:center;margin:6px 0 12px;">
          <button id="{key}"
                  style="padding:8px 12px;border:1px solid #d0d7de;border-radius:8px;cursor:pointer;background:#fff;">
            📋 Copy code
          </button>
        </div>
        <script>
          const txt_{key} = {payload};
          const btn_{key} = document.getElementById("{key}");
          if (btn_{key}) {{
            btn_{key}.addEventListener("click", async () => {{
              try {{
                await navigator.clipboard.writeText(txt_{key});
                const old = btn_{key}.innerText;
                btn_{key}.innerText = "✅ Copied";
                setTimeout(() => btn_{key}.innerText = old, 1200);
              }} catch (e) {{
                alert("Copy failed: " + e);
              }}
            }});
          }}
        </script>
        """,
        height=52,
    )

# --------- Category structure ----------
def category_structure() -> Dict[str, List[str]]:
    return {
        "Mathematics": ["Vector Analysis", "Calculus", "Algebra", "Other"],
        "Statistics": ["Descriptive", "Inference", "Regression"],
        "Physics": ["Classical Mechanics", "Electromagnetism", "Quantum Mechanics"],
        "Chemistry": ["Organic", "Inorganic", "Biochemistry"],
    }

# --------- Forms ----------
# Math — Vector Analysis (precise UI)
def render_math_vector_form() -> Optional[str]:
    st.markdown("##### Mathematics — Vector Analysis (precise form)")
    mode = st.selectbox("Mode", ["Single vector", "Two vectors (sum/diff/angle)", "Projection / dot / angle"], index=0, key="vec_mode")

    cA, cB, cC = st.columns(3)
    with cA:
        ax_lim = st.number_input("Axes ±X", value=6.0, step=1.0, key="vec_axx")
    with cB:
        ay_lim = st.number_input("Axes ±Y", value=6.0, step=1.0, key="vec_axy")
    with cC:
        grid = st.checkbox("Grid", value=True, key="vec_grid")

    if mode == "Single vector":
        c1, c2, c3, c4 = st.columns(4)
        with c1: vx = st.number_input("v_x", value=3.0, step=0.5, key="vx")
        with c2: vy = st.number_input("v_y", value=2.0, step=0.5, key="vy")
        with c3: x0 = st.number_input("origin x0", value=0.0, step=0.5, key="x0")
        with c4: y0 = st.number_input("origin y0", value=0.0, step=0.5, key="y0")
        components = st.checkbox("Show components (dashed)", value=True, key="vec_comp")
        unit = st.checkbox("Show unit vector", value=True, key="vec_unit")
        if st.button("Generate (single vector)", key="btn_single"):
            return (
                f"VectorAnalysis: mode=single, v=({_fmt(vx)},{_fmt(vy)}), origin=({_fmt(x0)},{_fmt(y0)}), "
                f"components={'true' if components else 'false'}, unit={'true' if unit else 'false'}, "
                f"grid={'true' if grid else 'false'}, axes=({_fmt(ax_lim)},{_fmt(ay_lim)})"
            )

    elif mode == "Two vectors (sum/diff/angle)":
        c1, c2, c3, c4 = st.columns(4)
        with c1: ax = st.number_input("a_x", value=2.0, step=0.5, key="ax")
        with c2: ay = st.number_input("a_y", value=1.0, step=0.5, key="ay")
        with c3: bx = st.number_input("b_x", value=1.0, step=0.5, key="bx")
        with c4: by = st.number_input("b_y", value=3.0, step=0.5, key="by")
        sum_on  = st.checkbox("Show a+b", value=True, key="sum_on")
        diff_on = st.checkbox("Show a-b", value=False, key="diff_on")
        para_on = st.checkbox("Parallelogram guides", value=True, key="para_on")
        ang_on  = st.checkbox("Angle ∠(a,b)", value=True, key="ang_on")
        proj_on = st.checkbox("Projection of a onto b", value=True, key="proj_on")
        if st.button("Generate (two vectors)", key="btn_pair"):
            return (
                f"VectorAnalysis: mode=pair, a=({_fmt(ax)},{_fmt(ay)}), b=({_fmt(bx)},{_fmt(by)}), "
                f"sum={'true' if sum_on else 'false'}, diff={'true' if diff_on else 'false'}, "
                f"parallelogram={'true' if para_on else 'false'}, angle={'true' if ang_on else 'false'}, "
                f"projection_a_on_b={'true' if proj_on else 'false'}, grid={'true' if grid else 'false'}, "
                f"axes=({_fmt(ax_lim)},{_fmt(ay_lim)})"
            )

    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1: ax = st.number_input("a_x", value=3.0, step=0.5, key="ax2")
        with c2: ay = st.number_input("a_y", value=1.0, step=0.5, key="ay2")
        with c3: bx = st.number_input("b_x", value=1.0, step=0.5, key="bx2")
        with c4: by = st.number_input("b_y", value=2.0, step=0.5, key="by2")
        ang_on = st.checkbox("Angle ∠(a,b)", value=True, key="ang_on2")
        proj_on = st.checkbox("Projection of a onto b", value=True, key="proj_on2")
        dot_on = st.checkbox("Dot product a·b", value=True, key="dot_on2")
        if st.button("Generate (projection/dot)", key="btn_proj"):
            return (
                f"VectorAnalysis: mode=pair, a=({_fmt(ax)},{_fmt(ay)}), b=({_fmt(bx)},{_fmt(by)}), "
                f"angle={'true' if ang_on else 'false'}, projection_a_on_b={'true' if proj_on else 'false'}, "
                f"dot={'true' if dot_on else 'false'}, grid={'true' if grid else 'false'}, "
                f"axes=({_fmt(ax_lim)},{_fmt(ay_lim)})"
            )
    return None

# Statistics
def render_stats_form(subcat: str) -> Optional[str]:
    st.markdown("##### Statistics — structured form")
    if subcat == "Descriptive":
        mode = st.radio("Mode", ["Scatter", "Histogram"], horizontal=True, key="stats_desc_mode")
        if mode == "Scatter":
            x_txt = st.text_input("x list", "1,2,3,4,5", key="stats_sc_x")
            y_txt = st.text_input("y list", "2,3,3.5,5,6", key="stats_sc_y")
            grid = st.checkbox("Grid", value=True, key="stats_sc_grid")
            if st.button("Generate (Scatter)", key="btn_stats_scatter"):
                return f"mode=scatter, x=[{x_txt}], y=[{y_txt}], grid={'true' if grid else 'false'}"
        else:
            data_txt = st.text_area("data", "1,1.2,0.9,1.5,2.1,2.0,2.2,2.8,3.0,2.6,3.2", height=100, key="stats_hist_data")
            bins = st.number_input("bins", value=10, min_value=1, max_value=100, key="stats_hist_bins")
            grid = st.checkbox("Grid", value=True, key="stats_hist_grid")
            if st.button("Generate (Histogram)", key="btn_stats_hist"):
                return f"mode=hist, data=[{data_txt}], bins={bins}, grid={'true' if grid else 'false'}"
    elif subcat == "Regression":
        x_txt = st.text_input("x list", "1,2,3,4,5", key="stats_reg_x")
        y_txt = st.text_input("y list", "1.8,2.2,2.9,3.8,5.1", key="stats_reg_y")
        line = st.checkbox("Show regression line", value=True, key="stats_reg_line")
        eq = st.checkbox("Show equation & R²", value=True, key="stats_reg_eq")
        grid = st.checkbox("Grid", value=True, key="stats_reg_grid")
        if st.button("Generate (Regression)", key="btn_stats_reg"):
            return (
                "mode=regression, "
                f"x=[{x_txt}], y=[{y_txt}], line={'true' if line else 'false'}, "
                f"equation={'true' if eq else 'false'}, grid={'true' if grid else 'false'}"
            )
    elif subcat == "Inference":
        mu = st.number_input("μ", value=0.0, key="stats_norm_mu")
        sigma = st.number_input("σ", value=1.0, min_value=0.01, key="stats_norm_sigma")
        rng = st.text_input("Range", "-4,4", key="stats_norm_range")
        if st.button("Generate (Normal curve)", key="btn_stats_norm"):
            a, b = rng.split(",") if "," in rng else ("-4","4")
            return f"Normal curve: mu={_fmt(mu)}, sigma={_fmt(sigma)} from {a} to {b}"
    return None

# Physics — Mechanics
def render_physics_form(subcat: str) -> Optional[str]:
    if subcat != "Classical Mechanics":
        return None
    st.markdown("##### Classical Mechanics — FBD / Projectile")
    mode = st.selectbox("Mode", ["Inclined plane (FBD)", "Projectile motion"], key="phys_mode")

    if mode == "Inclined plane (FBD)":
        theta = st.number_input("θ (deg)", value=30.0, min_value=0.0, max_value=85.0, step=1.0, key="phys_theta")
        mass = st.number_input("m (kg)", value=2.0, min_value=0.0, step=0.5, key="phys_mass")
        mu = st.number_input("μ", value=0.2, step=0.05, key="phys_mu")
        comp = st.checkbox("Show components", value=True, key="phys_comp")
        fric = st.checkbox("Show friction", value=True, key="phys_fric")
        tens = st.checkbox("Show tension", value=False, key="phys_tens")
        if st.button("Generate (Incline)", key="btn_phys_incline"):
            return f"mode=incline, theta={_fmt(theta)}, m={_fmt(mass)}, mu={_fmt(mu)}, components={comp}, friction={fric}, tension={tens}"

    else:
        v0 = st.number_input("v0 (m/s)", value=20.0, step=1.0, key="phys_v0")
        alpha = st.number_input("α (deg)", value=45.0, min_value=0.0, max_value=89.0, key="phys_alpha")
        g = st.number_input("g (m/s²)", value=9.8, step=0.1, key="phys_g")
        marks = st.checkbox("Mark range/apex", value=True, key="phys_marks")
        if st.button("Generate (Projectile)", key="btn_phys_proj"):
            return f"mode=projectile, v0={_fmt(v0)}, alpha={_fmt(alpha)}, g={_fmt(g)}, marks={marks}"
    return None

# Physics — Electromagnetism
def render_em_form(subcat: str) -> Optional[str]:
    if subcat != "Electromagnetism":
        return None
    st.markdown("##### Electromagnetism — Circuit Builder")
    top = st.radio("Topology", ["Series", "Parallel"], horizontal=True, key="em_topo")
    V = st.number_input("Voltage V (V)", value=9.0, step=0.5, key="em_V")
    Rtxt = st.text_input("Resistors (comma-separated)", "100,220,330", key="em_Rs")
    labels = st.checkbox("Show labels", value=True, key="em_labels")
    if st.button("Generate circuit", key="btn_em_gen"):
        topo = "series" if top == "Series" else "parallel"
        return f"mode=circuit, topology={topo}, V={_fmt(V)}, R=[{Rtxt}], labels={'true' if labels else 'false'}"
    return None

# Chemistry
def render_chem_form(subcat: str) -> Optional[str]:
    if subcat not in ["Organic","Inorganic","Biochemistry"]:
        return None
    st.markdown("##### Chemistry — Benzene / Raw molecule")
    mode = st.radio("Mode", ["Benzene (substituted)", "Raw molecule"], horizontal=True, key="chem_mode")

    if mode == "Raw molecule":
        mol = st.text_input("chemfig molecule", "CH3-CH2-OH", key="chem_raw")
        if st.button("Generate molecule", key="chem_btn_raw"):
            return f"mode=raw, mol={mol}"
        return None

    st.caption("Positions 1-6 around ring")
    cols = st.columns(3)
    fields = []
    for i in range(6):
        with cols[i%3]:
            fields.append(st.text_input(f"Pos {i+1}", key=f"chem_p{i+1}"))
    if st.button("Generate benzene", key="chem_btn_bz"):
        subs = []
        for i, s in enumerate(fields, start=1):
            if s.strip():
                subs.append(f"{i}-{s.strip()}")
        subs_part = ",".join(subs) if subs else "none"
        return f"mode=benzene, subs={subs_part}"
    return None

# --------- NL mode helper ----------
def nl_block(category: str):
    st.markdown("###### Natural language mode (English)")
    en_hint = {
        "Mathematics": "e.g., Plot y = sin(x) from -6.28 to 6.28",
        "Statistics":  "e.g., histogram with 10 bins for data 1, 1.2, 0.9, 1.5, 2.1; or scatter x:[1,2,3] y:[2,3,5]",
        "Physics":     "e.g., draw an inclined plane FBD at 30 degrees for a 2 kg block with friction 0.2, show components and friction; or projectile with v0=20, angle=45",
        "Chemistry":   "e.g., benzene ring with CH3 at position 1 and NO2 at position 4; or molecule CH3-CH2-OH",
    }
    nl_text = st.text_area("Describe the figure (English only)", value="", height=110, placeholder=en_hint.get(category, "Describe..."))
    ir_preview = None
    if st.button("Parse & Preview", key=f"parse_{category}"):
        try:
            ir = nl_parse_and_render(nl_text, category, preview_only=True)
            ir_preview = ir
            st.success("Parsed successfully. Please review the parameters below.")
            st.json(ir_preview)
        except ValidationError as ve:
            st.error(str(ve))
        except Exception as e:
            st.error(f"Parse failed: {e}")
    gen_click = st.button("Generate from NL", key=f"gen_{category}")
    return nl_text, ir_preview, gen_click

# --------- Category UI ----------
def category_ui() -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    tabs = st.tabs(["🧮 Mathematics", "📊 Statistics", "🧲 Physics", "🧪 Chemistry"])
    struct = category_structure()
    for tab, label in zip(tabs, struct.keys()):
        with tab:
            st.subheader(label)
            sub = st.selectbox("Subcategory", struct[label], key=f"sub_{label}")

            # NL mode toggle
            nl_mode = st.toggle("Natural language mode (English)", value=False, key=f"nl_{label}")

            if nl_mode:
                nl_text, ir_preview, gen_click = nl_block(label)
                if gen_click:
                    return nl_text, label, sub, "NL"
                # still allow fallback forms below if desired
                st.divider()

            # Hook forms (classic)
            if label == "Mathematics" and sub == "Vector Analysis":
                form_prompt = render_math_vector_form()
                with st.expander("Or type directive directly"):
                    ta_val = st.text_area("Instruction", key=f"ta_{label}_{sub}", height=140)
                    if st.button(f"Generate in {label} · {sub} (text)", key=f"go_{label}_{sub}_text"):
                        return ta_val, "Mathematics", sub, "DIRECTIVE"
                if form_prompt:
                    return form_prompt, "Mathematics", sub, "DIRECTIVE"
                return None, None, None, None

            if label == "Statistics":
                form_prompt = render_stats_form(sub)
                with st.expander("Or type directive directly"):
                    ta_val = st.text_area("Instruction", key=f"ta_{label}_{sub}", height=140)
                    if st.button(f"Generate in {label} · {sub} (text)", key=f"go_{label}_{sub}_text"):
                        return ta_val, "Statistics", sub, "DIRECTIVE"
                if form_prompt:
                    return form_prompt, "Statistics", sub, "DIRECTIVE"
                return None, None, None, None

            if label == "Physics":
                if sub == "Classical Mechanics":
                    form_prompt = render_physics_form(sub)
                elif sub == "Electromagnetism":
                    form_prompt = render_em_form(sub)
                else:
                    form_prompt = None
                with st.expander("Or type directive directly"):
                    ta_val = st.text_area("Instruction", key=f"ta_{label}_{sub}", height=140)
                    if st.button(f"Generate in {label} · {sub} (text)", key=f"go_{label}_{sub}_text"):
                        return ta_val, "Physics", sub, "DIRECTIVE"
                if form_prompt:
                    return form_prompt, "Physics", sub, "DIRECTIVE"
                return None, None, None, None

            if label == "Chemistry":
                form_prompt = render_chem_form(sub)
                with st.expander("Or type directive directly"):
                    ta_val = st.text_area("Instruction", key=f"ta_{label}_{sub}", height=140)
                    if st.button(f"Generate in {label} · {sub} (text)", key=f"go_{label}_{sub}_text"):
                        return ta_val, "Chemistry", sub, "DIRECTIVE"
                if form_prompt:
                    return form_prompt, "Chemistry", sub, "DIRECTIVE"
                return None, None, None, None

            # Default text-only flow for other subcategories
            ta_val = st.text_area("Instruction", key=f"ta_{label}_{sub}", height=160)
            if st.button(f"Generate in {label} · {sub}", key=f"go_{label}_{sub}"):
                return ta_val, label, sub, "DIRECTIVE"
    return None, None, None, None

# --------- Main ----------
def main_page():
    header_bar()
    st.markdown("### Generate TikZ/LaTeX from natural language or structured directives")
    prompt, category, subcategory, mode = category_ui()
    if prompt and category:
        try:
            if mode == "NL":
                result: GenerationResult = nl_parse_and_render(prompt, category, preview_only=False)
            else:
                result: GenerationResult = generate_document(prompt, category, subcategory)
            st.success(f"Generated successfully in {category} · {subcategory} ({'NL' if mode=='NL' else 'Directive/Form'})")
            st.code(result.latex, language="latex")
            c1, c2 = st.columns([1,2])
            with c1:
                copy_button(result.latex, key=f"copy_tex_{category}")
            with c2:
                b = io.BytesIO(result.latex.encode("utf-8"))
                st.download_button("⬇️ Download .tex", data=b, file_name="scientikz_output.tex", mime="text/plain")
            st.markdown(
                "- Requires LaTeX packages (auto-included): tikz, pgfplots (if needed), circuitikz (if needed), chemfig (if needed)\n"
                "- Compile: `pdflatex --halt-on-error scientikz_output.tex`"
            )
        except ValidationError as ve:
            st.error(str(ve))
        except Exception as e:
            st.error(f"Error: {e}")

def router():
    main_page()

router()
