# app.py
# scientikz · Streamlit + Supabase optional
# Full UI with all features integrated
import os
import io
import time
import json
from typing import Optional, Dict, Tuple, List

import streamlit as st
from streamlit.components.v1 import html

try:
    from supabase import create_client, Client
except Exception:
    create_client = None
    Client = None

from tikz_generator import (
    generate_document,
    GenerationResult,
    _fmt
)

# -----------------------
# App config
# -----------------------
APP_NAME = os.getenv("SCIENTIKZ_APP_NAME", "scientikz")
st.set_page_config(page_title=f"{APP_NAME} · NLP → TikZ/LaTeX", layout="wide")

# CSS polish
st.markdown("""
<style>
.stCodeBlock { font-size: 0.92rem; }
div[data-baseweb="tab-list"] { gap: 6px; }
div[data-baseweb="tab"] { border-radius: 10px; }
.pill { padding: 6px 10px; border: 1px solid #d0d7de; border-radius: 999px; }
</style>
""", unsafe_allow_html=True)

# -----------------------
# Supabase client optional
# -----------------------
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_KEY and create_client is not None:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)  # type: ignore
    except Exception:
        supabase = None

# -----------------------
# Session state init
# -----------------------
def init_state():
    if "auth" not in st.session_state:
        st.session_state["auth"] = {"user": None, "role": "free", "access_token": None}
    if "page" not in st.session_state:
        st.session_state["page"] = "home"

init_state()

# -----------------------
# UI helpers
# -----------------------
def header_bar():
    left, right = st.columns([3, 2])
    with left:
        st.title(APP_NAME)
        st.caption("Natural language → TikZ/PGFPlots/Circuitikz/Chemfig")
    with right:
        if supabase and st.session_state["auth"]["user"]:
            st.markdown(f"**{st.session_state['auth']['user']['email']}** · Role: `{st.session_state['auth']['role']}`")
            if st.button("Sign out"):
                st.session_state["auth"] = {"user": None, "role": "free", "access_token": None}
                st.rerun()
        elif supabase:
            st.markdown('<span class="pill">Sign in disabled in MVP</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="pill">Offline mode</span>', unsafe_allow_html=True)

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

# -----------------------
# Form renderers
# -----------------------

# --- Statistics ---
def render_stats_form(subcat: str) -> Optional[str]:
    st.markdown("##### Statistics — structured form")
    if subcat == "Descriptive":
        mode = st.radio("Mode", ["Scatter", "Histogram"], horizontal=True, key="stats_desc_mode")
        if mode == "Scatter":
            x_txt = st.text_input("x list", "1,2,3,4,5", key="stats_sc_x")
            y_txt = st.text_input("y list", "2,3,3.5,5,6", key="stats_sc_y")
            grid = st.checkbox("Grid", value=True, key="stats_sc_grid")
            if st.button("Generate (Scatter)", key="btn_stats_scatter"):
                return f"Stats: mode=scatter, x=[{x_txt}], y=[{y_txt}], grid={'true' if grid else 'false'}"
        else:
            data_txt = st.text_area("data", "1,1.2,0.9,1.5,2.1,2.0,2.2,2.8,3.0,2.6,3.2", height=100, key="stats_hist_data")
            bins = st.number_input("bins", value=10, min_value=1, max_value=100, key="stats_hist_bins")
            grid = st.checkbox("Grid", value=True, key="stats_hist_grid")
            if st.button("Generate (Histogram)", key="btn_stats_hist"):
                return f"Stats: mode=hist, data=[{data_txt}], bins={bins}, grid={'true' if grid else 'false'}"
    elif subcat == "Regression":
        x_txt = st.text_input("x list", "1,2,3,4,5", key="stats_reg_x")
        y_txt = st.text_input("y list", "1.8,2.2,2.9,3.8,5.1", key="stats_reg_y")
        line = st.checkbox("Show regression line", value=True, key="stats_reg_line")
        eq = st.checkbox("Show equation & R²", value=True, key="stats_reg_eq")
        grid = st.checkbox("Grid", value=True, key="stats_reg_grid")
        if st.button("Generate (Regression)", key="btn_stats_reg"):
            return (
                "Stats: mode=regression, "
                f"x=[{x_txt}], y=[{y_txt}], line={'true' if line else 'false'}, "
                f"equation={'true' if eq else 'false'}, grid={'true' if grid else 'false'}"
            )
    elif subcat == "Inference":
        mu = st.number_input("μ", value=0.0, key="stats_norm_mu")
        sigma = st.number_input("σ", value=1.0, min_value=0.01, key="stats_norm_sigma")
        rng = st.text_input("Range", "-4,4", key="stats_norm_range")
        if st.button("Generate (Normal curve)", key="btn_stats_norm"):
            a, b = rng.split(",") if "," in rng else ("-4","4")
            return f"Normal curve: mu={mu}, sigma={sigma} from {a} to {b}"
    return None

# --- Physics: Mechanics ---
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
            return f"Physics: mode=incline, theta={theta}, m={mass}, mu={mu}, components={comp}, friction={fric}, tension={tens}"

    else:
        v0 = st.number_input("v0 (m/s)", value=20.0, step=1.0, key="phys_v0")
        alpha = st.number_input("α (deg)", value=45.0, min_value=0.0, max_value=89.0, key="phys_alpha")
        g = st.number_input("g (m/s²)", value=9.8, step=0.1, key="phys_g")
        marks = st.checkbox("Mark range/apex", value=True, key="phys_marks")
        if st.button("Generate (Projectile)", key="btn_phys_proj"):
            return f"Physics: mode=projectile, v0={v0}, alpha={alpha}, g={g}, marks={marks}"
    return None

# --- Physics: Electromagnetism ---
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
        return f"Physics: mode=circuit, topology={topo}, V={_fmt(V)}, R=[{Rtxt}], labels={'true' if labels else 'false'}"
    return None

# --- Chemistry ---
def render_chem_form(subcat: str) -> Optional[str]:
    if subcat not in ["Organic","Inorganic","Biochemistry"]:
        return None
    st.markdown("##### Chemistry — Benzene / Raw molecule")
    mode = st.radio("Mode", ["Benzene (substituted)", "Raw molecule"], horizontal=True, key="chem_mode")

    if mode == "Raw molecule":
        mol = st.text_input("chemfig molecule", "CH3-CH2-OH", key="chem_raw")
        if st.button("Generate molecule", key="chem_btn_raw"):
            return f"Chem: mode=raw, mol={mol}"
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
        return f"Chem: mode=benzene, subs={subs_part}"
    return None

# -----------------------
# Category definitions
# -----------------------
def category_structure() -> Dict[str, List[str]]:
    return {
        "Mathematics": ["Vector Analysis","Calculus","Algebra","Other"],
        "Statistics": ["Descriptive","Inference","Regression"],
        "Physics": ["Classical Mechanics","Electromagnetism","Quantum Mechanics"],
        "Chemistry": ["Organic","Inorganic","Biochemistry"],
    }

# -----------------------
# Main UI
# -----------------------
def category_ui(role: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    tabs = st.tabs(["🧮 Mathematics","📊 Statistics","🧲 Physics","🧪 Chemistry"])
    struct = category_structure()

    for tab, label in zip(tabs, struct.keys()):
        with tab:
            st.subheader(label)
            sub = st.selectbox("Subcategory", struct[label], key=f"sub_{label}")

            # Hook different forms
            if label == "Statistics":
                form_prompt = render_stats_form(sub)
                if form_prompt: return form_prompt, "Statistics", sub
            elif label == "Physics":
                if sub == "Classical Mechanics":
                    form_prompt = render_physics_form(sub)
                    if form_prompt: return form_prompt, "Physics", sub
                elif sub == "Electromagnetism":
                    form_prompt = render_em_form(sub)
                    if form_prompt: return form_prompt, "Physics", sub
            elif label == "Chemistry":
                form_prompt = render_chem_form(sub)
                if form_prompt: return form_prompt, "Chemistry", sub

            # Fallback: text area
            ta_val = st.text_area("Instruction", key=f"ta_{label}_{sub}")
            go = st.button(f"Generate {label}", key=f"go_{label}_{sub}")
            if go: return ta_val, label, sub
    return None, None, None

# -----------------------
# Main page
# -----------------------
def main_page():
    header_bar()
    st.markdown("### Generate TikZ/LaTeX from natural language")

    prompt, category, subcategory = category_ui("free")
    if prompt and category:
        try:
            result: GenerationResult = generate_document(prompt, category, subcategory)
            st.success(f"Generated successfully in {category} · {subcategory}")
            st.code(result.latex, language="latex")
            c1, c2 = st.columns([1,2])
            with c1:
                copy_button(result.latex, key="copy_tex_main")
            with c2:
                b = io.BytesIO(result.latex.encode("utf-8"))
                st.download_button("⬇️ Download .tex", data=b, file_name="scientikz_output.tex", mime="text/plain")
        except Exception as e:
            st.error(f"Error: {e}")

# -----------------------
# Router
# -----------------------
def router():
    if st.session_state["page"] == "home":
        main_page()

router()
