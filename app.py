import streamlit as st
import streamlit.components.v1 as components
import time
from tikz_generator import generate_document

st.set_page_config(page_title="ScienTikZ | 专业科学绘图", layout="wide", page_icon="📐")

# --- Day 1 Strategy Navigation ---
DOMAINS = {
    "Mathematics": ["Analysis & Calculus", "Vectors & Geometry", "Trigonometric Functions", "Complex Numbers & Polar", "Probability & Statistics"],
    "Physics": ["Mechanics", "Modern Physics", "Electromagnetism"],
    "Astronomy": ["Astrophysics", "Celestial Mechanics"],
    "Chemistry": ["Organic Chemistry", "Structural Chemistry"]
}

TIPS = {
    "Analysis & Calculus": "Try: 'Area under x^2 from 0 to 2' (Calculus Demo)",
    "Vectors & Geometry": "Try: 'Vector addition of u and v' (Vector Demo)",
    "Complex Numbers & Polar": "Try: 'Draw the 8th roots of unity'"
}

# --- Monetization Logic ---
FREE_LIMIT = 3
if 'usage' not in st.session_state: st.session_state.usage = 0

with st.sidebar:
    st.title("📐 ScienTikZ Engine")
    st.info("🎯 **Day 1 Goal:** Perfecting Functions & Vectors")
    
    domain = st.selectbox("Domain", list(DOMAINS.keys()))
    sub_cat = st.selectbox(f"{domain} Category", DOMAINS[domain])
    
    st.markdown("---")
    st.header("Account")
    remaining = FREE_LIMIT - st.session_state.usage
    if remaining > 0:
        st.success(f"Free: {remaining} / {FREE_LIMIT}")
    else:
        st.error("Quota reached. Contact for Pro.")
    
    scale_factor = st.slider("Scale", 0.5, 2.0, 1.2, 0.1)

st.title("Professional Academic Diagramming")
st.info(f"💡 **Expert Tip:** {TIPS.get(sub_cat, 'Input in English or Chinese...')}")

# Preset Buttons for Day 1
if sub_cat in ["Analysis & Calculus", "Vectors & Geometry"]:
    cols = st.columns(3)
    if sub_cat == "Analysis & Calculus":
        if cols[0].button("Area under x^2"): st.session_state.inp = "Area under x^2 from 0 to 2"
    else:
        if cols[0].button("Vector Sum"): st.session_state.inp = "Vector addition of u and v"

prompt = st.text_area("Request", value=st.session_state.get('inp', ''), height=100)

def tikz_preview(code):
    html = f"""<script src="https://tikzjax.com/v1/tikzjax.js"></script>
    <div style="background:white; display:flex; justify-content:center; align-items:center; min-height:450px; padding:20px; border-radius:12px; border:1px solid #ddd;">
        <script type="text/tikz">{code}</script>
    </div>"""
    components.html(html, height=500)

if st.button("🚀 Render TikZ", type="primary", disabled=st.session_state.usage >= FREE_LIMIT):
    if prompt:
        with st.spinner("Processing Math Logic..."):
            res = generate_document(prompt, domain, sub_cat, scale_factor)
            st.session_state.res = res
            st.session_state.usage += 1
            st.rerun()

if "res" in st.session_state:
    res = st.session_state.res
    c1, c2 = st.columns([6, 4])
    with c1:
        st.subheader("Live Preview")
        tikz_preview(res.latex)
    with c2:
        st.subheader("TikZ Code")
        st.code(res.latex, language="latex")
        full_tex = f"\\documentclass[tikz,border=2mm]{{standalone}}\n\\usepackage{{amsmath,amssymb}}\n\\begin{{document}}\n{res.latex}\n\\end{{document}}"
        st.download_button("Download .tex", full_tex, file_name="scientikz.tex", use_container_width=True)

st.markdown("---")
st.caption("ScienTikZ Alpha v0.3 | Focused on Mathematical Accuracy")
