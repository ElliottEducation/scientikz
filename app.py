import streamlit as st
import streamlit.components.v1 as components
import time
from tikz_generator import generate_document

# --- Page Config ---
st.set_page_config(page_title="ScienTikZ | 科学作图引擎", layout="wide", page_icon="📐")

# --- Usage Limit (Monetization Logic) ---
FREE_LIMIT = 3
if 'usage_count' not in st.session_state:
    st.session_state.usage_count = 0

def has_quota():
    return st.session_state.usage_count < FREE_LIMIT

# --- Taxonomy Definition ---
DOMAINS = {
    "Mathematics": ["Complex Plane & Polar", "Analysis & Calculus", "Trigonometric Functions", "Linear Algebra", "Geometry"],
    "Physics": ["Mechanics", "Modern Physics", "Electromagnetism", "Optics & Thermodynamics"],
    "Astronomy": ["Celestial Mechanics", "Astrophysics"],
    "Chemistry": ["Organic Chemistry", "Structural Chemistry"]
}

TIPS = {
    "Complex Plane & Polar": "Try: 'Draw the 8th roots of unity' (Auto-detects numbers!)",
    "Trigonometric Functions": "Try: 'Show cos(x) wave'",
    "Mechanics": "Try: 'Draw an object on a 45 degree inclined plane'"
}

PRESETS = {
    "Complex Plane & Polar": ["5th Roots of Unity", "8th Roots of Unity"],
    "Trigonometric Functions": ["Sine Wave", "Cosine Wave"],
    "Mechanics": ["30 Degree Inclined Plane", "45 Degree Inclined Plane"]
}

# --- Sidebar Navigation ---
with st.sidebar:
    st.title("📐 ScienTikZ Engine")
    st.markdown("Professional LaTeX diagram generator.")
    
    st.header("1. Domain Settings")
    domain = st.selectbox("Select Domain", list(DOMAINS.keys()))
    sub_cat = st.selectbox(f"Select Category", DOMAINS[domain])
    
    st.markdown("---")
    st.header("2. Account Status")
    remaining = FREE_LIMIT - st.session_state.usage_count
    
    if remaining > 0:
        st.success(f"**Free Trial:** {remaining} / {FREE_LIMIT} remaining")
        st.progress(remaining / FREE_LIMIT)
    else:
        st.error("🔒 **Free Quota Exhausted**")
        st.info("Upgrade to **ScienTikZ Pro** for unlimited generation, higher resolution exports, and premium math templates. Contact Elliott Coaching for access.")
    
    st.markdown("---")
    with st.expander("⚙️ Advanced Settings"):
        scale_factor = st.slider("Global Scale (Zoom)", min_value=0.5, max_value=2.0, value=1.2, step=0.1)

# --- Main App ---
st.title("AI Scientific Diagram Generation")

# Prompt Guidance
current_tip = TIPS.get(sub_cat, "Describe your academic diagram in English or Chinese...")
st.info(f"💡 **Smart Parsing Active:** {current_tip}")

# Quick Presets Buttons
if sub_cat in PRESETS:
    st.write("**Quick Generators:**")
    cols = st.columns(len(PRESETS[sub_cat]))
    for idx, preset in enumerate(PRESETS[sub_cat]):
        if cols[idx].button(preset, use_container_width=True, disabled=not has_quota()):
            st.session_state.input_val = f"Generate {preset}"

# Input Area
user_input = st.text_area("Your Request", 
                         value=st.session_state.get('input_val', ''),
                         placeholder="e.g., 画一个包含支持力和摩擦力的45度斜面受力图", 
                         height=100)

def tikz_preview(code):
    html = f"""<script src="https://tikzjax.com/v1/tikzjax.js"></script>
    <div style="background:white; display:flex; justify-content:center; align-items:center; min-height:450px; padding:20px; border-radius:8px; border:1px solid #e0e0e0;">
        <script type="text/tikz">{code}</script>
    </div>"""
    components.html(html, height=500)

can_gen = has_quota()

if st.button("🚀 Render TikZ Code", type="primary", use_container_width=True, disabled=not can_gen):
    if user_input:
        progress_text = "Translating natural language to TikZ logic..."
        my_bar = st.progress(0, text=progress_text)
        for percent_complete in range(100):
            time.sleep(0.005)
            my_bar.progress(percent_complete + 1, text=progress_text)
        
        res = generate_document(user_input, domain, sub_cat, scale=scale_factor)
        st.session_state.current_res = res
        st.session_state.usage_count += 1
        my_bar.empty()
        st.toast('Render Complete!', icon='✅')
        st.rerun() # Refresh to update quota

if not can_gen:
    st.warning("⚠️ Daily limit reached. Please upgrade to Pro.")

# Display Results
if "current_res" in st.session_state:
    res = st.session_state.current_res
    c1, c2 = st.columns([6, 4])
    
    with c1:
        st.subheader("🖼️ Live Preview")
        tikz_preview(res.latex)
    
    with c2:
        st.subheader("💻 TikZ Source Code")
        st.code(res.latex, language="latex")
        
        full_tex = f"\\documentclass[tikz,border=2mm]{{standalone}}\n\\usepackage{{amsmath,amssymb}}\n\\begin{{document}}\n{res.latex}\n\\end{{document}}"
        st.download_button("⬇️ Download full .tex file", full_tex, file_name="scientikz_export.tex", use_container_width=True)
        st.caption("✨ Tip: Hover over the code block to use the native 'Copy' button.")
