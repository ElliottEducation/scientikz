import streamlit as st
import httpx
from supabase import create_client, Client

from tikz_generator import (
    generate_document,
    GenerationResult,
    _fmt,
)

# -------------------------------
# Streamlit Configuration
# -------------------------------
st.set_page_config(
    page_title="ScienTikZ",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------
# Title and Header
# -------------------------------
st.title("ScienTikZ")
st.caption("Generate TikZ/LaTeX from natural language or structured directives")

# -------------------------------
# Tabs for different disciplines
# -------------------------------
tabs = st.tabs(["Mathematics", "Statistics", "Physics", "Chemistry"])

disciplines = {
    "Mathematics": ["Algebra", "Vector Analysis", "Calculus"],
    "Statistics": ["Probability", "Regression", "Distributions"],
    "Physics": ["Classical Mechanics", "Electromagnetism", "Quantum Mechanics"],
    "Chemistry": ["Organic", "Inorganic", "Molecules"],
}

# -------------------------------
# Render tab content
# -------------------------------
for idx, (disc_name, subcats) in enumerate(disciplines.items()):
    with tabs[idx]:
        st.subheader(disc_name)

        # Subcategory dropdown
        subcat = st.selectbox("Subcategory", subcats)

        # Natural Language Mode toggle
        nl_mode = st.toggle("Natural language mode (English)", value=True)

        # Input area
        instruction = st.text_area(
            "Instruction",
            placeholder=f"Describe the {disc_name.lower()} diagram or plot to generate...",
            height=120
        )

        # Generate Button
        if st.button(f"Generate in {disc_name} · {subcat}"):
            if not instruction.strip():
                st.warning("Please enter an instruction before generating.")
            else:
                st.write("Generating LaTeX/TikZ code...")

                # Use the new generator function
                result = generate_document(
                    prompt=instruction,
                    category_hint=disc_name,
                    subcategory_hint=subcat
                )

                if isinstance(result, GenerationResult):
                    st.success("Generation completed!")
                    st.code(result.latex, language="latex")

                    # Copy or download
                    st.download_button(
                        label="Download .tex file",
                        data=result.latex,
                        file_name=f"{disc_name}_{subcat}.tex",
                        mime="text/plain"
                    )
                else:
                    st.error("Failed to generate TikZ code.")
