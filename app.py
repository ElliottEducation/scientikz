import streamlit as st
from supabase import create_client, Client  
from tikz_generator import (
    generate_document,
    GenerationResult,
    _fmt,
)

st.set_page_config(
    page_title="ScienTikZ",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("ScienTikZ")
st.caption("Generate TikZ/LaTeX from natural language or structured directives")

disciplines = {
    "Mathematics": ["Algebra", "Vector Analysis", "Calculus"],
    "Statistics": ["Probability", "Regression", "Distributions"],
    "Physics": ["Classical Mechanics", "Electromagnetism", "Quantum Mechanics"],
    "Chemistry": ["Organic", "Inorganic", "Molecules"],
}

tabs = st.tabs(list(disciplines.keys()))

for idx, (disc_name, subcats) in enumerate(disciplines.items()):
    key_prefix = disc_name.lower().replace(" ", "_")  

    with tabs[idx]:
        st.subheader(disc_name)

        subcat = st.selectbox(
            "Subcategory",
            subcats,
            key=f"{key_prefix}_subcat",
        )

        nl_mode = st.toggle(
            "Natural language mode (English)",
            value=True,
            key=f"{key_prefix}_nl_toggle",
        )

        instruction = st.text_area(
            "Instruction",
            placeholder=f"Describe the {disc_name.lower()} diagram or plot to generate...",
            height=120,
            key=f"{key_prefix}_instruction",
        )

        if st.button(
            f"Generate in {disc_name} · {subcat}",
            key=f"{key_prefix}_generate_btn",
        ):
            if not instruction.strip():
                st.warning("Please enter an instruction before generating.")
            else:
                with st.spinner("Generating LaTeX/TikZ code..."):
                    result = generate_document(
                        prompt=instruction,
                        category_hint=disc_name,
                        subcategory_hint=subcat,
                    )

                if isinstance(result, GenerationResult):
                    st.success("Generation completed!")
                    st.code(result.latex, language="latex")

                    st.download_button(
                        label="Download .tex file",
                        data=result.latex,
                        file_name=f"{disc_name}_{subcat}.tex",
                        mime="text/plain",
                        key=f"{key_prefix}_download_btn",
                    )
                else:
                    st.error("Failed to generate TikZ code.")
