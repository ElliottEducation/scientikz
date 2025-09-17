# app.py
# Streamlit + Supabase MVP for scientikz (NLP → TikZ/LaTeX generator)
# Tech parity with Math2TikZ: Supabase auth, role gating (Free/Pro), history logging
# English-only comments, standard quotes and hash symbols

import os
import io
import time
import streamlit as st
from typing import Optional, Dict
from supabase import create_client, Client
from tikz_generator import generate_document, detect_category, GenerationResult

# -----------------------
# App config
# -----------------------
APP_NAME = os.getenv("SCIENTIKZ_APP_NAME", "scientikz")
st.set_page_config(page_title=f"{APP_NAME} · NLP → TikZ/LaTeX", layout="wide")

# -----------------------
# Supabase client
# -----------------------
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
if not SUPABASE_URL or not SUPABASE_KEY:
    st.warning("Supabase URL/KEY not set. Set SUPABASE_URL and SUPABASE_KEY in environment.")

supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# -----------------------
# Session state helpers
# -----------------------
def init_state():
    if "auth" not in st.session_state:
        st.session_state["auth"] = {
            "user": None,     # {"id":..., "email":...}
            "role": "free",   # free|pro|admin
            "access_token": None
        }
    if "page" not in st.session_state:
        st.session_state["page"] = "home"  # or "login", "signup"
    if "prompt" not in st.session_state:
        st.session_state["prompt"] = "Plot y = sin(x) from -6.28 to 6.28"

init_state()

# -----------------------
# Supabase auth flows
# -----------------------
def sign_up(email: str, password: str) -> Dict:
    assert supabase is not None
    res = supabase.auth.sign_up({"email": email, "password": password})
    user = res.user
    return {"user": user}

def sign_in(email: str, password: str) -> Dict:
    assert supabase is not None
    res = supabase.auth.sign_in_with_password({"email": email, "password": password})
    session = res.session
    user = res.user
    return {"user": user, "session": session}

def sign_out():
    if supabase:
        try:
            supabase.auth.sign_out()
        except Exception:
            pass
    st.session_state["auth"] = {"user": None, "role": "free", "access_token": None}

# -----------------------
# Profiles/roles
# -----------------------
def ensure_profile(user_id: str, email: str) -> str:
    # Upsert user profile; return role
    assert supabase is not None
    # Try select
    sel = supabase.table("profiles").select("*").eq("id", user_id).execute()
    if sel.data:
        role = sel.data[0].get("role", "free") or "free"
        return role
    # Insert new
    ins = supabase.table("profiles").insert({"id": user_id, "email": email, "role": "free"}).execute()
    return "free"

def get_user_role(user_id: str) -> str:
    assert supabase is not None
    sel = supabase.table("profiles").select("role").eq("id", user_id).single().execute()
    if sel.data and "role" in sel.data:
        return sel.data["role"] or "free"
    return "free"

# -----------------------
# History logging
# -----------------------
def log_generation(user_id: str, prompt: str, category: str, summary: str):
    if not supabase or not user_id:
        return
    try:
        supabase.table("generation_logs").insert({
            "user_id": user_id,
            "prompt": prompt,
            "category": category,
            "summary": summary
        }).execute()
    except Exception:
        pass

# -----------------------
# UI Components
# -----------------------
def auth_box():
    st.subheader("Sign in")
    with st.form("signin_form", clear_on_submit=False):
        em = st.text_input("Email", key="login_email")
        pw = st.text_input("Password", type="password", key="login_password")
        do_login = st.form_submit_button("Sign in")
    if do_login and supabase:
        try:
            res = sign_in(em, pw)
            user = res.get("user")
            session = res.get("session")
            if user and session:
                st.session_state["auth"]["user"] = {"id": user.id, "email": user.email}
                st.session_state["auth"]["access_token"] = session.access_token
                role = ensure_profile(user.id, user.email or "")
                st.session_state["auth"]["role"] = role
                st.success("Signed in successfully.")
                time.sleep(0.5)
                st.session_state["page"] = "home"
                st.rerun()
        except Exception as e:
            st.error(f"Sign in failed: {e}")

    st.markdown("---")
    st.subheader("Sign up")
    with st.form("signup_form", clear_on_submit=False):
        em2 = st.text_input("Email (new account)", key="signup_email")
        pw2 = st.text_input("Password (min 6 chars)", type="password", key="signup_password")
        do_signup = st.form_submit_button("Create account")
    if do_signup and supabase:
        try:
            sign_up(em2, pw2)
            st.success("Sign-up successful. Please check your email if confirmation is required, then sign in.")
        except Exception as e:
            st.error(f"Sign up failed: {e}")

def header_bar():
    left, right = st.columns([3,2])
    with left:
        st.title(APP_NAME)
        st.caption("Natural language → TikZ/PGFPlots/Circuitikz/Chemfig (LaTeX)")
    with right:
        auth = st.session_state["auth"]
        if auth["user"]:
            st.markdown(f"**{auth['user']['email']}**  · Role: `{auth['role']}`")
            if st.button("Sign out"):
                sign_out()
                st.rerun()
        else:
            if st.button("Sign in / Sign up"):
                st.session_state["page"] = "login"
                st.rerun()

def pro_gate():
    stripe_url = os.getenv("SCIENTIKZ_STRIPE_URL", "")
    st.markdown("### Upgrade to Pro")
    st.write(
        "- Free: core generators (Math/Physics/Stats/Chem basic), shorter prompts\n"
        "- Pro: longer prompts, more templates (parametric/polar, optics, regression, benzene, etc.), priority updates"
    )
    if stripe_url:
        st.link_button("Upgrade — Go Pro", stripe_url)
    else:
        st.info("Set SCIENTIKZ_STRIPE_URL env to enable the upgrade button.")

# -----------------------
# Main page (generation UI)
# -----------------------
def examples_panel():
    st.markdown("#### Examples")
    cols = st.columns(4)
    examples = {
        "Math": [
            "Plot y = sin(x) from -6.28 to 6.28",
            "Plot y = exp(-x^2) from -3 to 3",
            "Draw vector v = (3,2)"
        ],
        "Physics": [
            "Inclined plane: angle=30 deg, block mass m, show forces mg, N, friction",
            "DC circuit: series R=220 ohm, V=9 V battery"
        ],
        "Statistics": [
            "Scatter: x=[1,2,3,4], y=[2,3,3.5,5]",
            "Normal curve: mu=0, sigma=1 from -4 to 4"
        ],
        "Chemistry": [
            "Molecule: H2O",
            "Molecule: CH3-CH2-OH"
        ]
    }
    for i, cat in enumerate(examples.keys()):
        with cols[i]:
            st.write(f"**{cat}**")
            for ex in examples[cat]:
                if st.button(f"Use: {ex}", key=f"ex_{cat}_{ex}"):
                    st.session_state["prompt"] = ex
                    st.rerun()

def main_page():
    header_bar()

    # Role banner
    role = st.session_state["auth"]["role"]
    if role == "free":
        with st.expander("Pro features"):
            pro_gate()

    # Prompt + category
    c1, c2 = st.columns([2,1])
    with c1:
        prompt = st.text_area(
            "Instruction",
            value=st.session_state["prompt"],
            height=160,
            placeholder="e.g., 'Plot y = sin(x) from -6.28 to 6.28' or 'Molecule: CH3-CH2-OH'"
        )
    with c2:
        autodetected = detect_category(prompt)
        override = st.selectbox(
            "Category (auto-detected, can override):",
            options=["Auto", "Math", "Physics", "Statistics", "Chemistry"],
            index=0
        )
        category = autodetected if override == "Auto" else override
        st.write(f"Auto-detected: **{autodetected}**")

        # Free token-length limit sample (keep parity with Math2TikZ style)
        if role == "free" and len(prompt) > 280:
            st.warning("Free tier prompt too long. Please shorten it or upgrade to Pro.")
    st.session_state["prompt"] = prompt

    st.divider()
    examples_panel()
    st.divider()

    # Generate
    go = st.button("Generate TikZ/LaTeX")
    if go:
        try:
            result: GenerationResult = generate_document(prompt, category)
            st.success("Generated successfully.")
            st.code(result.latex, language="latex")

            # Download .tex
            b = io.BytesIO(result.latex.encode("utf-8"))
            st.download_button(
                label="Download .tex",
                data=b,
                file_name="scientikz_output.tex",
                mime="text/plain"
            )

            # Log history
            user = st.session_state["auth"]["user"]
            if user:
                log_generation(user["id"], prompt, result.category, result.summary)

            st.markdown(
                "#### Compile tips\n"
                "- Requires LaTeX packages: tikz, pgfplots, circuitikz, chemfig, siunitx\n"
                "- Compile: `pdflatex --shell-escape scientikz_output.tex`\n"
            )
        except Exception as e:
            st.error(f"Error: {e}")

# -----------------------
# Router
# -----------------------
def router():
    if st.session_state["page"] == "login":
        header_bar()
        auth_box()
    else:
        main_page()

router()
