# app.py
# scientikz · Streamlit + (optional) Supabase
# Full UI with 4 main categories and subcategory selectors, copy + download actions
# English-only comments and standard quotes/hash as requested

import os
import io
import time
import json
from typing import Optional, Dict, Tuple, List

import streamlit as st
from streamlit.components.v1 import html

try:
    from supabase import create_client, Client  # optional; UI degrades gracefully if not configured
except Exception:
    create_client = None
    Client = None  # type: ignore

from tikz_generator import (
    generate_document,
    detect_category,
    GenerationResult,
)

# -----------------------
# App config
# -----------------------
APP_NAME = os.getenv("SCIENTIKZ_APP_NAME", "scientikz")
st.set_page_config(page_title=f"{APP_NAME} · NLP → TikZ/LaTeX", layout="wide")

# Minimal CSS polish
st.markdown(
    """
    <style>
      .stCodeBlock { font-size: 0.92rem; }
      div[data-baseweb="tab-list"] { gap: 6px; }
      div[data-baseweb="tab"] { border-radius: 10px; }
      .pill { padding: 6px 10px; border: 1px solid #d0d7de; border-radius: 999px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------
# Supabase client (optional)
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
# Session state
# -----------------------
def init_state():
    if "auth" not in st.session_state:
        st.session_state["auth"] = {
            "user": None,     # {"id":..., "email":...}
            "role": "free",   # "free" | "pro" | "admin"
            "access_token": None,
        }
    if "page" not in st.session_state:
        st.session_state["page"] = "home"

init_state()

# -----------------------
# Auth helpers (only if supabase is configured)
# -----------------------
def sign_up(email: str, password: str) -> Dict:
    assert supabase is not None
    res = supabase.auth.sign_up({"email": email, "password": password})
    return {"user": res.user}

def sign_in(email: str, password: str) -> Dict:
    assert supabase is not None
    res = supabase.auth.sign_in_with_password({"email": email, "password": password})
    return {"user": res.user, "session": res.session}

def sign_out():
    if supabase:
        try:
            supabase.auth.sign_out()
        except Exception:
            pass
    st.session_state["auth"] = {"user": None, "role": "free", "access_token": None}

def ensure_profile(user_id: str, email: str) -> str:
    if not supabase:
        return "free"
    sel = supabase.table("profiles").select("*").eq("id", user_id).execute()
    if sel.data:
        return sel.data[0].get("role", "free") or "free"
    supabase.table("profiles").insert({"id": user_id, "email": email, "role": "free"}).execute()
    return "free"

def log_generation(user_id: str, prompt: str, category: str, summary: str, subcategory: str = ""):
    if not supabase or not user_id:
        return
    try:
        supabase.table("generation_logs").insert({
            "user_id": user_id,
            "prompt": prompt,
            "category": f"{category}:{subcategory}" if subcategory else category,
            "summary": summary
        }).execute()
    except Exception:
        pass

# -----------------------
# UI helpers
# -----------------------
def header_bar():
    left, right = st.columns([3, 2])
    with left:
        st.title(APP_NAME)
        st.caption("Natural language → TikZ/PGFPlots/Circuitikz/Chemfig (LaTeX)")
    with right:
        auth = st.session_state["auth"]
        if supabase:
            if auth["user"]:
                st.markdown(f"**{auth['user']['email']}** · Role: `{auth['role']}`")
                if st.button("Sign out"):
                    sign_out()
                    st.rerun()
            else:
                if st.button("Sign in / Sign up"):
                    st.session_state["page"] = "login"
                    st.rerun()
        else:
            st.markdown('<span class="pill">Offline mode</span> (no Supabase configured)', unsafe_allow_html=True)

def auth_box():
    st.subheader("Sign in")
    with st.form("signin_form"):
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
                time.sleep(0.4)
                st.session_state["page"] = "home"
                st.rerun()
        except Exception as e:
            st.error(f"Sign in failed: {e}")

    st.markdown("---")
    st.subheader("Sign up")
    with st.form("signup_form"):
        em2 = st.text_input("Email (new account)", key="signup_email")
        pw2 = st.text_input("Password (min 6 chars)", type="password", key="signup_password")
        do_signup = st.form_submit_button("Create account")
    if do_signup and supabase:
        try:
            sign_up(em2, pw2)
            st.success("Sign-up successful. Check your inbox if confirmation is required, then sign in.")
        except Exception as e:
            st.error(f"Sign up failed: {e}")

def pro_gate():
    stripe_url = os.getenv("SCIENTIKZ_STRIPE_URL", "")
    st.markdown("### Upgrade to Pro")
    st.write(
        "- Free: core generators, basic prompts\n"
        "- Pro: longer prompts, more templates (polar/parametric, optics, regression, benzene, etc.)"
    )
    if stripe_url:
        st.link_button("Upgrade — Go Pro", stripe_url)
    else:
        st.info("Set SCIENTIKZ_STRIPE_URL to enable the upgrade button.")

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
# Category/subcategory structure and examples
# -----------------------
def category_structure() -> Dict[str, List[str]]:
    return {
        "Mathematics": ["Calculus", "Geometry", "Algebra", "Other"],
        "Statistics": ["Descriptive", "Inference", "Regression"],
        "Physics": ["Classical Mechanics", "Electromagnetism", "Quantum Mechanics"],
        "Chemistry": ["Organic", "Inorganic", "Biochemistry"],
    }

def examples_by_subcategory() -> Dict[str, Dict[str, List[str]]]:
    return {
        "Mathematics": {
            "Calculus": [
                "Plot y = sin(x) from -6.28 to 6.28",
                "Plot y = exp(-x^2) from -3 to 3",
            ],
            "Geometry": [
                "Draw vector v = (3,2)",
            ],
            "Algebra": [
                "Plot y = x^2 - 2*x + 1 from -3 to 3",
            ],
            "Other": [
                "Plot y = cos(x) from -6.28 to 6.28",
            ],
        },
        "Statistics": {
            "Descriptive": [
                "Scatter: x=[1,2,3,4], y=[2,3,3.5,5]",
            ],
            "Inference": [
                "Normal curve: mu=0, sigma=1 from -4 to 4",
            ],
            "Regression": [
                "Scatter: x=[1,2,3,4,5], y=[1.8,2.2,2.9,3.8,5.1]",
            ],
        },
        "Physics": {
            "Classical Mechanics": [
                "Inclined plane: angle=30 deg, block mass m, show forces mg, N, friction",
            ],
            "Electromagnetism": [
                "DC circuit: series R=220 ohm, V=9 V battery",
            ],
            "Quantum Mechanics": [
                "Show a generic force vector diagram",
            ],
        },
        "Chemistry": {
            "Organic": [
                "Molecule: CH3-CH2-OH",
            ],
            "Inorganic": [
                "Molecule: H2O",
            ],
            "Biochemistry": [
                "Molecule: CH3-CH(NH2)-COOH",
            ],
        },
    }

# -----------------------
# Tabbed category UI
# -----------------------
def category_ui(role: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    tabs = st.tabs(["🧮 Mathematics", "📊 Statistics", "🧲 Physics", "🧪 Chemistry"])
    struct = category_structure()
    exmap = examples_by_subcategory()

    default_prompts = {
        "Mathematics": "Plot y = sin(x) from -6.28 to 6.28",
        "Statistics": "Scatter: x=[1,2,3,4], y=[2,3,3.5,5]",
        "Physics": "Inclined plane: angle=30 deg, block mass m, show forces mg, N, friction",
        "Chemistry": "Molecule: H2O",
    }
    if "tab_prompts" not in st.session_state:
        st.session_state["tab_prompts"] = default_prompts.copy()
    if "tab_subcat" not in st.session_state:
        st.session_state["tab_subcat"] = {
            "Mathematics": "Calculus",
            "Statistics": "Descriptive",
            "Physics": "Classical Mechanics",
            "Chemistry": "Inorganic",
        }

    def render_one(tab, label):
        with tab:
            st.subheader(label)
            sub = st.selectbox(
                "Subcategory",
                struct[label],
                index=struct[label].index(st.session_state["tab_subcat"][label]),
                key=f"sub_{label}",
            )
            st.session_state["tab_subcat"][label] = sub

            ta_val = st.text_area(
                "Instruction",
                value=st.session_state["tab_prompts"][label],
                height=160,
                key=f"ta_{label}",
            )
            st.session_state["tab_prompts"][label] = ta_val

            # Free tier length reminder
            if role == "free" and len(ta_val) > 280:
                st.warning("Free tier prompt too long. Please shorten it or upgrade to Pro.")

            # Examples for current subcategory
            with st.expander("Examples"):
                cols = st.columns(2)
                for i, ex in enumerate(exmap[label].get(sub, [])):
                    with cols[i % 2]:
                        if st.button(f"Use: {ex}", key=f"ex_{label}_{sub}_{i}"):
                            st.session_state["tab_prompts"][label] = ex
                            st.rerun()

            go = st.button(f"Generate in {label} · {sub}", key=f"go_{label}_{sub}")
            if go:
                # Map UI main labels to generator category tokens
                cat_map = {
                    "Mathematics": "Math",
                    "Statistics": "Statistics",
                    "Physics": "Physics",
                    "Chemistry": "Chemistry",
                }
                return ta_val, cat_map[label], sub
        return None, None, None

    # Order: Mathematics, Statistics, Physics, Chemistry
    for tab, label in zip(tabs, ["Mathematics", "Statistics", "Physics", "Chemistry"]):
        p, c, s = render_one(tab, label)
        if p and c and s:
            return p, c, s

    return None, None, None

# -----------------------
# Main page
# -----------------------
def main_page():
    header_bar()

    role = st.session_state["auth"]["role"]
    if role == "free":
        with st.expander("Pro features"):
            pro_gate()

    st.markdown("### Choose a discipline and subcategory, then generate LaTeX/TikZ")
    prompt, category, subcategory = category_ui(role)

    if prompt and category and subcategory:
        try:
            result: GenerationResult = generate_document(prompt, category, subcategory)
            st.success(f"Generated successfully in {category} · {subcategory}")
            st.code(result.latex, language="latex")

            c1, c2 = st.columns([1, 2])
            with c1:
                copy_button(result.latex, key="copy_tex_main")
            with c2:
                b = io.BytesIO(result.latex.encode("utf-8"))
                st.download_button(
                    label="⬇️ Download .tex",
                    data=b,
                    file_name="scientikz_output.tex",
                    mime="text/plain",
                )

            user = st.session_state["auth"]["user"]
            if user:
                log_generation(user["id"], prompt, result.category, result.summary, subcategory)

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
        if supabase:
            auth_box()
        else:
            st.info("Supabase is not configured. To enable sign-in, set SUPABASE_URL and SUPABASE_KEY.")
    else:
        main_page()

router()
