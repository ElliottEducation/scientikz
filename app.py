import os
import streamlit as st

# 可选依赖：若没连 Supabase，也能离线运行
try:
    from supabase import create_client, Client  # pip install supabase
except Exception:
    create_client = None
    Client = None

from tikz_generator import (
    generate_document,
    GenerationResult,
    _fmt,
)

# -------------------------------
# Streamlit 基础设置
# -------------------------------
st.set_page_config(
    page_title="ScienTikZ",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("ScienTikZ")
st.caption("You Input Text Naturally, I Provide You Codes TikZily")

# -------------------------------
# Supabase 客户端（可选）
# -------------------------------
SUPABASE_URL = st.secrets.get("SUPABASE_URL", os.environ.get("SUPABASE_URL", ""))
SUPABASE_ANON_KEY = st.secrets.get("SUPABASE_ANON_KEY", os.environ.get("SUPABASE_ANON_KEY", ""))

SUPA_ENABLED = bool(SUPABASE_URL and SUPABASE_ANON_KEY and create_client is not None)
supabase: Client | None = None
if SUPA_ENABLED:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    except Exception:
        SUPA_ENABLED = False
        supabase = None

# -------------------------------
# 会话状态：用户与额度
# -------------------------------
if "user" not in st.session_state:
    st.session_state.user = None  # {"id": "...", "email": "..."}
if "anon_tries" not in st.session_state:
    st.session_state.anon_tries = 0

FREE_TRIALS = 3  # 未登录配额；登录用户默认 free_quota 也是 3（表里可调）

# -------------------------------
# Supabase 认证/额度函数（带兜底）
# -------------------------------
def sb_sign_up(email: str, password: str) -> tuple[bool, str]:
    if not SUPA_ENABLED or not supabase:
        return False, "Supabase not configured."
    try:
        res = supabase.auth.sign_up({"email": email, "password": password})
        if res and getattr(res, "user", None):
            return True, "Sign-up success. Please verify mail if required."
        return False, "Sign-up failed."
    except Exception as e:
        return False, f"Sign-up error: {e}"

def sb_sign_in(email: str, password: str) -> tuple[bool, str, dict | None]:
    if not SUPA_ENABLED or not supabase:
        return False, "Supabase not configured.", None
    try:
        res = supabase.auth.sign_in_with_password({"email": email, "password": password})
        user = getattr(res, "user", None)
        if user:
            return True, "Signed in.", {"id": user.id, "email": user.email}
        return False, "Invalid credentials.", None
    except Exception as e:
        return False, f"Sign-in error: {e}", None

def sb_sign_out() -> None:
    if SUPA_ENABLED and supabase:
        try:
            supabase.auth.sign_out()
        except Exception:
            pass

def sb_init_quota(user_id: str) -> None:
    """确保 usage_quota 里有一条记录；若表不存在也不会抛出到页面。"""
    if not (SUPA_ENABLED and supabase and user_id):
        return
    try:
        row = (
            supabase.table("usage_quota")
            .select("*")
            .eq("user_id", user_id)
            .single()
            .execute()
        )
        # 如果有记录，直接返回
        if getattr(row, "data", None):
            return
    except Exception:
        # 可能是表不存在或 single() 报错，尝试 upsert
        pass
    try:
        supabase.table("usage_quota").upsert(
            {"user_id": user_id, "plan": "free", "free_quota": FREE_TRIALS, "used_count": 0}
        ).execute()
    except Exception:
        pass  # 表还没建也不会影响页面

def sb_get_remaining(user_id: str) -> int | None:
    if not (SUPA_ENABLED and supabase and user_id):
        return None
    try:
        r = (
            supabase.table("usage_quota")
            .select("free_quota, used_count")
            .eq("user_id", user_id)
            .single()
            .execute()
        )
        d = getattr(r, "data", None) or {}
        return int(d.get("free_quota", FREE_TRIALS)) - int(d.get("used_count", 0))
    except Exception:
        return None

def sb_consume_one(user_id: str) -> bool:
    if not (SUPA_ENABLED and supabase and user_id):
        return False
    try:
        # 乐观更新：先读后写
        r = (
            supabase.table("usage_quota")
            .select("free_quota, used_count")
            .eq("user_id", user_id)
            .single()
            .execute()
        )
        d = getattr(r, "data", None) or {}
        used = int(d.get("used_count", 0))
        fq = int(d.get("free_quota", FREE_TRIALS))
        if used >= fq:
            return False
        supabase.table("usage_quota").update({"used_count": used + 1}).eq("user_id", user_id).execute()
        return True
    except Exception:
        return False

# -------------------------------
# 侧边栏：账号区
# -------------------------------
with st.sidebar:
    st.markdown("### Account")
    if not SUPA_ENABLED:
        st.info("Supabase env not set. Running without login (3 free trials).")
    if st.session_state.user:
        u = st.session_state.user
        st.success(f"Logged in as **{u.get('email','')}**")
        rem = sb_get_remaining(u.get("id", "")) if SUPA_ENABLED else None
        if rem is not None:
            st.write(f"Remaining runs: **{rem}**")
        if st.button("Sign out", key="sb_signout_btn"):
            sb_sign_out()
            st.session_state.user = None
            st.rerun()
    else:
        with st.expander("Sign in"):
            si_email = st.text_input("Email", key="sb_si_email")
            si_pwd = st.text_input("Password", type="password", key="sb_si_pwd")
            if st.button("Sign in", key="sb_si_btn"):
                ok, msg, user = sb_sign_in(si_email, si_pwd)
                if ok and user:
                    st.session_state.user = user
                    sb_init_quota(user["id"])
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
        with st.expander("Sign up"):
            su_email = st.text_input("Email (new)", key="sb_su_email")
            su_pwd = st.text_input("Password (new)", type="password", key="sb_su_pwd")
            if st.button("Create account", key="sb_su_btn"):
                ok, msg = sb_sign_up(su_email, su_pwd)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

# -------------------------------
# 额度检查函数
# -------------------------------
def can_generate() -> tuple[bool, str]:
    """返回 (是否允许, 提示语)。"""
    # 登录用户：查 Supabase 表
    if st.session_state.user:
        user_id = st.session_state.user.get("id", "")
        rem = sb_get_remaining(user_id)
        if rem is None:
            # 表未建立或查询失败，允许但提示
            return True, "⚠️ Quota table not found; allowing generation."
        if rem <= 0:
            return False, "Your free quota has been used. Please upgrade plan."
        # 还有额度
        return True, ""
    # 未登录：本地 3 次
    remain = FREE_TRIALS - int(st.session_state.anon_tries or 0)
    if remain <= 0:
        return False, "Please sign in to continue. Free plan includes 3 trial runs."
    return True, f"Anonymous trials left: {remain}"

def consume_one() -> None:
    """成功生成后调用，消耗一次额度。"""
    if st.session_state.user:
        sb_consume_one(st.session_state.user.get("id", ""))
    else:
        st.session_state.anon_tries = int(st.session_state.anon_tries or 0) + 1

# -------------------------------
# 主体：Tabs + 表单
# -------------------------------
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

        # 额度提示
        allowed, note = can_generate()
        if note:
            st.info(note)

        btn = st.button(
            f"Generate in {disc_name} · {subcat}",
            key=f"{key_prefix}_generate_btn",
            disabled=not allowed,
        )

        if btn:
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
                    consume_one()
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
