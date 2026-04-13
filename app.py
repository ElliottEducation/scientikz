# app.py
import os
from io import BytesIO
import streamlit as st
import streamlit.components.v1 as components

# 可选依赖：Supabase
try:
    from supabase import create_client, Client
except Exception:
    create_client = None
    Client = None

# ========================== 导入升级后的 tikz_generator ==========================
from tikz_generator import (
    generate_document,
    GenerationResult,
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

st.markdown(
    """
    <h1 style="text-align:center; color:#1E88E5;">ScienTikZ</h1>
    <p style="text-align:center; font-size:1.1em; color:#555;">
        You Input Text Naturally, I Provide You Codes TikZily
    </p>
    """,
    unsafe_allow_html=True,
)

# -------------------------------
# TikZJax 实时渲染脚本（B）
# -------------------------------
TIKZ_JAX_HTML = """
<div style="border:2px solid #f0f0f0; padding:20px; background:#fff; min-height:480px; border-radius:8px;">
    <script src="https://tikzjax.com/tikzjax.js"></script>
</div>
"""

# -------------------------------
# Supabase 配置（保持原样）
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
# 会话状态
# -------------------------------
if "user" not in st.session_state:
    st.session_state.user = None
if "anon_tries" not in st.session_state:
    st.session_state.anon_tries = 0
if "last_result" not in st.session_state:
    st.session_state.last_result = None

FREE_TRIALS = 3

# -------------------------------
# Supabase 认证函数（保持原样，略微精简）
# -------------------------------
def sb_sign_up(email: str, password: str) -> tuple[bool, str]:
    if not SUPA_ENABLED or not supabase:
        return False, "Supabase not configured."
    try:
        res = supabase.auth.sign_up({"email": email, "password": password})
        return True, "Sign-up success." if getattr(res, "user", None) else "Sign-up failed."
    except Exception as e:
        return False, f"Error: {e}"

def sb_sign_in(email: str, password: str) -> tuple[bool, str, dict | None]:
    if not SUPA_ENABLED or not supabase:
        return False, "Supabase not configured.", None
    try:
        res = supabase.auth.sign_in_with_password({"email": email, "password": password})
        user = getattr(res, "user", None)
        return (True, "Signed in.", {"id": user.id, "email": user.email}) if user else (False, "Invalid credentials.", None)
    except Exception as e:
        return False, f"Error: {e}", None

def sb_sign_out():
    if SUPA_ENABLED and supabase:
        try:
            supabase.auth.sign_out()
        except Exception:
            pass

def sb_init_quota(user_id: str):
    if not (SUPA_ENABLED and supabase and user_id):
        return
    try:
        supabase.table("usage_quota").upsert(
            {"user_id": user_id, "plan": "free", "free_quota": FREE_TRIALS, "used_count": 0}
        ).execute()
    except Exception:
        pass

def sb_get_remaining(user_id: str) -> int | None:
    if not (SUPA_ENABLED and supabase and user_id):
        return None
    try:
        r = supabase.table("usage_quota").select("free_quota,used_count").eq("user_id", user_id).single().execute()
        d = getattr(r, "data", {}) or {}
        return int(d.get("free_quota", FREE_TRIALS)) - int(d.get("used_count", 0))
    except Exception:
        return None

def sb_consume_one(user_id: str) -> bool:
    if not (SUPA_ENABLED and supabase and user_id):
        return False
    try:
        r = supabase.table("usage_quota").select("free_quota,used_count").eq("user_id", user_id).single().execute()
        d = getattr(r, "data", {}) or {}
        used = int(d.get("used_count", 0))
        fq = int(d.get("free_quota", FREE_TRIALS))
        if used >= fq:
            return False
        supabase.table("usage_quota").update({"used_count": used + 1}).eq("user_id", user_id).execute()
        return True
    except Exception:
        return False

# -------------------------------
# 额度检查
# -------------------------------
def can_generate() -> tuple[bool, str]:
    if st.session_state.user:
        rem = sb_get_remaining(st.session_state.user.get("id", ""))
        if rem is None or rem > 0:
            return True, ""
        return False, "Your free quota has been used. Please upgrade to Pro."
    remain = FREE_TRIALS - st.session_state.anon_tries
    if remain <= 0:
        return False, "Please sign in. Free plan includes 3 trial runs."
    return True, f"Anonymous trials left: {remain}"

def consume_one():
    if st.session_state.user:
        sb_consume_one(st.session_state.user.get("id", ""))
    else:
        st.session_state.anon_tries += 1

# -------------------------------
# 侧边栏（Account）
# -------------------------------
with st.sidebar:
    st.markdown("### Account")
    if not SUPA_ENABLED:
        st.info("Supabase 未配置 → 离线模式（3次免费试用）")
    if st.session_state.user:
        u = st.session_state.user
        st.success(f"✅ 已登录：**{u.get('email','')}**")
        rem = sb_get_remaining(u.get("id", ""))
        if rem is not None:
            st.write(f"剩余次数：**{rem}**")
        if st.button("Sign out"):
            sb_sign_out()
            st.session_state.user = None
            st.rerun()
    else:
        with st.expander("Sign in"):
            si_email = st.text_input("Email", key="si_email")
            si_pwd = st.text_input("Password", type="password", key="si_pwd")
            if st.button("Sign in"):
                ok, msg, user = sb_sign_in(si_email, si_pwd)
                if ok:
                    st.session_state.user = user
                    sb_init_quota(user["id"])
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
        with st.expander("Sign up"):
            su_email = st.text_input("Email", key="su_email")
            su_pwd = st.text_input("Password", type="password", key="su_pwd")
            if st.button("Create account"):
                ok, msg = sb_sign_up(su_email, su_pwd)
                st.success(msg) if ok else st.error(msg)

    st.markdown("---")
    if st.button("💎 升级 Pro（¥29/月 或 ¥299/年）"):
        st.info("Stripe 支付页面即将弹出（已在代码中准备好）")

# -------------------------------
# 主页面 Tabs（C）
# -------------------------------
tab1, tab2, tab3 = st.tabs(["📐 公式模式", "🗣️ 指令模式（自然语言）", "📚 模板模式"])

prompt = ""

with tab1:
    st.info("直接输入数学公式，例如：y = sin(x) 或 r = 3cos(2θ)")
    formula_input = st.text_input("公式", placeholder="y = x^2", key="formula_input")
    prompt = formula_input

with tab2:
    st.info("用自然语言描述，例如：画一个向量 (3,2) 并显示分量")
    natural_input = st.text_area("描述", placeholder="画一个二次函数 y=x² 的图像，标注顶点", key="natural_input", height=120)
    prompt = natural_input

with tab3:
    st.info("选择模板（后续可继续扩展）")
    template_choice = st.selectbox("模板", ["向量图", "函数图像", "抛体运动"])
    template_param = st.text_input("参数", "向量 (3,2)")
    prompt = f"{template_choice} {template_param}"

# -------------------------------
# 生成按钮
# -------------------------------
if st.button("🚀 Generate in Mathematics", type="primary", use_container_width=True):
    ok, msg = can_generate()
    if not ok:
        st.error(msg)
    elif not prompt.strip():
        st.warning("请输入内容")
    else:
        with st.spinner("AI 正在生成 TikZ..."):
            result: GenerationResult = generate_document(prompt, category="Mathematics")
            st.session_state.last_result = result
            consume_one()
        st.success("✅ 生成成功！")
        st.rerun()

# -------------------------------
# 输出区（B + 实时预览）
# -------------------------------
if st.session_state.last_result:
    result: GenerationResult = st.session_state.last_result
    
    col1, col2 = st.columns([7, 5])
    
    with col1:
        st.subheader("📊 实时预览（TikZJax）")
        components.html(
            f"""
            <div style="border:2px solid #e0e0e0; padding:25px; background:#ffffff; border-radius:10px;">
                <script type="text/tikz">{result.latex}</script>
            </div>
            {TIKZ_JAX_HTML}
            """,
            height=550,
            scrolling=False
        )
    
    with col2:
        st.subheader("📋 TikZ 代码")
        st.code(result.latex, language="latex")
        
        st.subheader("📄 完整 LaTeX 文档")
        full_tex = f"""\\documentclass{{standalone}}
\\usepackage{{tikz}}
\\begin{{document}}
{result.latex}
\\end{{document}}"""
        st.code(full_tex, language="latex")
        
        if st.button("📋 一键复制 TikZ 代码"):
            st.success("已复制到剪贴板！")
            st.clipboard(result.latex)

        st.caption(f"分类：{result.category} | 摘要：{result.summary}")

# -------------------------------
# 底部提示
# -------------------------------
st.markdown("---")
st.caption("ScienTikZ MVP v1.1 · Powered by GraphSpec + TikZJax · Offline mode 已关闭")
