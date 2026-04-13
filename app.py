# app.py - ScienTikZ 完整版（支持中英文双语 + 实时预览）
import os
import streamlit as st
import streamlit.components.v1 as components
from tikz_generator import generate_document, GenerationResult

# ========================== Streamlit 配置 ==========================
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

# ========================== Supabase 配置（保持你原来的设置） ==========================
try:
    from supabase import create_client, Client
except Exception:
    create_client = None
    Client = None

SUPABASE_URL = st.secrets.get("SUPABASE_URL", os.environ.get("SUPABASE_URL", ""))
SUPABASE_ANON_KEY = st.secrets.get("SUPABASE_ANON_KEY", os.environ.get("SUPABASE_ANON_KEY", ""))

SUPA_ENABLED = bool(SUPABASE_URL and SUPABASE_ANON_KEY and create_client is not None)
supabase: Client | None = None
if SUPA_ENABLED:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    except Exception:
        SUPA_ENABLED = False

# ========================== 会话状态 ==========================
if "user" not in st.session_state:
    st.session_state.user = None
if "anon_tries" not in st.session_state:
    st.session_state.anon_tries = 0
if "last_result" not in st.session_state:
    st.session_state.last_result = None

FREE_TRIALS = 3

# ========================== Supabase 辅助函数（简洁版） ==========================
def can_generate() -> tuple[bool, str]:
    if st.session_state.user:
        # 这里你可以后续扩展 Pro 用户无限次数
        return True, ""
    remain = FREE_TRIALS - st.session_state.anon_tries
    if remain <= 0:
        return False, "请登录后使用。免费用户每日限 3 次。"
    return True, f"匿名试用剩余：{remain} 次"

def consume_one():
    if not st.session_state.user:
        st.session_state.anon_tries += 1

# ========================== 侧边栏 ==========================
with st.sidebar:
    st.markdown("### Account")
    if st.session_state.user:
        st.success(f"✅ 已登录：{st.session_state.user.get('email', '')}")
        if st.button("退出登录"):
            st.session_state.user = None
            st.rerun()
    else:
        with st.expander("Sign in"):
            email = st.text_input("Email", key="login_email")
            pwd = st.text_input("Password", type="password", key="login_pwd")
            if st.button("登录"):
                # 简化版（你可自行补充完整 sb_sign_in 函数）
                st.session_state.user = {"id": "demo", "email": email}
                st.success("登录成功（演示模式）")
                st.rerun()
        with st.expander("Sign up"):
            st.info("Supabase 注册功能已准备好，可自行补全")

    st.markdown("---")
    if st.button("💎 升级 Pro（¥29/月 或 ¥299/年）", use_container_width=True):
        st.info("Stripe 支付即将上线")

# ========================== 主页面：学科分类 ==========================
st.markdown("### 选择学科领域")
cat1, cat2, cat3, cat4 = st.tabs(["📐 Mathematics", "⚙️ Physics", "🧪 Chemistry", "📊 Statistics"])

selected_category = "Mathematics"
with cat1: selected_category = "Mathematics"
with cat2: selected_category = "Physics"
with cat3: selected_category = "Chemistry"
with cat4: selected_category = "Statistics"

# ========================== 输入模式 ==========================
tab_formula, tab_natural, tab_template = st.tabs([
    "📐 公式模式", 
    "🗣️ 指令模式（自然语言）", 
    "📚 模板模式"
])

prompt = ""
with tab_formula:
    st.info("直接输入公式（支持中英文）")
    prompt = st.text_input("公式", placeholder="y = sin(x) 或 画正弦函数图像", key="formula_input")

with tab_natural:
    st.info("用自然语言描述（支持中文 / 英文 / 混合）")
    prompt = st.text_area(
        "描述", 
        placeholder="画一个平抛运动示意图 / Draw a projectile motion diagram / 画 vector (3,2)",
        height=130,
        key="natural_input"
    )

with tab_template:
    st.info("选择模板")
    template = st.selectbox("模板", ["向量图", "函数图像", "抛体运动", "受力分析"])
    prompt = template

# ========================== 生成按钮 ==========================
if st.button("🚀 Generate", type="primary", use_container_width=True):
    ok, msg = can_generate()
    if not ok:
        st.error(msg)
    elif not prompt.strip():
        st.warning("请输入内容")
    else:
        with st.spinner("AI 正在生成 TikZ..."):
            result: GenerationResult = generate_document(prompt, category=selected_category)
            st.session_state.last_result = result
            consume_one()
        st.success("✅ 生成成功！")
        st.rerun()

# ========================== 输出区（实时预览 + 代码） ==========================
if st.session_state.get("last_result"):
    result: GenerationResult = st.session_state.last_result
    
    col1, col2 = st.columns([7, 5])
    
    with col1:
        st.subheader("📊 实时预览（TikZJax）")
        components.html(
            f"""
            <script src="https://tikzjax.com/tikzjax.js"></script>
            <div style="border:2px solid #e0e0e0; padding:30px; background:#ffffff; border-radius:12px; min-height:520px; box-shadow:0 4px 15px rgba(0,0,0,0.08); display:flex; align-items:center; justify-content:center;">
                <script type="text/tikz">{result.latex}</script>
            </div>
            """,
            height=560,
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
        
        if st.button("📋 一键复制 TikZ 代码", use_container_width=True):
            st.success("✅ 已复制到剪贴板！")
            st.clipboard(result.latex)
        
        st.caption(f"学科：{result.category} | 摘要：{result.summary}")

st.markdown("---")
st.caption("ScienTikZ · 支持中英文双语 · Powered by GraphSpec + TikZJax")
