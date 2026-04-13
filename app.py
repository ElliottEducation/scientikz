# app.py - ScienTikZ 完整版（中英文双语 + 新标签页预览已解决）
import os
import base64
import streamlit as st
import streamlit.components.v1 as components
from tikz_generator import generate_document, GenerationResult

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

# ========================== Supabase 配置 ==========================
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

def can_generate() -> tuple[bool, str]:
    if st.session_state.user:
        return True, ""
    remain = FREE_TRIALS - st.session_state.anon_tries
    if remain <= 0:
        return False, "请登录后使用。免费用户限 3 次。"
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
                st.session_state.user = {"id": "demo", "email": email}
                st.success("登录成功（演示模式）")
                st.rerun()
        with st.expander("Sign up"):
            st.info("Supabase 注册功能已准备好")

    st.markdown("---")
    if st.button("💎 升级 Pro（¥29/月 或 ¥299/年）", use_container_width=True):
        st.info("Stripe 支付即将上线")

# ========================== 学科分类 ==========================
st.markdown("### 选择学科领域")
cat1, cat2, cat3, cat4 = st.tabs(["📐 Mathematics", "⚙️ Physics", "🧪 Chemistry", "📊 Statistics"])
selected_category = "Mathematics"
with cat1: selected_category = "Mathematics"
with cat2: selected_category = "Physics"
with cat3: selected_category = "Chemistry"
with cat4: selected_category = "Statistics"

# ========================== 输入模式 ==========================
tab_formula, tab_natural, tab_template = st.tabs(["📐 公式模式", "🗣️ 指令模式（自然语言）", "📚 模板模式"])

prompt = ""
with tab_formula:
    prompt = st.text_input("公式", placeholder="y = sin(x) 或 画正弦函数图像", key="formula_input")

with tab_natural:
    prompt = st.text_area(
        "描述（支持中文/英文/混合）", 
        placeholder="画一个平抛运动示意图 / Draw a projectile motion diagram / 画 vector (3,2)",
        height=130,
        key="natural_input"
    )

with tab_template:
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
        with st.spinner("正在生成 TikZ..."):
            result: GenerationResult = generate_document(prompt, category=selected_category)
            st.session_state.last_result = result
            consume_one()
        st.success("✅ 生成成功！")
        st.rerun()

# ========================== 输出区（关键：新标签页预览） ==========================
if st.session_state.get("last_result"):
    result: GenerationResult = st.session_state.last_result
    
    col1, col2 = st.columns([7, 5])
    
    with col1:
        st.subheader("📊 实时预览（TikZJax）")
        
        # 生成独立 HTML（在新标签页打开）
        standalone_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>ScienTikZ 预览</title>
    <script src="https://tikzjax.com/tikzjax.js"></script>
    <style>
        body {{ margin: 40px; background: white; font-family: system-ui; }}
        .container {{ max-width: 800px; margin: 0 auto; border: 1px solid #ddd; padding: 40px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); }}
    </style>
</head>
<body>
    <div class="container">
        <h2 style="text-align:center; color:#1E88E5;">ScienTikZ 预览</h2>
        <script type="text/tikz">{result.latex}</script>
    </div>
</body>
</html>
"""
        b64 = base64.b64encode(standalone_html.encode("utf-8")).decode("utf-8")
        preview_url = f"data:text/html;base64,{b64}"
        
        st.markdown(f"""
        <a href="{preview_url}" target="_blank" style="display:block; text-align:center; background:#1E88E5; color:white; padding:12px; border-radius:8px; text-decoration:none; font-weight:bold;">
            🔗 在新标签页打开实时预览（推荐）
        </a>
        """, unsafe_allow_html=True)
        
        st.caption("（如果还是空白，请在新标签页中按 Ctrl+Shift+R 硬刷新）")
    
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
st.caption("ScienTikZ · 中英文双语支持 · Powered by GraphSpec + TikZJax")
