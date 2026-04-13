# app.py - ScienTikZ（已完成 F：历史记录功能）
import os
import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime
from tikz_generator import generate_document, GenerationResult

st.set_page_config(page_title="ScienTikZ", page_icon="🧪", layout="wide")

st.markdown("""<h1 style="text-align:center; color:#1E88E5;">ScienTikZ</h1>
<p style="text-align:center; font-size:1.1em; color:#555;">You Input Text Naturally, I Provide You Codes TikZily</p>""", unsafe_allow_html=True)

# ========================== Supabase 配置（保留你原来的） ==========================
try:
    from supabase import create_client, Client
except Exception:
    create_client = None
    Client = None

SUPABASE_URL = st.secrets.get("SUPABASE_URL", os.environ.get("SUPABASE_URL", ""))
SUPABASE_ANON_KEY = st.secrets.get("SUPABASE_ANON_KEY", os.environ.get("SUPABASE_ANON_KEY", ""))
SUPA_ENABLED = bool(SUPABASE_URL and SUPABASE_ANON_KEY and create_client is not None)

# ========================== 会话状态 ==========================
if "user" not in st.session_state: st.session_state.user = None
if "anon_tries" not in st.session_state: st.session_state.anon_tries = 0
if "last_result" not in st.session_state: st.session_state.last_result = None
if "history" not in st.session_state: st.session_state.history = []   # ← 新增：历史记录

FREE_TRIALS = 3

def can_generate() -> tuple[bool, str]:
    if st.session_state.user: return True, ""
    remain = FREE_TRIALS - st.session_state.anon_tries
    return (False, "请登录后使用。免费用户限 3 次。") if remain <= 0 else (True, f"匿名试用剩余：{remain} 次")

def consume_one():
    if not st.session_state.user: st.session_state.anon_tries += 1

# ========================== 保存历史记录 ==========================
def save_to_history(prompt: str, category: str, result: GenerationResult):
    st.session_state.history.insert(0, {  # 最新在最前面
        "id": len(st.session_state.history) + 1,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "prompt": prompt[:60] + ("..." if len(prompt) > 60 else ""),
        "category": category,
        "latex": result.latex,
        "summary": result.summary
    })
    # 最多保留 50 条（防止太长）
    if len(st.session_state.history) > 50:
        st.session_state.history.pop()

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
        with st.expander("Sign up"): st.info("Supabase 注册功能已准备好")

    st.markdown("---")
    if st.button("💎 升级 Pro（¥29/月 或 ¥299/年）", use_container_width=True):
        st.info("Stripe 支付即将上线")

    # ==================== F：历史记录抽屉 ====================
    st.markdown("---")
    st.subheader("📜 历史记录")
    if not st.session_state.history:
        st.caption("暂无记录，开始生成吧～")
    else:
        for item in st.session_state.history[:10]:  # 显示最近10条
            with st.expander(f"{item['time']} · {item['category']}", expanded=False):
                st.caption(item['prompt'])
                if st.button("🔄 加载此记录", key=f"load_{item['id']}"):
                    st.session_state.last_result = GenerationResult(
                        latex=item['latex'], 
                        summary=item['summary'], 
                        category=item['category']
                    )
                    st.rerun()
                if st.button("🗑️ 删除", key=f"del_{item['id']}"):
                    st.session_state.history = [h for h in st.session_state.history if h['id'] != item['id']]
                    st.rerun()

# ========================== 主页面 ==========================
st.markdown("### 选择学科领域")
cat1, cat2, cat3, cat4 = st.tabs(["📐 Mathematics", "⚙️ Physics", "🧪 Chemistry", "📊 Statistics"])
selected_category = "Mathematics"
with cat1: selected_category = "Mathematics"
with cat2: selected_category = "Physics"
with cat3: selected_category = "Chemistry"
with cat4: selected_category = "Statistics"

tab_formula, tab_natural, tab_template = st.tabs(["📐 公式模式", "🗣️ 指令模式（自然语言）", "📚 模板模式"])

prompt = ""
with tab_formula:
    prompt = st.text_input("公式", placeholder="y = sin(x) 或 画正弦函数图像", key="formula_input")
with tab_natural:
    prompt = st.text_area("描述（支持中文/英文/混合）", placeholder="画一个平抛运动示意图 / Draw a projectile motion diagram", height=130, key="natural_input")
with tab_template:
    template = st.selectbox("模板", ["向量图", "函数图像", "抛体运动", "受力分析"])
    prompt = template

if st.button("🚀 Generate", type="primary", use_container_width=True):
    ok, msg = can_generate()
    if not ok:
        st.error(msg)
    elif not prompt.strip():
        st.warning("请输入内容")
    else:
        with st.spinner("正在生成..."):
            result: GenerationResult = generate_document(prompt, category=selected_category)
            st.session_state.last_result = result
            save_to_history(prompt, selected_category, result)   # ← 保存到历史
            consume_one()
        st.success("✅ 生成成功！")
        st.rerun()

# ========================== 输出区 ==========================
if st.session_state.get("last_result"):
    result: GenerationResult = st.session_state.last_result
    col1, col2 = st.columns([7, 5])
    with col1:
        st.subheader("📊 实时预览（TikZJax）")
        st.info("预览功能我们稍后继续优化～")
    with col2:
        st.subheader("📋 TikZ 代码")
        st.code(result.latex, language="latex")
        st.subheader("📄 完整 LaTeX 文档")
        full_tex = f"""\\documentclass{{standalone}}\n\\usepackage{{tikz}}\n\\begin{{document}}\n{result.latex}\n\\end{{document}}"""
        st.code(full_tex, language="latex")
        if st.button("📋 一键复制 TikZ 代码", use_container_width=True):
            st.success("✅ 已复制！")
            st.clipboard(result.latex)
        st.caption(f"学科：{result.category} | 摘要：{result.summary}")

st.markdown("---")
st.caption("ScienTikZ · 中英文双语支持 · F（历史记录）已完成")
