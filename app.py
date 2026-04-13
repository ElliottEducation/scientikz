# app.py （E 已完成版）
import streamlit as st
import streamlit.components.v1 as components
from tikz_generator import generate_document, GenerationResult
# ...（你原来的 Supabase、额度、登录代码全部保留，这里省略以节省篇幅，直接复制你之前的对应部分）

# ====================== 新增：四大分类 Tabs ======================
st.markdown("### 选择学科领域")
cat_tab1, cat_tab2, cat_tab3, cat_tab4 = st.tabs(["📐 Mathematics", "⚙️ Physics", "🧪 Chemistry", "📊 Statistics"])

selected_category = "Mathematics"
with cat_tab1:
    selected_category = "Mathematics"
with cat_tab2:
    selected_category = "Physics"
with cat_tab3:
    selected_category = "Chemistry"
with cat_tab4:
    selected_category = "Statistics"

# ====================== 原有的模式 Tabs ======================
tab1, tab2, tab3 = st.tabs(["📐 公式模式", "🗣️ 指令模式（自然语言）", "📚 模板模式"])

prompt = ""
with tab1:
    formula_input = st.text_input("公式", "y = sin(x)", key="formula_input")
    prompt = formula_input
with tab2:
    natural_input = st.text_area("描述", "画一个平抛运动示意图", key="natural_input", height=120)
    prompt = natural_input
with tab3:
    template = st.selectbox("模板", ["向量图", "函数图像", "抛体运动", "受力分析"])
    prompt = template

# ====================== 生成按钮 ======================
if st.button("🚀 Generate", type="primary", use_container_width=True):
    # ...（你原来的 can_generate + consume_one 逻辑保持不变）
    with st.spinner("生成中..."):
        result: GenerationResult = generate_document(prompt, category=selected_category)
        st.session_state.last_result = result
        # consume_one()
    st.success("生成成功！")
    st.rerun()

# ====================== 实时预览区（已修复） ======================
if st.session_state.get("last_result"):
    result = st.session_state.last_result
    col1, col2 = st.columns([7, 5])
    with col1:
        st.subheader("📊 实时预览（TikZJax）")
        components.html(
            f"""
            <div style="border:2px solid #e0e0e0; padding:25px; background:#ffffff; border-radius:10px;">
                <script type="text/tikz">{result.latex}</script>
            </div>
            """,
            height=550,
            scrolling=False
        )
    with col2:
        st.subheader("📋 TikZ 代码")
        st.code(result.latex, language="latex")
        # ...（你原来的完整 LaTeX 和复制按钮保持不变）
