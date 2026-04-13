import streamlit as st
import streamlit.components.v1 as components
from tikz_generator import generate_document

st.set_page_config(page_title="ScienTikZ Pro", page_icon="🧪", layout="wide")

# ====================== 安全处理 Secrets ======================
def get_supabase_config():
    try:
        return st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_ANON_KEY"]
    except Exception:
        return None, None

URL, KEY = get_supabase_config()

# ====================== TikZJax 预览组件 ======================
def tikz_preview_component(tikz_code: str):
    """
    在网页中实时渲染 TikZ 代码
    """
    html_content = f"""
    <script src="https://tikzjax.com/v1/tikzjax.js"></script>
    <div style="display: flex; justify-content: center; align-items: center; min-height: 400px; background: #ffffff; border: 1px solid #eee; border-radius: 12px; box-shadow: inset 0 0 10px rgba(0,0,0,0.05);">
        <script type="text/tikz">
            {tikz_code}
        </script>
    </div>
    """
    components.html(html_content, height=480)

# ====================== UI 布局 ======================
st.title("🧪 ScienTikZ: AI 驱动的科学绘图")
st.markdown("---")

with st.sidebar:
    st.header("⚙️ 配置")
    selected_category = st.selectbox("学科领域", ["Mathematics", "Physics", "Chemistry"])
    
    if not URL:
        st.info("💡 提示：本地运行模式（匿名）")
    else:
        st.success("🔐 已连接到 Supabase")

# 输入区
prompt = st.text_area("描述你想生成的图形 (例如：画一个受力分析图，或者正弦函数)", 
                     placeholder="输入自然语言描述...", height=120)

if st.button("🚀 开始生成", type="primary", use_container_width=True):
    if prompt.strip():
        with st.spinner("AI 正在解析绘图逻辑并应用 TikZ 模板..."):
            result = generate_document(prompt, selected_category)
            st.session_state.current_res = result
    else:
        st.warning("请先输入绘图描述内容")

# 结果展示
if "current_res" in st.session_state:
    res = st.session_state.current_res
    
    col_viz, col_code = st.columns([6, 4])
    
    with col_viz:
        st.subheader("🖼️ 图形预览 (TikZJax)")
        tikz_preview_component(res.latex)
    
    with col_code:
        st.subheader("💻 TikZ 代码")
        st.code(res.latex, language="latex")
        
        # 导出选项
        with st.expander("📄 完整 LaTeX 源码"):
            full_tex = f"""\\documentclass[tikz, border=2mm]{{standalone}}
\\usepackage{{amsmath, amssymb}}
\\begin{{document}}
{res.latex}
\\end{{document}}"""
            st.code(full_tex, language="latex")
        
        st.download_button("⬇️ 下载 .tex 文件", full_tex, file_name="scientikz_output.tex")

st.markdown("---")
st.caption("ScienTikZ Engine v2.0 | 自动矢量箭头 & 水印保护已开启")
