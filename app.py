"""ScienTikZ — precise, compile-checked TikZ diagrams for maths & physics.

Run:  streamlit run app.py
"""
import hashlib
import os

import streamlit as st

from diagrams import DIAGRAMS, GROUPS, find
from engine import ExprError
from render import RenderError, render

st.set_page_config(page_title="ScienTikZ · 学术级 TikZ 绘图", page_icon="📐", layout="wide")


def setting(name, default=""):
    try:
        return st.secrets[name]
    except Exception:
        return os.environ.get(name, default)


def licence_ok(key: str) -> bool:
    hashes = setting("LICENSE_HASHES", [])
    if isinstance(hashes, str):
        hashes = [h.strip() for h in hashes.split(",") if h.strip()]
    return bool(key) and hashlib.sha256(key.strip().upper().encode()).hexdigest() in hashes


def on_search():
    d, values = find(st.session_state.query)
    if d is None:
        st.session_state.search_msg = "没有找到匹配的图形 / No matching diagram"
        return
    st.session_state.search_msg = ""
    st.session_state.diagram = d.id
    for k, v in values.items():
        st.session_state[f"{d.id}:{k}"] = v


def widget(d, p):
    key = f"{d.id}:{p.key}"
    if key not in st.session_state:
        st.session_state[key] = p.default
    if p.kind == "float":
        return st.slider(p.label, float(p.lo), float(p.hi), step=float(p.step), key=key)
    if p.kind == "int":
        return st.slider(p.label, int(p.lo), int(p.hi), step=int(p.step), key=key)
    if p.kind == "bool":
        return st.toggle(p.label, key=key)
    if p.kind == "choice":
        return st.radio(p.label, p.options, horizontal=True, key=key)
    help_ = "可用 / allowed: x, pi, e, + - * / ^, sin cos tan exp ln sqrt abs" if p.kind == "expr" else "可写 pi，如 -2pi"
    return st.text_input(p.label, key=key, help=help_)


# keep parameter values when switching diagrams (Streamlit drops state of widgets not drawn this run)
for k in [k for k in st.session_state if ":" in str(k)]:
    st.session_state[k] = st.session_state[k]

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("## 📐 ScienTikZ")
    st.caption("参数化 · 真实编译 · 可直接放进论文和讲义\n\nPrecise, compile-checked TikZ for maths & physics")
    st.text_input("🔍 搜索 / Search", key="query", on_change=on_search,
                  placeholder="斜面 30度 · y = x^3 - 3x · 单位根 n=8")
    if st.session_state.get("search_msg"):
        st.warning(st.session_state.search_msg)
    group = st.radio("学科 / Subject", ("全部 All",) + GROUPS, horizontal=True)
    ids = [k for k, d in DIAGRAMS.items() if group == "全部 All" or d.group == group]
    if st.session_state.get("diagram") not in ids:
        st.session_state.diagram = ids[0]
    st.radio("图形 / Diagram", ids, key="diagram",
             format_func=lambda k: f"{DIAGRAMS[k].zh} · {DIAGRAMS[k].en}" + ("" if DIAGRAMS[k].free else "  🔒"))
    st.divider()
    key = st.text_input("授权码 / Licence key", type="password", placeholder="STZ-XXXX-XXXX-XXXX")
    pro = licence_ok(key)
    if pro:
        st.success("Pro 已解锁 / unlocked ✓")
    elif key:
        st.error("授权码无效 / Invalid key")
    buy = setting("BUY_URL")
    st.caption("🔒 图形：免费预览 + PNG；源码、.tex、PDF、SVG 需 Pro。" + (f" [获取授权码 / Get Pro]({buy})" if buy else ""))

# ------------------------------------------------------------------ main
d = DIAGRAMS[st.session_state.diagram]
st.markdown(f"### {d.zh} &nbsp;·&nbsp; <span style='color:#888'>{d.en}</span>", unsafe_allow_html=True)
left, right = st.columns([1, 2], gap="large")
with left:
    values = {p.key: widget(d, p) for p in d.params}

try:
    code, doc = d.code(values), d.document(values)
    out = render(doc)
except (ExprError, RenderError) as err:
    right.error(str(err))
    st.stop()

right.image(out["png"], use_container_width=True)
unlocked = d.free or pro
st.markdown("#### TikZ 代码 / Code")
if unlocked:
    st.code(code, language="latex")
else:
    st.code("\n".join(code.splitlines()[:6]) + "\n% ……", language="latex")
    st.info("🔒 完整源码与 .tex / PDF / SVG 下载需要 Pro 授权码。PNG 可免费下载。")

cols = st.columns(4)
name = f"scientikz-{d.id}"
cols[0].download_button("⬇ PNG", out["png"], f"{name}.png", "image/png", use_container_width=True)
for col, (label, data, ext, mime) in zip(cols[1:], [("⬇ .tex", doc, "tex", "text/x-tex"),
                                                   ("⬇ PDF", out["pdf"], "pdf", "application/pdf"),
                                                   ("⬇ SVG", out["svg"], "svg", "image/svg+xml")]):
    col.download_button(label, data, f"{name}.{ext}", mime, disabled=not unlocked, use_container_width=True)

st.caption("ScienTikZ v1.0 · 每张图都由 pdflatex 真实编译生成，代码可直接粘贴到 Overleaf（需 \\usepackage{tikz}）。")
