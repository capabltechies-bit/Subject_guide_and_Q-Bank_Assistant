import streamlit as st
import tempfile, os
from pathlib import Path

from vector_store import get_stats, add_documents
from document_processor import process_document
from rag_engine import answer_topic, solve_question

# ══════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════
st.set_page_config(
    page_title="Scholar AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════
if "history" not in st.session_state:  st.session_state.history  = []
if "indexed" not in st.session_state:  st.session_state.indexed  = False
if "mode"    not in st.session_state:  st.session_state.mode     = "explain"
if "view"    not in st.session_state:  st.session_state.view     = "chat"

# ══════════════════════════════════════════════════════
#  GLOBAL CSS  — dark theme, all text fully visible
# ══════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=Lora:ital,wght@0,400;0,600;1,400&display=swap');

/* ─── Design Tokens ─── */
:root {
  --bg:         #0d0f14;
  --surface:    #151820;
  --surface2:   #1c2030;
  --surface3:   #222840;
  --border:     #2a3045;
  --border2:    #333d58;

  /* All text colours guaranteed visible on dark */
  --text:       #eceef5;   /* primary — near-white */
  --text2:      #a0a8c0;   /* secondary — light grey-blue */
  --text3:      #606880;   /* muted — still readable */
  --text-inv:   #0d0f14;   /* text on light/accent backgrounds */

  --accent:     #6c8ef5;
  --accent2:    #4a6de0;
  --accent-lt:  rgba(108,142,245,.14);
  --accent-glow:rgba(108,142,245,.25);

  --gold:       #f0b84a;
  --gold-lt:    rgba(240,184,74,.13);
  --sage:       #4ecb8d;
  --sage-lt:    rgba(78,203,141,.13);
  --rose:       #f07070;
  --rose-lt:    rgba(240,112,112,.12);

  --r:    11px;
  --r-lg: 18px;
  --sh:   0 2px 8px rgba(0,0,0,.4);
  --sh-lg:0 8px 32px rgba(0,0,0,.55);
}

/* ─── Base reset ─── */
*, *::before, *::after { box-sizing: border-box; }
html, body, .stApp {
  background: var(--bg) !important;
  font-family: 'Sora', sans-serif;
  color: var(--text);
}

/* ─── Hide Streamlit chrome ─── */
#MainMenu, footer, header, [data-testid="stToolbar"] { visibility:hidden; height:0; }
.block-container { padding:0 !important; max-width:100% !important; }

/* ════════════════════════════════════════
   FORCE ALL STREAMLIT TEXT VISIBLE
   (catches labels, captions, help text,
    radio options, selectbox, file uploader)
════════════════════════════════════════ */
.stApp p,
.stApp span,
.stApp label,
.stApp div,
.stApp li,
.stApp small,
.stApp caption,
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[class*="stText"],
[class*="stCaption"] {
  color: var(--text) !important;
}

/* Muted helper/caption text */
.stApp .stCaption,
.stApp [data-testid="stCaptionContainer"] p,
.stApp small { color: var(--text2) !important; }

/* ─── File uploader — all text visible ─── */
[data-testid="stFileUploader"] section,
[data-testid="stFileUploader"] section *,
[data-testid="stFileUploader"] button,
[data-testid="stFileUploader"] button span,
[data-testid="stFileUploadDropzone"] *,
[data-testid="stFileUploaderFileName"],
[data-testid="stFileUploaderFileData"],
[data-testid="stFileUploaderFileData"] *  {
  color: var(--text2) !important;
}
[data-testid="stFileUploader"] button {
  background: var(--surface3) !important;
  border: 1px solid var(--border2) !important;
  border-radius: 8px !important;
  color: var(--text) !important;
}
[data-testid="stFileUploader"] button:hover {
  background: var(--border2) !important;
}

/* ─── Text input — visible text & placeholder ─── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
  background: var(--surface2) !important;
  border: 1.5px solid var(--border2) !important;
  border-radius: 12px !important;
  padding: 14px 16px !important;
  font-size: 14px !important;
  font-family: 'Sora', sans-serif !important;
  color: var(--text) !important;
  caret-color: var(--accent) !important;
  resize: none !important;
  transition: border-color .2s, box-shadow .2s !important;
}
.stTextInput > div > div > input::placeholder,
.stTextArea > div > div > textarea::placeholder { color: var(--text3) !important; }
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px var(--accent-glow) !important;
  outline: none !important;
}
.stTextInput > div > div,
.stTextInput > div,
.stTextArea > div > div,
.stTextArea > div,
[data-baseweb="input"],
[data-baseweb="textarea"] {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}
.stTextInput label,
.stTextArea label { color: var(--text2) !important; font-size: 13px !important; }

/* ─── Selectbox ─── */
[data-baseweb="select"] > div {
  background: var(--surface2) !important;
  border: 1.5px solid var(--border2) !important;
  border-radius: 10px !important;
}
[data-baseweb="select"] span,
[data-baseweb="select"] div,
[data-baseweb="popover"] li,
[data-baseweb="popover"] span {
  color: var(--text) !important;
  background: var(--surface2) !important;
}
[data-baseweb="popover"] [aria-selected="true"] {
  background: var(--accent-lt) !important;
}

/* ─── Radio buttons ─── */
[data-testid="stRadio"] label,
[data-testid="stRadio"] span {
  color: var(--text2) !important;
}
[data-testid="stRadio"] label:has(input:checked) span {
  color: var(--text) !important;
}

/* ─── Checkbox ─── */
[data-testid="stCheckbox"] span { color: var(--text) !important; }

/* ─── Slider ─── */
[data-testid="stSlider"] label,
[data-testid="stSlider"] p { color: var(--text2) !important; }

/* ─── Progress ─── */
.stProgress > div { background: var(--surface3) !important; border-radius:99px !important; height:5px !important; }
.stProgress > div > div { background: var(--accent) !important; border-radius:99px !important; }

/* ─── Alerts / Toast ─── */
.stAlert { border-radius: 10px !important; font-family:'Sora',sans-serif !important; font-size:13px !important; }
.stAlert p, .stAlert div { color: inherit !important; }

/* ─── Expander ─── */
[data-testid="stExpander"] summary p { color: var(--text2) !important; }
[data-testid="stExpander"] { background: var(--surface2) !important; border: 1px solid var(--border) !important; border-radius: 10px !important; }

/* ─── Spinner ─── */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* ─── Divider ─── */
hr { border-color: var(--border) !important; margin: 20px 0 !important; }

/* ─── Metric ─── */
[data-testid="stMetricLabel"] p  { color: var(--text3) !important; font-size:11px !important; text-transform:uppercase; letter-spacing:.07em; }
[data-testid="stMetricValue"]    { color: var(--text) !important; font-size:28px !important; font-weight:700 !important; }

/* ════════════════════════════════════════
   SIDEBAR
════════════════════════════════════════ */
section[data-testid="stSidebar"] > div:first-child {
  background: var(--surface) !important;
  border-right: 1px solid var(--border);
  padding: 0 !important;
}
/* ensure all sidebar text is visible */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div { color: var(--text2) !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: var(--text) !important; }
section[data-testid="stSidebar"] hr  { border-color: var(--border) !important; }

/* sidebar clear button */
section[data-testid="stSidebar"] .stButton > button {
  background: var(--rose-lt) !important;
  color: var(--rose) !important;
  border: 1px solid rgba(240,112,112,.25) !important;
  border-radius: 9px !important;
  font-family: 'Sora', sans-serif !important;
  font-size: 12px !important; font-weight: 600 !important;
  padding: 9px 14px !important; width: 100% !important;
  transition: background .15s !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
  background: rgba(240,112,112,.2) !important;
}

/* ════════════════════════════════════════
   CUSTOM COMPONENTS
════════════════════════════════════════ */

/* sidebar logo */
.sb-logo {
  display:flex; align-items:center; gap:10px;
  padding: 22px 20px 18px;
  border-bottom: 1px solid var(--border);
}
.sb-logo-icon {
  width:38px; height:38px; border-radius:10px;
  background: linear-gradient(135deg,#6c8ef5,#4a6de0);
  display:flex; align-items:center; justify-content:center;
  font-size:20px; box-shadow:0 2px 10px var(--accent-glow);
}
.sb-logo-text {
  font-family:'Lora',Georgia,serif;
  font-size:18px; font-weight:600; color:var(--text) !important;
}
.sb-logo-beta {
  font-size:9px; font-weight:700; letter-spacing:.1em;
  text-transform:uppercase; color:var(--accent) !important;
  background:var(--accent-lt); border-radius:4px;
  padding:2px 6px; margin-left:4px;
}

/* sidebar stats */
.sb-stats { display:flex; gap:8px; margin:18px 20px 0; }
.sb-stat {
  flex:1; background:var(--surface2);
  border:1px solid var(--border); border-radius:10px;
  padding:12px 10px; text-align:center;
}
.sb-stat-n { font-size:26px; font-weight:700; color:var(--text) !important; line-height:1; }
.sb-stat-l {
  font-size:10px; font-weight:600; color:var(--text3) !important;
  text-transform:uppercase; letter-spacing:.08em; margin-top:4px;
}

/* status pill */
.status-pill {
  display:inline-flex; align-items:center; gap:6px;
  padding:5px 12px; border-radius:99px;
  font-size:11px; font-weight:600; margin:14px 20px 0;
}
.status-on  { background:var(--sage-lt);  color:var(--sage)  !important; border:1px solid rgba(78,203,141,.3); }
.status-off { background:var(--gold-lt);  color:var(--gold)  !important; border:1px solid rgba(240,184,74,.3); }
.status-dot { width:6px; height:6px; border-radius:50%; background:currentColor; }

/* sidebar section label */
.sb-sec {
  font-size:10px; font-weight:700; letter-spacing:.12em;
  text-transform:uppercase; color:var(--text3) !important;
  padding:18px 20px 8px; display:block;
}

/* file rows */
.sb-files { padding:0 12px; }
.sb-file-row {
  display:flex; align-items:center; gap:9px;
  padding:8px 8px; border-radius:8px;
  font-size:12px; color:var(--text2) !important;
  transition:background .12s; margin-bottom:2px;
}
.sb-file-row:hover { background:var(--surface2); }
.sb-file-icon {
  width:28px; height:28px; border-radius:7px;
  display:flex; align-items:center; justify-content:center;
  font-size:14px; flex-shrink:0;
}
.fi-pdf  { background:rgba(240,112,112,.15); }
.fi-docx { background:var(--accent-lt); }
.fi-pptx { background:var(--gold-lt); }
.fi-txt  { background:var(--sage-lt); }
.fi-gen  { background:var(--surface3); }
.sb-file-name { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; flex:1; color:var(--text2) !important; }

/* ── Top bar nav/mode buttons ── */
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg,var(--accent),var(--accent2)) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 9px !important;
  font-family: 'Sora', sans-serif !important;
  font-size: 13px !important; font-weight: 600 !important;
  padding: 9px 18px !important;
  box-shadow: 0 2px 8px var(--accent-glow) !important;
  transition: opacity .15s, transform .1s !important;
}
.stButton > button[kind="primary"]:hover { opacity:.88 !important; transform:translateY(-1px) !important; }

.stButton > button[kind="secondary"] {
  background: var(--surface2) !important;
  color: var(--text2) !important;
  border: 1px solid var(--border2) !important;
  border-radius: 9px !important;
  font-family: 'Sora', sans-serif !important;
  font-size: 13px !important; font-weight: 500 !important;
  padding: 9px 18px !important;
  transition: all .15s !important;
}
.stButton > button[kind="secondary"]:hover {
  background: var(--surface3) !important;
  color: var(--text) !important;
}

/* ── Welcome card ── */
.welcome-card {
  max-width:560px; margin:0 auto;
  text-align:center; padding:64px 24px 48px;
}
.welcome-glyph {
  width:76px; height:76px; border-radius:22px;
  background:var(--accent-lt);
  border:1px solid rgba(108,142,245,.25);
  display:flex; align-items:center; justify-content:center;
  font-size:36px; margin:0 auto 26px;
  box-shadow:0 4px 16px var(--accent-glow);
}
.welcome-h {
  font-family:'Lora',Georgia,serif;
  font-size:30px; font-weight:600;
  color:var(--text); margin:0 0 12px; line-height:1.25;
}
.welcome-h em { font-style:italic; color:var(--accent); }
.welcome-p { font-size:15px; color:var(--text2); line-height:1.7; margin:0 0 28px; }
.welcome-chips { display:flex; flex-wrap:wrap; gap:8px; justify-content:center; }
.welcome-chip {
  background:var(--surface2); border:1px solid var(--border2);
  border-radius:8px; padding:8px 14px;
  font-size:12px; color:var(--text2);
}

/* ── Chat messages ── */
.msg-row { display:flex; margin-bottom:24px; align-items:flex-start; }
.msg-row-user { justify-content:flex-end; }
.msg-row-ai   { justify-content:flex-start; }

.bubble { max-width:68%; padding:14px 18px; font-size:14px; line-height:1.75; border-radius:var(--r-lg); }

.bubble-user {
  background: linear-gradient(135deg,var(--accent),var(--accent2));
  color: #fff !important;
  border-radius: var(--r-lg) var(--r-lg) 5px var(--r-lg);
  box-shadow: 0 4px 16px var(--accent-glow);
  font-weight: 500;
}

.bubble-ai {
  background: var(--surface2);
  border: 1px solid var(--border2);
  color: var(--text) !important;
  border-radius: 5px var(--r-lg) var(--r-lg) var(--r-lg);
  box-shadow: var(--sh);
}

.avatar {
  width:34px; height:34px; border-radius:50%;
  display:flex; align-items:center; justify-content:center;
  font-size:16px; flex-shrink:0; margin-top:4px;
}
.avatar-user { background:var(--accent-lt); margin-left:10px; order:2; }
.avatar-ai   { background:linear-gradient(135deg,var(--accent),var(--accent2)); box-shadow:0 2px 8px var(--accent-glow); margin-right:10px; }

.ai-label {
  font-size:10px; font-weight:700; letter-spacing:.1em;
  text-transform:uppercase; color:var(--accent) !important; margin-bottom:6px;
}

/* source chips */
.src-bar { display:flex; flex-wrap:wrap; gap:5px; margin-top:12px; padding-top:10px; border-top:1px solid var(--border); }
.src-chip {
  background:var(--accent-lt); border:1px solid rgba(108,142,245,.25);
  border-radius:6px; padding:2px 9px;
  font-size:11px; color:var(--accent) !important; font-weight:500;
}

/* ask button */
.ask-btn-wrap .stButton > button {
  background: linear-gradient(135deg,var(--accent),var(--accent2)) !important;
  color: #fff !important; border:none !important;
  border-radius: 11px !important; padding: 12px 24px !important;
  font-family: 'Sora', sans-serif !important;
  font-size:13px !important; font-weight:700 !important;
  box-shadow: 0 3px 10px var(--accent-glow) !important;
  transition: opacity .15s, transform .1s !important;
}
.ask-btn-wrap .stButton > button:hover { opacity:.88 !important; transform:translateY(-1px) !important; }

/* ── Upload view ── */
.ucard-title {
  font-family:'Lora',Georgia,serif;
  font-size:22px; font-weight:600; color:var(--text);
  margin:0 0 6px;
}
.ucard-sub { font-size:13px; color:var(--text2); margin-bottom:20px; line-height:1.6; }

[data-testid="stFileUploader"] > div {
  background: var(--surface2) !important;
  border: 2px dashed var(--border2) !important;
  border-radius: 14px !important;
  padding: 36px 28px !important;
  transition: border-color .2s, background .2s !important;
}
[data-testid="stFileUploader"] > div:hover {
  border-color: var(--accent) !important;
  background: var(--accent-lt) !important;
}

/* file result pills */
.fpill {
  display:flex; align-items:center; gap:10px;
  background:var(--surface2); border:1px solid var(--border);
  border-radius:10px; padding:10px 14px; margin-bottom:6px;
  font-size:13px; box-shadow:var(--sh);
}
.fpill-ok  { border-color:rgba(78,203,141,.3);  background:var(--sage-lt); }
.fpill-err { border-color:rgba(240,112,112,.3); background:var(--rose-lt); }
.fpill-name { flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; color:var(--text) !important; }
.fpill-ext {
  background:var(--accent-lt); color:var(--accent) !important;
  border-radius:5px; padding:2px 8px;
  font-size:10px; font-weight:700; letter-spacing:.06em;
}
.fpill-chunks {
  background:var(--sage-lt); color:var(--sage) !important;
  border-radius:5px; padding:2px 8px; font-size:11px; font-weight:600;
}

/* process button */
.proc-btn-wrap .stButton > button {
  background: linear-gradient(135deg,var(--accent),var(--accent2)) !important;
  color:#fff !important; border:none !important;
  border-radius:11px !important; padding:13px 28px !important;
  font-family:'Sora',sans-serif !important;
  font-size:14px !important; font-weight:700 !important;
  box-shadow:0 4px 14px var(--accent-glow) !important;
  transition:opacity .15s, transform .1s !important;
}
.proc-btn-wrap .stButton > button:hover { opacity:.88 !important; transform:translateY(-1px) !important; }

/* how it works card */
.how-card {
  background: var(--surface2);
  border: 1px solid var(--border2);
  border-radius:14px; padding:22px 24px; box-shadow:var(--sh);
}
.how-card-title {
  font-size:11px; font-weight:700; color:var(--text3) !important;
  letter-spacing:.12em; text-transform:uppercase; margin-bottom:16px;
}
.how-step { display:flex; align-items:flex-start; gap:12px; margin-bottom:14px; }
.how-step:last-child { margin-bottom:0; }
.how-num {
  width:26px; height:26px; border-radius:50%;
  background:var(--accent-lt); color:var(--accent) !important;
  font-size:12px; font-weight:700;
  display:flex; align-items:center; justify-content:center;
  flex-shrink:0; margin-top:1px;
  border:1px solid rgba(108,142,245,.25);
}
.how-text { font-size:13px; color:var(--text2) !important; line-height:1.6; }
.how-text strong { color:var(--text) !important; font-weight:600; }

/* format badges */
.fmt-pdf  { background:var(--rose-lt);  color:var(--rose)   !important; }
.fmt-docx { background:var(--accent-lt);color:var(--accent) !important; }
.fmt-pptx { background:var(--gold-lt);  color:var(--gold)   !important; }
.fmt-txt  { background:var(--sage-lt);  color:var(--sage)   !important; }
.fmt-badge { border-radius:6px; padding:3px 10px; font-size:12px; font-weight:700; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════
def file_icon(name: str) -> tuple:
    ext = Path(name).suffix.lower()
    return {
        ".pdf":  ("📄", "fi-pdf"),
        ".docx": ("📝", "fi-docx"),
        ".pptx": ("📊", "fi-pptx"),
        ".txt":  ("📃", "fi-txt"),
    }.get(ext, ("📁", "fi-gen"))


# ══════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sb-logo">
      <div class="sb-logo-icon">🎓</div>
      <div>
        <span class="sb-logo-text">Scholar AI</span>
        <span class="sb-logo-beta">beta</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    stats = get_stats()

    st.markdown(f"""
    <div class="sb-stats">
      <div class="sb-stat">
        <div class="sb-stat-n">{stats["documents"]}</div>
        <div class="sb-stat-l">Docs</div>
      </div>
      <div class="sb-stat">
        <div class="sb-stat-n">{stats["chunks"]}</div>
        <div class="sb-stat-l">Chunks</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.indexed:
        st.markdown('<span class="status-pill status-on"><span class="status-dot"></span> Active</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-pill status-off"><span class="status-dot"></span> No docs indexed</span>', unsafe_allow_html=True)

    st.markdown('<span class="sb-sec">Indexed Files</span>', unsafe_allow_html=True)
    st.markdown('<div class="sb-files">', unsafe_allow_html=True)
    if stats["sources"]:
        for s in stats["sources"]:
            ico, cls = file_icon(s)
            display = s if len(s) <= 26 else s[:23] + "…"
            st.markdown(f"""
            <div class="sb-file-row">
              <span class="sb-file-icon {cls}">{ico}</span>
              <span class="sb-file-name">{display}</span>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown('<p style="font-size:12px;font-style:italic;padding:8px 4px;color:var(--text3)!important;">No files yet</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑  Clear Knowledge Base", use_container_width=True):
        import vector_store as vs
        vs.chunks_store.clear()
        vs.metadata_store.clear()
        vs.faiss_index = None
        st.session_state.history = []
        st.session_state.indexed = False
        st.rerun()

    st.markdown("""
    <p style="font-size:11px;color:var(--text3)!important;margin-top:20px;line-height:1.7;padding:0 4px;">
    Supports <strong style="color:var(--text2)!important;">PDF · DOCX · PPTX · TXT</strong><br>
    Powered by Gemini + FAISS
    </p>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
#  TOP BAR
# ══════════════════════════════════════════════════════
tb_l, tb_m, tb_r = st.columns([2, 2, 2])

with tb_l:
    st.markdown("""
    <div style="padding:14px 0 10px;">
      <span style="font-family:'Lora',serif;font-size:18px;font-weight:600;color:var(--text);">
        Scholar <em style="font-style:italic;color:var(--accent);">AI</em>
      </span>
    </div>""", unsafe_allow_html=True)

with tb_m:
    vc1, vc2 = st.columns(2)
    with vc1:
        if st.button("💬  Chat", use_container_width=True,
                     type="primary" if st.session_state.view == "chat" else "secondary"):
            st.session_state.view = "chat"; st.rerun()
    with vc2:
        if st.button("📤  Upload", use_container_width=True,
                     type="primary" if st.session_state.view == "upload" else "secondary"):
            st.session_state.view = "upload"; st.rerun()

with tb_r:
    mc1, mc2 = st.columns(2)
    with mc1:
        if st.button("📖 Explain", use_container_width=True,
                     type="primary" if st.session_state.mode == "explain" else "secondary"):
            st.session_state.mode = "explain"; st.rerun()
    with mc2:
        if st.button("📝 Exam Q", use_container_width=True,
                     type="primary" if st.session_state.mode == "exam" else "secondary"):
            st.session_state.mode = "exam"; st.rerun()

st.markdown('<hr style="margin:0 !important;">', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
#  UPLOAD VIEW
# ══════════════════════════════════════════════════════
if st.session_state.view == "upload":
    st.markdown('<div style="padding:32px 44px;">', unsafe_allow_html=True)

    col_up, col_how = st.columns([3, 2], gap="large")

    with col_up:
        st.markdown("""
        <div class="ucard-title">📤 Upload Study Materials</div>
        <div class="ucard-sub">Drag & drop or browse your files. Supports PDF, DOCX, PPTX, and TXT.</div>
        """, unsafe_allow_html=True)

        files = st.file_uploader(
            "Drop files here",
            type=["pdf", "docx", "pptx", "txt"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if files:
            for f in files:
                ext = Path(f.name).suffix.upper().lstrip(".")
                st.markdown(
                    f'<div class="fpill">📄 <span class="fpill-name">{f.name}</span>'
                    f'<span class="fpill-ext">{ext}</span></div>',
                    unsafe_allow_html=True
                )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="proc-btn-wrap">', unsafe_allow_html=True)
        process_btn = st.button("⚡  Process & Index Documents", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if process_btn and files:
            new_docs = []
            prog = st.progress(0, text="Reading files…")
            for i, file in enumerate(files):
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.name).suffix) as tmp:
                    tmp.write(file.read())
                    path = tmp.name
                try:
                    doc = process_document(path)
                    doc["filename"] = file.name
                    new_docs.append(doc)
                    st.markdown(
                        f'<div class="fpill fpill-ok">✅ <span class="fpill-name">{file.name}</span>'
                        f'<span class="fpill-chunks">{doc["num_chunks"]} chunks</span></div>',
                        unsafe_allow_html=True
                    )
                except Exception as e:
                    st.markdown(
                        f'<div class="fpill fpill-err">❌ <span class="fpill-name">{file.name}: {e}</span></div>',
                        unsafe_allow_html=True
                    )
                finally:
                    os.unlink(path)
                prog.progress((i + 1) / len(files), text=f"Processing {i+1}/{len(files)}…")

            if new_docs:
                with st.spinner("Embedding & building vector index…"):
                    add_documents(new_docs)
                st.session_state.indexed = True
                prog.empty()
                st.success(f"🎉 {len(new_docs)} document(s) indexed! Switch to Chat to ask questions.")
        elif process_btn and not files:
            st.warning("Upload at least one file first.")

    with col_how:
        st.markdown("""
        <div style="margin-top:42px;">
          <div class="how-card">
            <div class="how-card-title">How it works</div>
            <div class="how-step">
              <div class="how-num">1</div>
              <div class="how-text"><strong>Upload</strong> your PDFs, notes, slides, or textbooks.</div>
            </div>
            <div class="how-step">
              <div class="how-num">2</div>
              <div class="how-text"><strong>Process</strong> — Scholar AI chunks, embeds, and indexes your content.</div>
            </div>
            <div class="how-step">
              <div class="how-num">3</div>
              <div class="how-text"><strong>Ask</strong> any question in the Chat tab.</div>
            </div>
          </div>

          <div class="how-card" style="margin-top:14px;">
            <div class="how-card-title">Supported formats</div>
            <div style="display:flex;flex-wrap:wrap;gap:7px;margin-top:4px;">
              <span class="fmt-badge fmt-pdf">PDF</span>
              <span class="fmt-badge fmt-docx">DOCX</span>
              <span class="fmt-badge fmt-pptx">PPTX</span>
              <span class="fmt-badge fmt-txt">TXT</span>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
#  CHAT VIEW
# ══════════════════════════════════════════════════════
else:
    if not st.session_state.history:
        mode_hint = (
            "Ask me to explain any topic from your uploaded documents."
            if st.session_state.mode == "explain"
            else "Paste an exam question — I'll write a mark-scheme-aware answer."
        )
        st.markdown(f"""
        <div class="welcome-card">
          <div class="welcome-glyph">🎓</div>
          <h2 class="welcome-h">Ask <em>Scholar AI</em> anything</h2>
          <p class="welcome-p">{mode_hint}<br>Upload your materials first, then ask away.</p>
          <div class="welcome-chips">
            <span class="welcome-chip">📖 Topic explanations</span>
            <span class="welcome-chip">📝 Exam question solving</span>
            <span class="welcome-chip">📚 Source-grounded answers</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # message history
    for item in st.session_state.history:
        mode_lbl = "Explain Topic" if item["mode"] == "explain" else "Exam Answer"

        # user bubble
        st.markdown(f"""
        <div class="msg-row msg-row-user">
          <div class="bubble bubble-user">{item["q"]}</div>
          <div class="avatar avatar-user">🧑</div>
        </div>""", unsafe_allow_html=True)

        # ai label
        st.markdown(f'<div class="ai-label" style="margin-left:44px;">Scholar AI · {mode_lbl}</div>', unsafe_allow_html=True)

        # ai answer — use st.container for proper markdown rendering
        with st.container():
            col_av, col_ans = st.columns([0.06, 0.94])
            with col_av:
                st.markdown('<div class="avatar avatar-ai" style="margin-top:2px;">🤖</div>', unsafe_allow_html=True)
            with col_ans:
                st.markdown(
                    '<div style="background:var(--surface2);border:1px solid var(--border2);'
                    'color:var(--text);padding:16px 20px;border-radius:5px 14px 14px 14px;'
                    'font-size:14px;line-height:1.75;box-shadow:var(--sh);">',
                    unsafe_allow_html=True
                )
                st.markdown(item["a"])
                if item.get("sources"):
                    seen = {s["source"]: s for s in item["sources"]}
                    chips = "".join(f'<span class="src-chip">📄 {s["source"]}</span>' for s in seen.values())
                    st.markdown(f'<div class="src-bar">{chips}</div>', unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

    # input bar
    st.markdown('<hr style="margin:8px 0 16px !important;">', unsafe_allow_html=True)

    inp_col, btn_col = st.columns([4, 1], gap="small")
    with inp_col:
        placeholder_txt = (
            "Explain a topic from your documents…"
            if st.session_state.mode == "explain"
            else "Paste your exam question here… (e.g. 8 marks)"
        )
        query = st.text_area(
            "query", placeholder=placeholder_txt,
            label_visibility="collapsed", key="chat_input",
            height=120,
        )
    with btn_col:
        st.markdown('<div class="ask-btn-wrap">', unsafe_allow_html=True)
        ask_btn = st.button("Send →", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    mode_desc = "Explain Topic — structured explanation with examples" if st.session_state.mode == "explain" \
        else "Solve Exam Question — mark-scheme-aware full answer"
    st.markdown(f'<p style="font-size:11px;color:var(--text3)!important;margin-top:4px;">Mode: <strong style="color:var(--text2)!important;">{mode_desc}</strong></p>', unsafe_allow_html=True)

    if ask_btn and query:
        if not st.session_state.indexed:
            st.warning("⚠️ Please switch to the Upload tab and index your documents first.")
        else:
            with st.spinner("Scholar AI is thinking…"):
                if st.session_state.mode == "explain":
                    answer, sources = answer_topic(query)
                else:
                    answer, sources = solve_question(query)
            st.session_state.history.append({
                "q": query, "a": answer,
                "sources": sources, "mode": st.session_state.mode,
            })
            st.rerun()