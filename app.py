# ═══════════════════════════════════════════════════════════════════════════════
#  CropSense AI  ·  AgriCast UI (strict copy) + CropSense backend
#  UI:      100% AgriCast — dark theme, Syne/Outfit, neon-green tokens,
#           navbar, auth (users.json), chatbot (Groq), session-state routing
#  Backend: 100% CropSense — YOLO model, detection, severity, treatments
# ═══════════════════════════════════════════════════════════════════════════════

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import json
import hashlib
from pathlib import Path
from groq import Groq

# ─── USER STORE  (AgriCast — unchanged) ─────────────────────────────────────
USERS_FILE = Path("users.json")

def _ensure_users_file():
    if not USERS_FILE.exists():
        USERS_FILE.write_text(json.dumps({}))

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def add_user(username: str, password: str) -> bool:
    _ensure_users_file()
    users = json.loads(USERS_FILE.read_text())
    if username in users:
        return False
    users[username] = {"password": hash_password(password)}
    USERS_FILE.write_text(json.dumps(users, indent=2))
    return True

def authenticate_user(username: str, password: str) -> bool:
    _ensure_users_file()
    users = json.loads(USERS_FILE.read_text())
    if username not in users:
        return False
    return users[username]["password"] == hash_password(password)

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CropSense AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ═══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS  —  100% AgriCast CSS (zero CropSense CSS)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=Outfit:wght@300;400;500;600;700&display=swap');

:root {
    --bg:           #071018;
    --bg-2:         #0B1620;
    --bg-3:         #101C28;
    --surface:      #162331;
    --surface-2:    #1B2A3A;
    --surface-3:    #223447;
    --border:       rgba(255,255,255,0.08);
    --border-2:     rgba(255,255,255,0.14);
    --border-3:     rgba(110,255,155,0.30);
    --green:        #7CFF9B;
    --green-dim:    #47C96D;
    --green-ghost:  rgba(124,255,155,0.10);
    --green-glow:   rgba(124,255,155,0.22);
    --amber:        #FFC857;
    --amber-dim:    rgba(255,200,87,0.15);
    --blue:         #7CB8FF;
    --blue-dim:     rgba(124,184,255,0.12);
    --text-1:       #FFFFFF;
    --text-2:       #D6E1EA;
    --text-3:       #9FB2C4;
    --text-green:   #8DFFAA;
    --radius-sm:    6px;
    --radius-md:    10px;
    --radius-lg:    14px;
    --radius-xl:    20px;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp {
    background-color: var(--bg) !important;
    font-family: 'Outfit', sans-serif !important;
    color: var(--text-1) !important;
}

#MainMenu, footer, header { visibility: hidden !important; }
section[data-testid="stSidebar"] { display: none !important; }

.block-container {
    padding: 0 2rem 4rem !important;
    max-width: 1280px !important;
}

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--surface-3); border-radius: 4px; }

.stApp::before {
    content: '';
    position: fixed; top: 0; left: 0; right: 0; height: 1px; z-index: 9999;
    background: linear-gradient(90deg, transparent 0%, var(--green) 30%, var(--green) 70%, transparent 100%);
    opacity: 0.6;
}

h1, h2, h3, h4, h5 { font-family: 'Syne', sans-serif !important; color: var(--text-1) !important; }

/* ══ NAVBAR ══ */
.nav-brand { display: flex; align-items: center; gap: 0.65rem; flex-shrink: 0; }
.nav-brand-icon {
    width: 34px; height: 34px;
    background: linear-gradient(135deg, var(--green-dim), #2E6E42);
    border-radius: var(--radius-md);
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem; box-shadow: 0 0 16px var(--green-glow);
}
.nav-brand-text { font-family: 'Syne', sans-serif; font-size: 1.2rem; font-weight: 700; color: var(--text-1); letter-spacing: -0.3px; }
.nav-brand-text span { color: var(--green); }
.nav-brand-sub { font-size: 0.62rem; color: var(--text-3); letter-spacing: 2px; text-transform: uppercase; font-weight: 500; margin-top: 1px; }
.nav-user-pill { display: inline-flex; align-items: center; gap: 0.5rem; background: var(--surface); border: 1px solid var(--border-2); border-radius: 100px; padding: 0.3rem 0.85rem 0.3rem 0.3rem; }
.nav-avatar { width: 28px; height: 28px; border-radius: 50%; background: linear-gradient(135deg, var(--green-dim), var(--blue)); display: flex; align-items: center; justify-content: center; font-size: 0.72rem; font-weight: 700; color: white; font-family: 'Syne', sans-serif; }
.nav-uname { font-size: 0.82rem; font-weight: 500; color: var(--text-1); }

/* ══ HERO ══ */
.hero-root { padding: 3.5rem 0 2.8rem; text-align: center; position: relative; }
.hero-badge { display: inline-flex; align-items: center; gap: 0.4rem; background: var(--green-ghost); border: 1px solid var(--border-3); border-radius: 100px; padding: 0.3rem 0.9rem; font-size: 0.72rem; font-weight: 600; color: var(--green); letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 1.3rem; }
.hero-badge::before { content: ''; width: 6px; height: 6px; background: var(--green); border-radius: 50%; box-shadow: 0 0 6px var(--green); }
.hero-title { font-family: 'Syne', sans-serif !important; font-size: clamp(2.2rem, 4vw, 3.4rem) !important; font-weight: 800 !important; color: var(--text-1) !important; line-height: 1.12 !important; letter-spacing: -1px !important; margin-bottom: 1.1rem !important; }
.hero-title em { font-style: normal; color: var(--green); }
.hero-sub { font-size: 1rem; color: var(--text-2); max-width: 480px; margin: 0 auto; line-height: 1.7; font-weight: 400; }

/* ══ KPI CARDS ══ */
.kpi-card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-lg); padding: 1.4rem 1.6rem; position: relative; overflow: hidden; transition: border-color 0.2s, transform 0.2s; }
.kpi-card:hover { border-color: var(--border-3); transform: translateY(-2px); }
.kpi-card::after { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, var(--green-dim), transparent); opacity: 0.6; }
.kpi-num { font-family: 'Syne', sans-serif; font-size: 2.4rem; font-weight: 800; color: var(--text-1); line-height: 1; margin-bottom: 0.3rem; letter-spacing: -1px; }
.kpi-label { font-size: 0.68rem; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; color: var(--green); margin-bottom: 0.55rem; }
.kpi-desc { font-size: 0.8rem; color: var(--text-3); line-height: 1.55; }

/* ══ SECTION CARDS ══ */
.sec-card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-xl); padding: 1.8rem 2rem; margin-bottom: 1.5rem; }
.sec-card-header { display: flex; align-items: center; gap: 0.75rem; padding-bottom: 1.2rem; margin-bottom: 1.4rem; border-bottom: 1px solid var(--border); }
.sec-icon { width: 36px; height: 36px; background: var(--green-ghost); border: 1px solid var(--border-3); border-radius: var(--radius-md); display: flex; align-items: center; justify-content: center; font-size: 1rem; }
.sec-title { font-family: 'Syne', sans-serif !important; font-size: 1rem !important; font-weight: 700 !important; color: var(--text-1) !important; margin: 0 !important; }
.sec-subtitle { font-size: 0.77rem; color: var(--text-3); margin-top: 1px; }

/* ══ RESULT CARD (prediction/detection output) ══ */
.result-card { background: var(--surface-2); border: 1px solid var(--border-3); border-radius: var(--radius-xl); padding: 2rem 2.2rem; margin-top: 1.6rem; position: relative; overflow: hidden; }
.result-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, var(--green-dim), var(--blue), var(--green-dim)); }
.result-tag { font-size: 0.65rem; letter-spacing: 2.5px; text-transform: uppercase; color: var(--text-3); margin-bottom: 1.4rem; font-weight: 600; }

/* ══ DETECTION CARDS ══ */
.detect-card { background: var(--surface-2); border: 1px solid var(--border-3); border-radius: var(--radius-xl); padding: 1.6rem 1.8rem; margin-bottom: 1rem; position: relative; overflow: hidden; transition: border-color 0.2s, transform 0.2s; }
.detect-card:hover { border-color: rgba(124,255,155,0.5); transform: translateY(-2px); }
.detect-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, var(--green-dim), transparent); }
.healthy-tag { display: inline-flex; align-items: center; gap: 5px; font-size: 0.65rem; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; padding: 3px 10px; background: rgba(124,255,155,0.1); border: 1px solid var(--border-3); color: var(--green); border-radius: 100px; margin-bottom: 0.9rem; }
.disease-tag { display: inline-flex; align-items: center; gap: 5px; font-size: 0.65rem; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; padding: 3px 10px; background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.35); color: #F87171; border-radius: 100px; margin-bottom: 0.9rem; }
.disease-name-label { font-size: 0.65rem; letter-spacing: 2px; text-transform: uppercase; color: var(--text-3); font-weight: 600; margin-bottom: 0.25rem; }
.disease-name-value { font-family: 'Syne', sans-serif; font-size: 1.2rem; font-weight: 700; color: var(--text-1); margin-bottom: 0.9rem; }
.conf-row { display: flex; align-items: center; gap: 10px; }
.conf-label { font-size: 0.72rem; color: var(--text-3); white-space: nowrap; }
.conf-bar-bg { flex: 1; height: 5px; background: var(--surface-3); border-radius: 100px; overflow: hidden; }
.conf-bar-fill { height: 100%; border-radius: 100px; background: linear-gradient(90deg, var(--green-dim), var(--green)); }
.conf-value { font-family: 'Syne', sans-serif; font-size: 0.85rem; font-weight: 700; color: var(--green); white-space: nowrap; }

/* ══ TREATMENT CARD ══ */
.treatment-card { background: var(--surface-2); border: 1px solid var(--border); border-radius: var(--radius-xl); padding: 1.6rem 1.8rem; margin-bottom: 1rem; transition: border-color 0.2s, transform 0.2s; }
.treatment-card:hover { border-color: var(--border-3); transform: translateY(-2px); }
.treatment-icon { font-size: 1.5rem; margin-bottom: 0.4rem; }
.treatment-label { font-size: 0.72rem; color: var(--text-3); letter-spacing: 0.03em; text-transform: uppercase; margin-bottom: 0.5rem; }
.treatment-text { font-size: 0.9rem; color: var(--text-2); line-height: 1.7; }

/* ══ SEVERITY CARD ══ */
.severity-card { background: var(--surface-2); border: 1px solid var(--border-3); border-radius: var(--radius-xl); padding: 1.8rem 2rem; position: relative; overflow: hidden; }
.severity-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, var(--green-dim), var(--blue), transparent); }
.severity-meter-bg { height: 8px; background: var(--surface-3); border-radius: 100px; overflow: hidden; margin: 0.9rem 0; }
.severity-stats { display: flex; align-items: center; justify-content: space-between; margin-top: 0.3rem; }
.severity-percent { font-family: 'Syne', sans-serif; font-size: 1.8rem; font-weight: 800; line-height: 1; }
.sev-low      { color: var(--green); }
.sev-moderate { color: var(--amber); }
.sev-high     { color: #F87171; }
.severity-badge { font-size: 0.68rem; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; padding: 4px 12px; border-radius: 100px; }
.badge-low      { background: rgba(124,255,155,0.1); color: var(--green);  border: 1px solid var(--border-3); }
.badge-moderate { background: rgba(255,200,87,0.1);  color: var(--amber);  border: 1px solid rgba(255,200,87,0.35); }
.badge-high     { background: rgba(239,68,68,0.1);   color: #F87171;       border: 1px solid rgba(239,68,68,0.35); }
.severity-level-low      { background: linear-gradient(90deg, var(--green-dim), var(--green)); }
.severity-level-moderate { background: linear-gradient(90deg, #D97706, var(--amber)); }
.severity-level-high     { background: linear-gradient(90deg, #DC2626, #F87171); }

/* ══ TIP BOX ══ */
.tip-box { display: flex; align-items: flex-start; gap: 12px; background: var(--bg-3); border: 1px solid var(--border); border-radius: var(--radius-md); padding: 0.75rem 1rem; margin-bottom: 0.6rem; }
.tip-icon { font-size: 1rem; flex-shrink: 0; }
.tip-text { font-size: 0.83rem; color: var(--text-2); line-height: 1.6; }

/* ══ NO-DETECT BANNER ══ */
.no-detect { display: flex; align-items: center; gap: 10px; background: rgba(255,200,87,0.08); border: 1px solid rgba(255,200,87,0.3); border-radius: var(--radius-md); padding: 1rem 1.2rem; font-size: 0.9rem; color: var(--amber); }

/* ══ PILLS ══ */
.pill-row { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.2rem; }
.pill { display: inline-flex; align-items: center; gap: 0.3rem; background: var(--surface-3); border: 1px solid var(--border-2); border-radius: 100px; padding: 0.3rem 0.75rem; font-size: 0.78rem; color: var(--text-2); font-weight: 500; }

/* ══ METRIC ══ */
[data-testid="stMetricValue"] { font-family: 'Syne', sans-serif !important; font-size: 1.75rem !important; font-weight: 800 !important; color: var(--text-1) !important; letter-spacing: -0.5px !important; }
[data-testid="stMetricLabel"] { font-size: 0.65rem !important; letter-spacing: 2px !important; text-transform: uppercase !important; color: var(--text-3) !important; font-weight: 600 !important; }
[data-testid="stMetricDelta"] { color: var(--green) !important; font-size: 0.75rem !important; }
[data-testid="metric-container"] { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: var(--radius-lg) !important; padding: 1.1rem 1.3rem !important; transition: border-color 0.2s !important; }
[data-testid="metric-container"]:hover { border-color: var(--border-3) !important; }

/* ══ INPUTS ══ */
.stTextInput label, .stNumberInput label, .stTextArea label, .stSelectbox label, .stSlider label {
    font-family: 'Outfit', sans-serif !important; font-size: 0.7rem !important; font-weight: 600 !important;
    letter-spacing: 1.8px !important; text-transform: uppercase !important;
    color: var(--text-3) !important; margin-bottom: 0.4rem !important;
}
.stTextInput input, .stNumberInput input, .stTextArea textarea {
    background: var(--bg-3) !important; border: 1px solid var(--border-2) !important;
    border-radius: var(--radius-md) !important; padding: 0.62rem 0.9rem !important;
    font-size: 0.9rem !important; color: var(--text-1) !important;
    font-family: 'Outfit', sans-serif !important; transition: border-color 0.18s, box-shadow 0.18s !important;
}
.stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus {
    border-color: rgba(99,210,130,0.4) !important;
    box-shadow: 0 0 0 3px rgba(99,210,130,0.08) !important; outline: none !important;
}
.stTextInput input::placeholder, .stTextArea textarea::placeholder { color: var(--text-3) !important; }
.stSelectbox > div > div { background: var(--bg-3) !important; border: 1px solid var(--border-2) !important; border-radius: var(--radius-md) !important; }
.stSelectbox > div > div:focus-within { border-color: rgba(99,210,130,0.4) !important; box-shadow: 0 0 0 3px rgba(99,210,130,0.08) !important; }
.stSelectbox span, .stSelectbox [data-baseweb="select"] div { font-family: 'Outfit', sans-serif !important; font-size: 0.9rem !important; color: var(--text-1) !important; }
[data-baseweb="slider"] [role="slider"] { background: var(--green) !important; border-color: var(--green) !important; box-shadow: 0 0 0 4px rgba(99,210,130,0.2) !important; }
[data-testid="stSliderTrackActive"] { background: var(--green-dim) !important; }

/* ══ BUTTONS ══ */
.stButton > button {
    background: var(--green-dim) !important; color: #FFFFFF !important;
    border: none !important; border-radius: var(--radius-md) !important;
    padding: 0.65rem 1.5rem !important; font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important; font-size: 0.84rem !important; letter-spacing: 0.5px !important;
    width: 100% !important; transition: all 0.18s ease !important;
    box-shadow: 0 4px 16px rgba(75,174,101,0.25) !important;
    text-shadow: -1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000, 0px 2px 4px rgba(0,0,0,0.35);
}
.stButton > button:hover {
    background: var(--green) !important; color: #FFFFFF !important;
    box-shadow: 0 6px 24px rgba(99,210,130,0.35) !important; transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

.nav-btn-row .stButton > button {
    background: #47C96D !important; color: #FFFFFF !important;
    border: 1px solid rgba(255,255,255,0.10) !important; border-radius: 12px !important;
    min-height: 48px !important; width: auto !important; min-width: 140px !important;
    padding: 0 22px !important; font-size: 0.72rem !important; font-weight: 700 !important;
    letter-spacing: 1px !important; text-transform: uppercase !important;
    white-space: nowrap !important; line-height: 1 !important;
    overflow: visible !important; word-break: normal !important; overflow-wrap: normal !important;
    transition: all 0.2s ease !important; box-shadow: 0 6px 20px rgba(71,201,109,0.20) !important;
    text-shadow: -1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000;
}
.nav-btn-row .stButton > button:hover {
    background: #7CFF9B !important; color: #FFFFFF !important;
    transform: translateY(-2px) !important; box-shadow: 0 10px 28px rgba(124,255,155,0.28) !important;
}

/* ══ FILE UPLOADER ══ */
[data-testid="stFileUploader"] > div {
    background: var(--surface) !important;
    border: 2px dashed rgba(124,255,155,0.35) !important;
    border-radius: var(--radius-lg) !important; transition: border-color 0.2s;
}
[data-testid="stFileUploader"] > div:hover { border-color: rgba(124,255,155,0.6) !important; }

/* ══ TABS ══ */
[data-testid="stTabs"] button { font-family: 'Outfit', sans-serif !important; color: var(--text-3) !important; font-weight: 500 !important; font-size: 0.85rem !important; }
[data-testid="stTabs"] button[aria-selected="true"] { color: var(--green) !important; border-bottom: 2px solid var(--green) !important; }

/* ══ DATAFRAME ══ */
[data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: var(--radius-lg) !important; overflow: hidden !important; }
[data-testid="stDataFrame"] thead th { background: var(--surface-2) !important; color: var(--text-3) !important; font-weight: 600 !important; font-size: 0.72rem !important; letter-spacing: 1.5px !important; text-transform: uppercase !important; border-bottom: 1px solid var(--border) !important; }
[data-testid="stDataFrame"] tbody tr:hover { background: var(--surface-2) !important; }

/* ══ SPINNER ══ */
.stSpinner > div { border-color: var(--green) transparent var(--green) var(--green) !important; }

/* ══ AUTH ══ */
.auth-outer { display: flex; align-items: center; justify-content: center; min-height: 80vh; }
.auth-card { background: var(--surface); border: 1px solid var(--border-2); border-radius: var(--radius-xl); padding: 2.6rem 2.4rem; width: 100%; box-shadow: 0 20px 60px rgba(0,0,0,0.4); position: relative; overflow: hidden; }
.auth-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, transparent, var(--green-dim), transparent); }
.auth-logo { text-align: center; margin-bottom: 1.8rem; }
.auth-logo-icon { width: 52px; height: 52px; background: var(--green-ghost); border: 1px solid var(--border-3); border-radius: 14px; display: flex; align-items: center; justify-content: center; font-size: 1.5rem; margin: 0 auto 0.9rem; box-shadow: 0 0 24px var(--green-glow); }
.auth-logo-name { font-family: 'Syne', sans-serif; font-size: 1.7rem; font-weight: 800; color: var(--text-1); letter-spacing: -0.5px; }
.auth-logo-name span { color: var(--green); }
.auth-logo-sub { font-size: 0.82rem; color: var(--text-3); margin-top: 0.3rem; }
.auth-divider { height: 1px; background: var(--border); margin: 1.4rem 0; }
.auth-switch { text-align: center; font-size: 0.82rem; color: var(--text-3); }

/* ══ PAGE HEADER ══ */
.page-header { margin-bottom: 2rem; }
.page-header h1 { font-family: 'Syne', sans-serif !important; font-size: 1.75rem !important; font-weight: 800 !important; color: var(--text-1) !important; letter-spacing: -0.5px !important; margin-bottom: 0.3rem !important; }
.page-header p { font-size: 0.88rem; color: var(--text-3); line-height: 1.6; }

/* ══ CHAT ══ */
.chat-response { background: var(--surface-2); border: 1px solid var(--border-3); border-radius: var(--radius-lg); padding: 1.4rem 1.6rem; margin-top: 1.2rem; }
.chat-resp-label { font-size: 0.65rem; letter-spacing: 2.5px; text-transform: uppercase; color: var(--green); font-weight: 700; margin-bottom: 0.7rem; }
.chat-resp-text { font-size: 0.9rem; color: var(--text-2); line-height: 1.75; }

/* ══ DIVIDERS ══ */
.g-divider { height: 1px; background: var(--border); margin: 1.5rem 0; }

/* ══ ABOUT ══ */
.about-mission { font-size: 0.92rem; color: var(--text-2); line-height: 1.8; }

/* ══ ALERTS ══ */
.stAlert { border-radius: var(--radius-md) !important; border: none !important; font-family: 'Outfit', sans-serif !important; font-size: 0.88rem !important; }
.stSuccess { background: rgba(99,210,130,0.08) !important; border: 1px solid rgba(99,210,130,0.25) !important; color: var(--green) !important; }
.stError   { background: rgba(239,68,68,0.08) !important;  border: 1px solid rgba(239,68,68,0.25) !important; }
.stWarning { background: rgba(240,180,41,0.08) !important; border: 1px solid rgba(240,180,41,0.25) !important; }

/* ══ MISC ══ */
.spacer-sm { display: block; height: 0.8rem; }
.spacer-md { display: block; height: 1.4rem; }
.spacer-lg { display: block; height: 2rem; }

.inline-stat { display: flex; align-items: center; gap: 0.5rem; padding: 0.6rem 0.85rem; background: var(--bg-3); border: 1px solid var(--border); border-radius: var(--radius-md); font-size: 0.82rem; color: var(--text-2); }
.inline-stat strong { color: var(--green); font-weight: 700; font-family: 'Syne', sans-serif; }

p  { font-family: 'Outfit', sans-serif !important; color: var(--text-2) !important; line-height: 1.7 !important; }
li { font-family: 'Outfit', sans-serif !important; color: var(--text-2) !important; font-size: 0.9rem !important; margin-bottom: 0.3rem !important; }
a  { color: var(--green) !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  TREATMENT DATABASE  (CropSense backend — unchanged)
# ═══════════════════════════════════════════════════════════════════════════════
TREATMENT = {
    "Apple Scab Leaf":                      ("🍎", "Apply Captan or Mancozeb fungicide every 7–10 days during wet seasons. Remove fallen leaves to reduce spore load."),
    "Apple leaf":                           ("✅", "Healthy leaf — maintain balanced fertilisation and adequate irrigation. Monitor regularly for early signs of disease."),
    "Apple rust leaf":                      ("🍎", "Apply sulfur-based or myclobutanil fungicide at bud break. Remove nearby juniper hosts to break the disease cycle."),
    "Bell_pepper leaf":                     ("✅", "Healthy plant — ensure consistent watering and well-drained soil. Avoid overhead irrigation."),
    "Bell_pepper leaf spot":                ("🫑", "Spray copper-based fungicide (e.g., Copper Oxychloride). Remove infected leaves and avoid splashing water."),
    "Blueberry leaf":                       ("✅", "Healthy plant — maintain soil pH 4.5–5.5 and mulch around roots to retain moisture."),
    "Cherry leaf":                          ("✅", "Healthy leaf — no immediate action needed. Prune for airflow and inspect seasonally."),
    "Corn Gray leaf spot":                  ("🌽", "Apply foliar fungicide such as Azoxystrobin. Rotate crops annually and use resistant hybrids."),
    "Corn leaf blight":                     ("🌽", "Use resistant varieties and apply propiconazole fungicide at disease onset. Ensure proper field drainage."),
    "Corn rust leaf":                       ("🌽", "Apply fungicide (Triazole group) early. Monitor weather conditions and use rust-resistant cultivars."),
    "Peach leaf":                           ("✅", "Healthy leaf — regular pruning and balanced nutrition help prevent future infections."),
    "Potato leaf":                          ("✅", "Healthy plant — hill soil around base, water consistently and scout for early blight symptoms."),
    "Potato leaf early blight":             ("🥔", "Apply Mancozeb or Chlorothalonil spray every 7 days. Remove and destroy heavily infected foliage."),
    "Potato leaf late blight":              ("🥔", "Apply Chlorothalonil or metalaxyl-based fungicide immediately. Destroy infected plants to halt rapid spread."),
    "Raspberry leaf":                       ("✅", "Healthy leaf — ensure canes are well-spaced for airflow; remove old canes after harvest."),
    "Soyabean leaf":                        ("✅", "Healthy plant — maintain row spacing for airflow and monitor for soybean rust during humid weather."),
    "Soybean leaf":                         ("✅", "Healthy plant — use certified seeds and scout fields regularly during vegetative stages."),
    "Squash Powdery mildew leaf":           ("🎃", "Apply sulfur spray or potassium bicarbonate. Improve air circulation and avoid excess nitrogen fertiliser."),
    "Strawberry leaf":                      ("✅", "Healthy plant — replace beds every 2–3 years and keep mulch fresh to reduce soil-borne diseases."),
    "Tomato Early blight leaf":             ("🍅", "Apply Neem oil or copper fungicide every 7 days. Remove infected lower leaves and mulch to prevent soil splash."),
    "Tomato Septoria leaf spot":            ("🍅", "Apply chlorothalonil or copper fungicide. Remove infected leaves promptly and stake plants to improve airflow."),
    "Tomato leaf":                          ("✅", "Healthy leaf — maintain consistent moisture, calcium supply and scout for pests weekly."),
    "Tomato leaf bacterial spot":           ("🍅", "Apply copper-based bactericide. Avoid overhead watering and use disease-free transplants."),
    "Tomato leaf late blight":              ("🍅", "Apply fungicide (Mancozeb or Cymoxanil) immediately. Destroy infected plant material; do not compost."),
    "Tomato leaf mosaic virus":             ("🍅", "No chemical cure — remove and destroy infected plants promptly. Disinfect tools and control aphid vectors."),
    "Tomato leaf yellow virus":             ("🍅", "Control whitefly populations with insecticidal soap or neem oil. Use reflective mulches and resistant varieties."),
    "Tomato mold leaf":                     ("🍅", "Improve greenhouse ventilation. Apply fungicide (chlorothalonil). Reduce humidity and avoid leaf wetness."),
    "Tomato two spotted spider mites leaf": ("🍅", "Use insecticidal soap, neem oil, or predatory mites (Phytoseiidae). Avoid broad-spectrum insecticides."),
    "grape leaf":                           ("✅", "Healthy leaf — prune for open canopy, monitor for downy/powdery mildew during humid periods."),
    "grape leaf black rot":                 ("🍇", "Apply Myclobutanil or Mancozeb fungicide from bud break. Remove mummified berries and infected leaves."),
}

def is_healthy(name: str) -> bool:
    return name in TREATMENT and TREATMENT[name][0] == "✅"


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADER  (CropSense backend — unchanged)
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    try:
        model = YOLO("runs/detect/train/weights/best.pt")
        return model
    except Exception as e:
        st.error(f"❌ Model Loading Failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE  (AgriCast pattern — unchanged)
# ═══════════════════════════════════════════════════════════════════════════════
for k, v in [("authenticated", False), ("username", ""), ("page", "Home"), ("show_signup", False)]:
    if k not in st.session_state:
        st.session_state[k] = v


# ═══════════════════════════════════════════════════════════════════════════════
#  GROQ CLIENT  (AgriCast — unchanged)
# ═══════════════════════════════════════════════════════════════════════════════
client = Groq(
    api_key=""
)


# ═══════════════════════════════════════════════════════════════════════════════
#  AUTH  (AgriCast — code unchanged, brand adapted to CropSense)
# ═══════════════════════════════════════════════════════════════════════════════
def show_auth_modal():
    st.markdown('<div style="height:3rem;"></div>', unsafe_allow_html=True)
    _, col, _ = st.columns([1, 1, 1])

    with col:
        if st.session_state.show_signup:
            st.markdown("""
            <div class="auth-card">
                <div class="auth-logo">
                    <div class="auth-logo-icon">🌿</div>
                    <div class="auth-logo-name">Crop<span>Sense</span></div>
                    <div class="auth-logo-sub">Create your free account</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            new_user  = st.text_input("Username",         key="su_user",  placeholder="Choose a username")
            new_pass  = st.text_input("Password",         type="password", key="su_pass",  placeholder="Minimum 6 characters")
            new_pass2 = st.text_input("Confirm password", type="password", key="su_pass2", placeholder="Repeat your password")
            st.markdown('<div class="spacer-sm"></div>', unsafe_allow_html=True)

            if st.button("Create Account →", key="signup_submit", use_container_width=True):
                if not new_user.strip():
                    st.error("Username cannot be empty.")
                elif len(new_user.strip()) < 3:
                    st.error("Username must be at least 3 characters.")
                elif not new_pass:
                    st.error("Password cannot be empty.")
                elif len(new_pass) < 6:
                    st.error("Password must be at least 6 characters.")
                elif new_pass != new_pass2:
                    st.error("Passwords do not match.")
                else:
                    if add_user(new_user.strip(), new_pass):
                        st.success("Account created! Please sign in.")
                        st.session_state.show_signup = False
                        st.rerun()
                    else:
                        st.error("Username already taken.")

            st.markdown('<div class="auth-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="auth-switch">Already have an account?</div>', unsafe_allow_html=True)
            st.markdown('<div class="spacer-sm"></div>', unsafe_allow_html=True)
            if st.button("Sign In Instead", key="back_login", use_container_width=True):
                st.session_state.show_signup = False
                st.rerun()

        else:
            st.markdown("""
            <div class="auth-card">
                <div class="auth-logo">
                    <div class="auth-logo-icon">🌿</div>
                    <div class="auth-logo-name">Crop<span>Sense</span></div>
                    <div class="auth-logo-sub">AI-Powered Plant Disease Diagnostics</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            username = st.text_input("Username", key="li_user", placeholder="Enter your username")
            password = st.text_input("Password", type="password", key="li_pass", placeholder="Enter your password")
            st.markdown('<div class="spacer-sm"></div>', unsafe_allow_html=True)

            if st.button("Sign In →", key="login_submit", use_container_width=True):
                if not username.strip() or not password:
                    st.error("Please enter both username and password.")
                elif authenticate_user(username.strip(), password):
                    st.session_state.authenticated = True
                    st.session_state.username = username.strip()
                    st.session_state.page = "Home"
                    st.rerun()
                else:
                    st.error("Invalid credentials. Please try again.")

            st.markdown('<div class="auth-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="auth-switch">New to CropSense?</div>', unsafe_allow_html=True)
            st.markdown('<div class="spacer-sm"></div>', unsafe_allow_html=True)
            if st.button("Create a Free Account", key="goto_signup", use_container_width=True):
                st.session_state.show_signup = True
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
#  NAVBAR  (AgriCast — unchanged, pages adapted for CropSense)
# ═══════════════════════════════════════════════════════════════════════════════
def render_navbar():
    pages = ["Home", "Diagnose", "Chatbot", "About", "Logout"]
    nav_left, nav_mid, nav_right = st.columns([1.5, 5.2, 0.6])

    with nav_left:
        st.markdown("""
        <div class="nav-brand">
            <div class="nav-brand-icon">🌿</div>
            <div>
                <div class="nav-brand-text">Crop<span>Sense</span></div>
                <div class="nav-brand-sub">Disease Intelligence</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with nav_mid:
        st.markdown('<div class="nav-btn-row">', unsafe_allow_html=True)
        nav_cols = st.columns([0.9, 1.35, 1.15, 1.0, 1.1], gap="small")

        for idx, page in enumerate(pages):
            with nav_cols[idx]:
                if page == "Logout":
                    if st.button("LOGOUT", key="logout_btn"):
                        st.session_state.authenticated = False
                        st.session_state.username = ""
                        st.session_state.page = "Home"
                        st.rerun()
                else:
                    if st.button(page.upper(), key=f"nav_{page}"):
                        st.session_state.page = page
                        st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    with nav_right:
        init = st.session_state.username[0].upper() if st.session_state.username else "?"
        st.markdown(
            f'<div class="nav-user-pill">'
            f'<div class="nav-avatar">{init}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown('<div class="g-divider"></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  GATE
# ═══════════════════════════════════════════════════════════════════════════════
if not st.session_state.authenticated:
    show_auth_modal()
    st.stop()

render_navbar()
model = load_model()
if model is None:
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
#  HOME
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "Home":

    st.markdown("""
    <div class="hero-root">
        <div class="hero-badge">AI-Powered Plant Disease Diagnostics</div>
        <div class="hero-title">Detect crop diseases with<br><em>AI precision</em></div>
        <p class="hero-sub">Upload a leaf photo and get instant disease detection, confidence scoring, treatment protocols, and severity analysis — powered by YOLOv8.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-label">Disease Classes</div>
            <div class="kpi-num">30+</div>
            <div class="kpi-desc">Tomato · Potato · Corn · Apple · Grape · Pepper — all major crop diseases covered.</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-label">Detection Model</div>
            <div class="kpi-num">YOLOv8</div>
            <div class="kpi-desc">State-of-the-art real-time object detection with bounding box inference on leaf images.</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-label">Treatment DB</div>
            <div class="kpi-num">IPM</div>
            <div class="kpi-desc">Integrated Pest Management protocols — fungicide, biological, and cultural controls.</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="spacer-md"></div>', unsafe_allow_html=True)

    col_l, col_r = st.columns([1.1, 1], gap="large")

    with col_l:
        st.markdown("""
        <div class="sec-card">
            <div class="sec-card-header">
                <div class="sec-icon">🚀</div>
                <div>
                    <div class="sec-title">What is CropSense?</div>
                    <div class="sec-subtitle">Platform overview</div>
                </div>
            </div>
            <p class="about-mission">
            CropSense uses YOLOv8 deep learning to identify plant diseases from leaf photographs in real time.
            Upload a single image and receive an annotated detection overlay, confidence score,
            evidence-based treatment recommendation, and pixel-level severity estimate — all in seconds.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        st.markdown("""
        <div class="sec-card" style="height:100%;">
            <div class="sec-card-header">
                <div class="sec-icon">✦</div>
                <div>
                    <div class="sec-title">Core Capabilities</div>
                    <div class="sec-subtitle">What's inside</div>
                </div>
            </div>
            <div class="pill-row">
                <span class="pill">🔬 Real-time YOLO Detection</span>
                <span class="pill">📊 Confidence Scoring</span>
                <span class="pill">💊 Treatment Protocols</span>
                <span class="pill">📈 Severity Analysis</span>
                <span class="pill">💬 AI Chatbot</span>
                <span class="pill">🌿 30+ Disease Classes</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  DIAGNOSE  (CropSense backend — fully preserved)
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "Diagnose":

    st.markdown("""
    <div class="page-header">
        <h1>Leaf Disease Diagnosis</h1>
        <p>Upload a clear leaf photograph and receive an AI-powered disease classification, treatment plan, and severity score.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sec-card">
        <div class="sec-card-header">
            <div class="sec-icon">📷</div>
            <div>
                <div class="sec-title">Upload Leaf Image</div>
                <div class="sec-subtitle">Supported formats: JPG · PNG · JPEG — high-resolution recommended</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    col_up, col_info = st.columns([3, 2], gap="large")

    with col_up:
        uploaded_file = st.file_uploader(
            "Upload leaf image",
            type=["jpg", "png", "jpeg"],
            label_visibility="collapsed",
        )

    with col_info:
        st.markdown('<div class="spacer-sm"></div>', unsafe_allow_html=True)
        for step, text in [
            ("01", "Upload a clear photo of the affected leaf"),
            ("02", "AI detects disease class and confidence level"),
            ("03", "Review treatment plan and severity score"),
        ]:
            st.markdown(f"""
            <div style="display:flex;align-items:flex-start;gap:14px;margin-bottom:1rem;">
                <div style="min-width:34px;height:34px;border-radius:8px;
                            background:var(--green-ghost);border:1px solid var(--border-3);
                            display:flex;align-items:center;justify-content:center;
                            font-size:0.7rem;font-weight:700;color:var(--green);
                            letter-spacing:0.05em;font-family:'Syne',sans-serif;">
                    {step}
                </div>
                <p style="color:var(--text-3);font-size:0.88rem;line-height:1.55;margin-top:5px;">{text}</p>
            </div>
            """, unsafe_allow_html=True)

    if uploaded_file is not None:
        st.markdown('<div class="g-divider"></div>', unsafe_allow_html=True)

        image     = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)

        # ── RUN MODEL (CropSense backend — unchanged) ──────────────────────
        results = model(img_array)

        detections = []
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    cls  = int(box.cls[0])
                    conf = float(box.conf[0])
                    name = model.names[cls]
                    detections.append({"name": name, "conf": conf})

        # Severity calc (unchanged)
        img_bgr         = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        gray            = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, thresh       = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
        infected_pixels = int(np.sum(thresh == 255))
        total_pixels    = thresh.size
        severity_pct    = (infected_pixels / total_pixels) * 100

        if severity_pct < 20:
            sev_level, sev_css, badge_css = "Low",      "sev-low",      "badge-low"
        elif severity_pct < 50:
            sev_level, sev_css, badge_css = "Moderate", "sev-moderate", "badge-moderate"
        else:
            sev_level, sev_css, badge_css = "High",     "sev-high",     "badge-high"

        res_plotted = results[0].plot()
        res_rgb     = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

        # ── TABS ──────────────────────────────────────────────────────────
        tab1, tab2, tab3 = st.tabs(
            ["🔬  Detection Results", "💊  Treatment Protocol", "📊  Severity Analysis"]
        )

        with tab1:
            st.markdown('<div class="spacer-sm"></div>', unsafe_allow_html=True)
            img_col, res_col = st.columns(2, gap="large")

            with img_col:
                st.markdown("""
                <div class="sec-card-header" style="border-bottom:none;padding-bottom:0;margin-bottom:0.8rem;">
                    <div class="sec-icon">📷</div>
                    <div><div class="sec-title">Uploaded Image</div></div>
                </div>
                """, unsafe_allow_html=True)
                st.image(image, use_container_width=True, caption="Original leaf photograph")

            with res_col:
                st.markdown("""
                <div class="sec-card-header" style="border-bottom:none;padding-bottom:0;margin-bottom:0.8rem;">
                    <div class="sec-icon">🤖</div>
                    <div>
                        <div class="sec-title">AI Annotations</div>
                        <div class="sec-subtitle">YOLOv8 detection overlay</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.image(res_rgb, use_container_width=True, caption="Detection overlay")

            st.markdown('<div class="spacer-sm"></div>', unsafe_allow_html=True)

            if not detections:
                st.markdown("""
                <div class="no-detect">
                    <span>⚠️</span>
                    <span>No disease detected — the leaf appears healthy, or the image may not be clear enough. Try a closer, well-lit photograph.</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                for d in detections:
                    name    = d["name"]
                    conf    = d["conf"]
                    healthy = is_healthy(name)
                    tag_cls  = "healthy-tag" if healthy else "disease-tag"
                    tag_icon = "✅" if healthy else "🔴"

                    st.markdown(f"""
                    <div class="detect-card">
                        <div class="{tag_cls}">{tag_icon} {"Healthy" if healthy else "Disease Detected"}</div>
                        <div class="disease-name-label">Detected Class</div>
                        <div class="disease-name-value">{name}</div>
                        <div class="conf-row">
                            <span class="conf-label">Confidence</span>
                            <div class="conf-bar-bg">
                                <div class="conf-bar-fill" style="width:{conf*100:.1f}%"></div>
                            </div>
                            <span class="conf-value">{conf*100:.1f}%</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        with tab2:
            st.markdown('<div class="spacer-sm"></div>', unsafe_allow_html=True)

            if not detections:
                st.markdown("""
                <div class="no-detect">
                    <span>⚠️</span>
                    <span>No detections found — treatment protocols require a successful disease classification.</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                for d in detections:
                    name = d["name"]
                    if name in TREATMENT:
                        icon, rec = TREATMENT[name]
                    else:
                        icon, rec = ("💊", "Maintain general plant hygiene: remove infected material, ensure good airflow and balanced nutrition.")

                    st.markdown(f"""
                    <div class="treatment-card">
                        <div class="treatment-icon">{icon}</div>
                        <div class="treatment-label">Recommended Treatment for
                            <strong style="color:var(--green)">{name}</strong>
                        </div>
                        <div class="treatment-text">{rec}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    with st.expander("🌱 General IPM Tips"):
                        st.markdown("""
                        <div class="tip-box"><span class="tip-icon">💧</span>
                        <span class="tip-text"><strong>Irrigation:</strong> Water at the base of plants early in the morning; avoid wetting foliage to reduce fungal spread.</span></div>
                        <div class="tip-box"><span class="tip-icon">✂️</span>
                        <span class="tip-text"><strong>Sanitation:</strong> Remove and destroy infected leaves. Disinfect cutting tools between plants with 70% alcohol.</span></div>
                        <div class="tip-box"><span class="tip-icon">🔄</span>
                        <span class="tip-text"><strong>Crop rotation:</strong> Rotate with unrelated crops every season to break pathogen cycles in the soil.</span></div>
                        <div class="tip-box"><span class="tip-icon">🔬</span>
                        <span class="tip-text"><strong>Monitoring:</strong> Scout fields weekly. Early detection dramatically reduces treatment cost and crop loss.</span></div>
                        """, unsafe_allow_html=True)

        with tab3:
            st.markdown('<div class="spacer-sm"></div>', unsafe_allow_html=True)
            s_col, e_col = st.columns([3, 2], gap="large")

            with s_col:
                st.markdown(f"""
                <div class="severity-card">
                    <div class="sec-card-header" style="border-bottom:none;padding-bottom:0;margin-bottom:0.3rem;">
                        <div class="sec-icon">📊</div>
                        <div>
                            <div class="sec-title">Severity Score</div>
                            <div class="sec-subtitle">Pixel-intensity analysis</div>
                        </div>
                    </div>
                    <div style="color:var(--text-3);font-size:0.78rem;margin-bottom:0.3rem;">
                        Estimated proportion of leaf tissue showing abnormal colouration
                    </div>
                    <div class="severity-meter-bg">
                        <div class="severity-level-{sev_level.lower()}"
                             style="width:{min(severity_pct,100):.1f}%;height:100%;border-radius:100px;">
                        </div>
                    </div>
                    <div class="severity-stats">
                        <span class="severity-percent {sev_css}">{severity_pct:.1f}%</span>
                        <span class="severity-badge {badge_css}">{sev_level} Severity</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with e_col:
                st.markdown("""
                <div class="sec-card-header" style="border-bottom:none;padding-bottom:0;margin-bottom:0.8rem;">
                    <div class="sec-icon">📐</div>
                    <div><div class="sec-title">Scale Reference</div></div>
                </div>
                """, unsafe_allow_html=True)
                for pct, lvl, color, desc in [
                    ("0 – 20%",   "Low",      "#7CFF9B", "Minimal infection; early intervention likely sufficient."),
                    ("20 – 50%",  "Moderate", "#FFC857", "Significant spread; prompt treatment recommended."),
                    ("50 – 100%", "High",     "#F87171", "Severe infection; aggressive management required."),
                ]:
                    st.markdown(f"""
                    <div style="display:flex;gap:12px;align-items:flex-start;margin-bottom:1rem;">
                        <div style="min-width:10px;height:10px;border-radius:50%;background:{color};margin-top:5px;flex-shrink:0;box-shadow:0 0 6px {color}40;"></div>
                        <div>
                            <span style="font-size:0.78rem;font-weight:700;color:{color};font-family:'Syne',sans-serif;">{pct} — {lvl}</span><br>
                            <span style="font-size:0.78rem;color:var(--text-3);line-height:1.5;">{desc}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("""
            <div class="tip-box" style="margin-top:0.5rem">
                <span class="tip-icon">ℹ️</span>
                <span class="tip-text">
                    <strong>Methodology note:</strong> Severity is estimated by pixel-intensity thresholding on the uploaded image.
                    For clinical-grade assessments, combine with ground-truth lab diagnostics or consult an agronomist.
                </span>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  CHATBOT  (AgriCast — unchanged, system prompt adapted for CropSense)
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "Chatbot":

    st.markdown("""
    <div class="page-header">
        <h1>Agri AI Assistant</h1>
        <p>Ask anything about crop diseases, treatment protocols, soil health, fertilizers, or best farming practices.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sec-card">
        <div class="sec-card-header">
            <div class="sec-icon">🤖</div>
            <div>
                <div class="sec-title">Ask the AI Expert</div>
                <div class="sec-subtitle">Powered by Llama 3.1 · Plant pathology &amp; agronomy knowledge base</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    user_question = st.text_area(
        "Your question",
        height=140,
        placeholder="e.g. What fungicide works best for tomato late blight? How do I identify powdery mildew on squash? What is the ideal spray schedule for apple scab?"
    )

    st.markdown('<div class="spacer-sm"></div>', unsafe_allow_html=True)

    if st.button("⟢  Ask AI Expert", use_container_width=True):
        if user_question.strip():
            with st.spinner("Thinking..."):
                try:
                    completion = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are CropSense AI — an expert in plant pathology, crop diseases, "
                                    "fungicide and pesticide protocols, and integrated pest management. "
                                    "Give concise, practical, evidence-based farming advice. "
                                    "Focus on actionable treatment steps and preventive measures."
                                )
                            },
                            {"role": "user", "content": user_question}
                        ],
                        temperature=0.7,
                        max_tokens=500
                    )
                    answer = completion.choices[0].message.content

                    st.markdown(f"""
                    <div class="chat-response">
                        <div class="chat-resp-label">AI Response</div>
                        <div class="chat-resp-text">{answer}</div>
                    </div>
                    """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Please enter a question before submitting.")


# ═══════════════════════════════════════════════════════════════════════════════
#  ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "About":

    st.markdown("""
    <div class="page-header">
        <h1>About CropSense</h1>
        <p>Mission, technology, and the platform behind AI-powered plant disease diagnostics.</p>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_r = st.columns([1.1, 1], gap="large")

    with col_l:
        st.markdown("""
        <div class="sec-card">
            <div class="sec-card-header">
                <div class="sec-icon">🌿</div>
                <div>
                    <div class="sec-title">Our Mission</div>
                    <div class="sec-subtitle">Why we built CropSense</div>
                </div>
            </div>
            <p class="about-mission">
            CropSense is an AI-powered plant disease diagnostic platform that combines state-of-the-art
            computer vision, curated treatment databases, and large-language-model advisory to help
            farmers identify diseases early and respond with precision — reducing crop loss and
            minimising unnecessary chemical use.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="spacer-sm"></div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="sec-card">
            <div class="sec-card-header">
                <div class="sec-icon">📞</div>
                <div>
                    <div class="sec-title">Contact &amp; Support</div>
                    <div class="sec-subtitle">Get in touch with the team</div>
                </div>
            </div>
            <p class="about-mission">
            For support and inquiries, contact the CropSense team through official channels.
            We continuously expand disease coverage, add new crops, and improve model accuracy.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        st.markdown("""
        <div class="sec-card" style="height:100%;">
            <div class="sec-card-header">
                <div class="sec-icon">✨</div>
                <div>
                    <div class="sec-title">Key Features</div>
                    <div class="sec-subtitle">Platform capabilities</div>
                </div>
            </div>
            <div class="pill-row">
                <span class="pill">🔬 Real-time YOLO Detection</span>
                <span class="pill">📊 Confidence Scoring</span>
                <span class="pill">💊 IPM Treatment Database</span>
                <span class="pill">📈 Pixel Severity Analysis</span>
                <span class="pill">🤖 YOLOv8 ML Engine</span>
                <span class="pill">🌿 30+ Disease Classes</span>
                <span class="pill">💬 Groq AI Chatbot</span>
                <span class="pill">🔐 Secure Auth</span>
            </div>
            <div class="spacer-md"></div>
            <div class="sec-card-header" style="margin-top:1rem;">
                <div class="sec-icon">📐</div>
                <div>
                    <div class="sec-title">Tech Stack</div>
                    <div class="sec-subtitle">Built with</div>
                </div>
            </div>
            <div class="pill-row">
                <span class="pill">Python</span>
                <span class="pill">Streamlit</span>
                <span class="pill">YOLOv8 / Ultralytics</span>
                <span class="pill">Groq API</span>
                <span class="pill">OpenCV</span>
                <span class="pill">NumPy</span>
                <span class="pill">Pillow</span>
            </div>
        </div>
        """, unsafe_allow_html=True)