import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CropSense AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CUSTOM CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0d1117;
    color: #e8f5e8;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse at 20% 10%, rgba(34,197,94,0.07) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 80%, rgba(16,185,129,0.05) 0%, transparent 50%),
        #0d1117;
}

/* Hide default streamlit chrome */
#MainMenu, footer, header, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stSidebarNav"] { display: none !important; }

/* Remove default block padding */
[data-testid="block-container"] { padding: 0 !important; }
.main .block-container { padding: 2rem 3rem 4rem !important; max-width: 1200px; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #22c55e44; border-radius: 3px; }

/* ── Hero Header ── */
.hero {
    text-align: center;
    padding: 3.5rem 1rem 2.5rem;
    position: relative;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(34,197,94,0.12);
    border: 1px solid rgba(34,197,94,0.3);
    color: #4ade80;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 6px 16px;
    border-radius: 100px;
    margin-bottom: 1.4rem;
}
.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.4rem, 5vw, 4rem);
    font-weight: 900;
    line-height: 1.1;
    background: linear-gradient(135deg, #bbf7d0 0%, #4ade80 45%, #86efac 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.9rem;
}
.hero p {
    color: #94a3b8;
    font-size: 1.05rem;
    font-weight: 300;
    max-width: 520px;
    margin: 0 auto;
    line-height: 1.7;
}
.hero-line {
    width: 60px;
    height: 3px;
    background: linear-gradient(90deg, #22c55e, #4ade80);
    border-radius: 2px;
    margin: 1.8rem auto 0;
}

/* ── Divider ── */
.section-divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(34,197,94,0.25), transparent);
    margin: 2rem 0;
}

/* ── Upload Zone ── */
.upload-wrapper {
    background: linear-gradient(135deg, rgba(22,27,34,0.9), rgba(15,20,27,0.95));
    border: 2px dashed rgba(34,197,94,0.3);
    border-radius: 20px;
    padding: 2.5rem;
    text-align: center;
    transition: border-color 0.3s;
    margin-bottom: 1.5rem;
}
.upload-wrapper:hover { border-color: rgba(34,197,94,0.6); }
.upload-icon { font-size: 2.8rem; margin-bottom: 0.8rem; }
.upload-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem;
    color: #bbf7d0;
    margin-bottom: 0.4rem;
}
.upload-sub { color: #64748b; font-size: 0.85rem; }

/* Streamlit file uploader override */
[data-testid="stFileUploader"] {
    background: transparent !important;
    border: none !important;
}
[data-testid="stFileUploader"] > div {
    background: rgba(22,27,34,0.8) !important;
    border: 1.5px dashed rgba(34,197,94,0.35) !important;
    border-radius: 16px !important;
    padding: 1.8rem !important;
    transition: all 0.3s ease !important;
}
[data-testid="stFileUploader"] > div:hover {
    border-color: rgba(34,197,94,0.65) !important;
    background: rgba(22,27,34,1) !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] {
    color: #94a3b8 !important;
}

/* ── Section Headings ── */
.section-heading {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 1.2rem;
}
.section-heading h3 {
    font-family: 'Playfair Display', serif;
    font-size: 1.25rem;
    color: #e8f5e8;
    font-weight: 700;
}
.section-pill {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 100px;
    background: rgba(34,197,94,0.15);
    color: #4ade80;
    border: 1px solid rgba(34,197,94,0.3);
}

/* ── Result Card ── */
.result-card {
    background: linear-gradient(135deg, rgba(22,27,34,0.95), rgba(15,20,27,0.9));
    border: 1px solid rgba(34,197,94,0.2);
    border-radius: 18px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
}
.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #22c55e, #4ade80, #86efac);
}

/* ── Detection Tag ── */
.disease-tag {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(34,197,94,0.1);
    border: 1px solid rgba(34,197,94,0.35);
    color: #4ade80;
    font-size: 1rem;
    font-weight: 600;
    padding: 8px 18px;
    border-radius: 100px;
    margin-bottom: 1rem;
}
.healthy-tag {
    background: rgba(16,185,129,0.1);
    border-color: rgba(16,185,129,0.35);
    color: #34d399;
}
.disease-name-label {
    color: #64748b;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.disease-name-value {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #e8f5e8;
    margin-bottom: 0.5rem;
}

/* ── Confidence Bar ── */
.conf-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-top: 0.8rem;
}
.conf-label { color: #64748b; font-size: 0.82rem; min-width: 85px; }
.conf-bar-bg {
    flex: 1;
    height: 7px;
    background: rgba(255,255,255,0.07);
    border-radius: 100px;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #22c55e, #4ade80);
    transition: width 0.6s ease;
}
.conf-value {
    font-size: 0.82rem;
    font-weight: 600;
    color: #4ade80;
    min-width: 38px;
    text-align: right;
}

/* ── Treatment Card ── */
.treatment-card {
    background: linear-gradient(135deg, rgba(16,185,129,0.07), rgba(22,27,34,0.95));
    border: 1px solid rgba(16,185,129,0.2);
    border-radius: 18px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
}
.treatment-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #10b981, #34d399, #6ee7b7);
}
.treatment-icon { font-size: 2rem; margin-bottom: 0.6rem; }
.treatment-label {
    color: #64748b;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.treatment-text {
    font-size: 1.05rem;
    color: #d1fae5;
    font-weight: 500;
    line-height: 1.6;
}

/* ── Severity Card ── */
.severity-card {
    background: linear-gradient(135deg, rgba(22,27,34,0.95), rgba(15,20,27,0.9));
    border: 1px solid rgba(251,191,36,0.2);
    border-radius: 18px;
    padding: 1.6rem 1.8rem;
    position: relative;
    overflow: hidden;
}
.severity-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #f59e0b, #fbbf24, #fde68a);
}
.severity-meter-bg {
    width: 100%;
    height: 12px;
    background: rgba(255,255,255,0.07);
    border-radius: 100px;
    overflow: hidden;
    margin: 1rem 0 0.5rem;
}
.severity-level-low   { background: linear-gradient(90deg, #22c55e, #4ade80); }
.severity-level-moderate { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
.severity-level-high  { background: linear-gradient(90deg, #ef4444, #f87171); }

.severity-stats {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 0.5rem;
}
.severity-percent {
    font-family: 'Playfair Display', serif;
    font-size: 2.2rem;
    font-weight: 900;
    line-height: 1;
}
.sev-low   { color: #4ade80; }
.sev-moderate { color: #fbbf24; }
.sev-high  { color: #f87171; }

.severity-badge {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 5px 14px;
    border-radius: 100px;
}
.badge-low      { background: rgba(34,197,94,0.15); color: #4ade80; border: 1px solid rgba(34,197,94,0.3); }
.badge-moderate { background: rgba(251,191,36,0.12); color: #fbbf24; border: 1px solid rgba(251,191,36,0.3); }
.badge-high     { background: rgba(239,68,68,0.12); color: #f87171; border: 1px solid rgba(239,68,68,0.3); }

/* ── Info Tip ── */
.tip-box {
    background: rgba(34,197,94,0.06);
    border: 1px solid rgba(34,197,94,0.18);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    display: flex;
    gap: 10px;
    align-items: flex-start;
    margin-top: 0.8rem;
}
.tip-icon { font-size: 1rem; margin-top: 1px; }
.tip-text { color: #94a3b8; font-size: 0.85rem; line-height: 1.6; }
.tip-text strong { color: #86efac; }

/* ── Tabs override ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: rgba(22,27,34,0.7) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    border: 1px solid rgba(34,197,94,0.15) !important;
    gap: 2px !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important;
    color: #64748b !important;
    border-radius: 9px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    padding: 8px 18px !important;
    transition: all 0.2s !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: rgba(34,197,94,0.15) !important;
    color: #4ade80 !important;
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
    background: transparent !important;
}
[data-testid="stTabs"] [data-baseweb="tab-border"] {
    display: none !important;
}

/* ── Expander override ── */
[data-testid="stExpander"] {
    background: rgba(22,27,34,0.7) !important;
    border: 1px solid rgba(34,197,94,0.2) !important;
    border-radius: 14px !important;
}
[data-testid="stExpanderToggleIcon"] { color: #4ade80 !important; }

/* ── Image captions ── */
[data-testid="stImage"] > div > div {
    color: #4a5568 !important;
    font-size: 0.78rem !important;
}

/* ── No detection banner ── */
.no-detect {
    background: rgba(239,68,68,0.07);
    border: 1px solid rgba(239,68,68,0.2);
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    color: #fca5a5;
    display: flex;
    gap: 10px;
    align-items: center;
    font-size: 0.95rem;
}

/* ── Footer ── */
.footer {
    text-align: center;
    margin-top: 4rem;
    padding-top: 2rem;
    border-top: 1px solid rgba(34,197,94,0.1);
    color: #334155;
    font-size: 0.8rem;
    letter-spacing: 0.05em;
}

/* ── Columns spacing ── */
[data-testid="stHorizontalBlock"] { gap: 1.5rem !important; }

/* ── st.image border ── */
[data-testid="stImage"] img {
    border-radius: 14px !important;
    border: 1px solid rgba(34,197,94,0.15) !important;
}

/* ── st.columns ── */
.stColumn { padding: 0 !important; }
</style>
""", unsafe_allow_html=True)

# ─── TREATMENT DATABASE ──────────────────────────────────────────────────────────
TREATMENT = {
    "Apple Scab Leaf":                    ("🍎", "Apply Captan or Mancozeb fungicide every 7–10 days during wet seasons. Remove fallen leaves to reduce spore load."),
    "Apple leaf":                         ("✅", "Healthy leaf — maintain balanced fertilisation and adequate irrigation. Monitor regularly for early signs of disease."),
    "Apple rust leaf":                    ("🍎", "Apply sulfur-based or myclobutanil fungicide at bud break. Remove nearby juniper hosts to break the disease cycle."),
    "Bell_pepper leaf":                   ("✅", "Healthy plant — ensure consistent watering and well-drained soil. Avoid overhead irrigation."),
    "Bell_pepper leaf spot":              ("🫑", "Spray copper-based fungicide (e.g., Copper Oxychloride). Remove infected leaves and avoid splashing water."),
    "Blueberry leaf":                     ("✅", "Healthy plant — maintain soil pH 4.5–5.5 and mulch around roots to retain moisture."),
    "Cherry leaf":                        ("✅", "Healthy leaf — no immediate action needed. Prune for airflow and inspect seasonally."),
    "Corn Gray leaf spot":                ("🌽", "Apply foliar fungicide such as Azoxystrobin. Rotate crops annually and use resistant hybrids."),
    "Corn leaf blight":                   ("🌽", "Use resistant varieties and apply propiconazole fungicide at disease onset. Ensure proper field drainage."),
    "Corn rust leaf":                     ("🌽", "Apply fungicide (Triazole group) early. Monitor weather conditions and use rust-resistant cultivars."),
    "Peach leaf":                         ("✅", "Healthy leaf — regular pruning and balanced nutrition help prevent future infections."),
    "Potato leaf":                        ("✅", "Healthy plant — hill soil around base, water consistently and scout for early blight symptoms."),
    "Potato leaf early blight":           ("🥔", "Apply Mancozeb or Chlorothalonil spray every 7 days. Remove and destroy heavily infected foliage."),
    "Potato leaf late blight":            ("🥔", "Apply Chlorothalonil or metalaxyl-based fungicide immediately. Destroy infected plants to halt rapid spread."),
    "Raspberry leaf":                     ("✅", "Healthy leaf — ensure canes are well-spaced for airflow; remove old canes after harvest."),
    "Soyabean leaf":                      ("✅", "Healthy plant — maintain row spacing for airflow and monitor for soybean rust during humid weather."),
    "Soybean leaf":                       ("✅", "Healthy plant — use certified seeds and scout fields regularly during vegetative stages."),
    "Squash Powdery mildew leaf":         ("🎃", "Apply sulfur spray or potassium bicarbonate. Improve air circulation and avoid excess nitrogen fertiliser."),
    "Strawberry leaf":                    ("✅", "Healthy plant — replace beds every 2–3 years and keep mulch fresh to reduce soil-borne diseases."),
    "Tomato Early blight leaf":           ("🍅", "Apply Neem oil or copper fungicide every 7 days. Remove infected lower leaves and mulch to prevent soil splash."),
    "Tomato Septoria leaf spot":          ("🍅", "Apply chlorothalonil or copper fungicide. Remove infected leaves promptly and stake plants to improve airflow."),
    "Tomato leaf":                        ("✅", "Healthy leaf — maintain consistent moisture, calcium supply and scout for pests weekly."),
    "Tomato leaf bacterial spot":         ("🍅", "Apply copper-based bactericide. Avoid overhead watering and use disease-free transplants."),
    "Tomato leaf late blight":            ("🍅", "Apply fungicide (Mancozeb or Cymoxanil) immediately. Destroy infected plant material; do not compost."),
    "Tomato leaf mosaic virus":           ("🍅", "No chemical cure — remove and destroy infected plants promptly. Disinfect tools and control aphid vectors."),
    "Tomato leaf yellow virus":           ("🍅", "Control whitefly populations with insecticidal soap or neem oil. Use reflective mulches and resistant varieties."),
    "Tomato mold leaf":                   ("🍅", "Improve greenhouse ventilation. Apply fungicide (chlorothalonil). Reduce humidity and avoid leaf wetness."),
    "Tomato two spotted spider mites leaf": ("🍅", "Use insecticidal soap, neem oil, or predatory mites (Phytoseiidae). Avoid broad-spectrum insecticides."),
    "grape leaf":                         ("✅", "Healthy leaf — prune for open canopy, monitor for downy/powdery mildew during humid periods."),
    "grape leaf black rot":               ("🍇", "Apply Myclobutanil or Mancozeb fungicide from bud break. Remove mummified berries and infected leaves."),
}

def is_healthy(name: str) -> bool:
    return name in TREATMENT and TREATMENT[name][0] == "✅"

# ─── MODEL LOADER ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ─── HERO ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">🌿 AI-Powered Diagnostics</div>
    <h1>CropSense Advisor</h1>
    <p>Upload a leaf photograph and get instant disease detection, confidence scoring, treatment protocols, and severity analysis.</p>
    <div class="hero-line"></div>
</div>
""", unsafe_allow_html=True)

# ─── UPLOAD ──────────────────────────────────────────────────────────────────────
col_up, col_info = st.columns([3, 2], gap="large")

with col_up:
    uploaded_file = st.file_uploader(
        "Upload a high-resolution leaf image for best results",
        type=["jpg", "png", "jpeg"],
        label_visibility="collapsed",
    )

with col_info:
    st.markdown("""
    <div style="padding: 1.2rem 0;">
        <div class="section-heading">
            <h3>How it works</h3>
            <span class="section-pill">3 steps</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    for step, text in [
        ("01", "Upload a clear photo of the affected leaf"),
        ("02", "AI detects disease class and confidence level"),
        ("03", "Review treatment plan and severity score"),
    ]:
        st.markdown(f"""
        <div style="display:flex;align-items:flex-start;gap:14px;margin-bottom:1rem;">
            <div style="min-width:34px;height:34px;border-radius:8px;
                        background:rgba(34,197,94,0.12);border:1px solid rgba(34,197,94,0.3);
                        display:flex;align-items:center;justify-content:center;
                        font-size:0.7rem;font-weight:700;color:#4ade80;letter-spacing:0.05em;">
                {step}
            </div>
            <p style="color:#94a3b8;font-size:0.88rem;line-height:1.55;margin-top:5px;">{text}</p>
        </div>
        """, unsafe_allow_html=True)

# ─── ANALYSIS ────────────────────────────────────────────────────────────────────
if uploaded_file is not None:
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    # Run model
    results = model(img_array)

    # Gather detections
    detections = []
    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                cls  = int(box.cls[0])
                conf = float(box.conf[0])
                name = model.names[cls]
                detections.append({"name": name, "conf": conf})

    # Severity calc
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
    infected_pixels = int(np.sum(thresh == 255))
    total_pixels    = thresh.size
    severity_pct    = (infected_pixels / total_pixels) * 100

    if severity_pct < 20:
        sev_level, sev_css, badge_css = "Low",      "sev-low",      "badge-low"
    elif severity_pct < 50:
        sev_level, sev_css, badge_css = "Moderate", "sev-moderate", "badge-moderate"
    else:
        sev_level, sev_css, badge_css = "High",     "sev-high",     "badge-high"

    # Detection overlay image
    res_plotted = results[0].plot()
    res_rgb     = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

    # ── TABS ─────────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["🔬  Detection Results", "💊  Treatment Protocol", "📊  Severity Analysis"])

    # ── TAB 1: Detection ─────────────────────────────────────────────────────────
    with tab1:
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        img_col, res_col = st.columns(2, gap="large")

        with img_col:
            st.markdown("""
            <div class="section-heading">
                <h3>Uploaded Image</h3>
            </div>
            """, unsafe_allow_html=True)
            st.image(image, use_container_width=True, caption="Original leaf photograph")

        with res_col:
            st.markdown("""
            <div class="section-heading">
                <h3>AI Annotations</h3>
                <span class="section-pill">YOLO v8</span>
            </div>
            """, unsafe_allow_html=True)
            st.image(res_rgb, use_container_width=True, caption="Detection overlay")

        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

        if not detections:
            st.markdown("""
            <div class="no-detect">
                <span>⚠️</span>
                <span>No disease detected — the leaf appears healthy, or the image may not be clear enough. Try a closer, well-lit photograph.</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            for d in detections:
                name = d["name"]
                conf = d["conf"]
                healthy = is_healthy(name)
                tag_cls = "healthy-tag" if healthy else "disease-tag"
                tag_icon = "✅" if healthy else "🔴"

                st.markdown(f"""
                <div class="result-card">
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

    # ── TAB 2: Treatment ─────────────────────────────────────────────────────────
    with tab2:
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

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
                    icon, rec = "💊", "Maintain general plant hygiene: remove infected material, ensure good airflow and balanced nutrition."

                st.markdown(f"""
                <div class="treatment-card">
                    <div class="treatment-icon">{icon}</div>
                    <div class="treatment-label">Recommended Treatment for <strong style="color:#86efac">{name}</strong></div>
                    <div class="treatment-text">{rec}</div>
                </div>
                """, unsafe_allow_html=True)

                # General tips expander
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

    # ── TAB 3: Severity ──────────────────────────────────────────────────────────
    with tab3:
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        s_col, e_col = st.columns([3, 2], gap="large")

        with s_col:
            st.markdown(f"""
            <div class="severity-card">
                <div class="section-heading" style="margin-bottom:0.5rem">
                    <h3>Severity Score</h3>
                    <span class="section-pill">Pixel Analysis</span>
                </div>
                <div style="color:#64748b;font-size:0.82rem;margin-bottom:0.5rem;">
                    Estimated proportion of leaf tissue showing abnormal colouration
                </div>
                <div class="severity-meter-bg">
                    <div class="severity-level-{sev_level.lower()}" style="width:{min(severity_pct,100):.1f}%;height:100%;border-radius:100px;"></div>
                </div>
                <div class="severity-stats">
                    <span class="severity-percent {sev_css}">{severity_pct:.1f}%</span>
                    <span class="severity-badge {badge_css}">{sev_level} Severity</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with e_col:
            st.markdown("""
            <div style="padding: 0.5rem 0;">
                <div class="section-heading"><h3>Scale Reference</h3></div>
            </div>
            """, unsafe_allow_html=True)
            for pct, lvl, color, desc in [
                ("0 – 20%",  "Low",      "#4ade80", "Minimal infection; early intervention likely sufficient."),
                ("20 – 50%", "Moderate", "#fbbf24", "Significant spread; prompt treatment recommended."),
                ("50 – 100%","High",     "#f87171", "Severe infection; aggressive management required."),
            ]:
                st.markdown(f"""
                <div style="display:flex;gap:12px;align-items:flex-start;margin-bottom:1rem;">
                    <div style="min-width:10px;height:10px;border-radius:50%;
                                background:{color};margin-top:5px;flex-shrink:0;"></div>
                    <div>
                        <span style="font-size:0.8rem;font-weight:700;color:{color};">{pct} — {lvl}</span><br>
                        <span style="font-size:0.8rem;color:#64748b;line-height:1.5;">{desc}</span>
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

# ─── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    CropSense AI &nbsp;·&nbsp; Powered by YOLOv8 &nbsp;·&nbsp;
    For advisory purposes only — consult a certified agronomist for field decisions
</div>
""", unsafe_allow_html=True)
