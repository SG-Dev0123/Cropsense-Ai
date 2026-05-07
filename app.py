# (imports same)
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# ✅ MUST BE FIRST
st.set_page_config(
    page_title="CropSense AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# 🌿 UPDATED CLASSY NATURE CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [data-testid="stAppViewContainer"]{
    background:#F4F8F2 !important;
    color:#2F3E2F !important;
    font-family:'DM Sans',sans-serif;
}

/* Remove Streamlit junk */
#MainMenu, footer, header, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stSidebarNav"] {
    display:none !important;
}

/* Main container */
.main .block-container{
    padding:2rem 3rem;
    max-width:1200px;
}

/* 🌿 HERO */
.hero{
    text-align:center;
    padding:2rem 1rem;
}
.hero h1{
    font-family:'Playfair Display',serif;
    font-size:3rem;
    color:#2F3E2F;
}
.hero p{
    color:#7A8B7A;
}

/* 🌿 SECTION HEADINGS */
.section-heading h3{
    color:#2F3E2F;
}

/* 🌿 CARDS */
.result-card, .treatment-card, .severity-card{
    background:#FFFFFF !important;
    border:1px solid #DCE8D6;
    border-radius:16px;
    padding:1.5rem;
    margin-bottom:1rem;
    box-shadow:0 4px 12px rgba(0,0,0,0.05);
}

/* Hover effect */
.result-card:hover, .treatment-card:hover{
    transform:translateY(-2px);
    box-shadow:0 8px 20px rgba(0,0,0,0.08);
}

/* 🌿 LABELS */
.disease-name-label,
.conf-label{
    color:#7A8B7A;
}

/* 🌿 TEXT */
.disease-name-value{
    color:#2F3E2F;
    font-weight:600;
}
.treatment-text{
    color:#2F3E2F;
}

/* 🌿 TAGS */
.healthy-tag{
    background:#E6F4EA;
    border:1px solid #6AA84F;
    color:#2F7A3D;
}
.disease-tag{
    background:#FDECEA;
    border:1px solid #E57373;
    color:#C62828;
}

/* 🌿 CONFIDENCE BAR */
.conf-bar-bg{
    background:#EAF3E5;
}
.conf-bar-fill{
    background:linear-gradient(90deg,#6AA84F,#93C47D);
}
.conf-value{
    color:#6AA84F;
}

/* 🌿 SEVERITY */
.severity-level-low{
    background:linear-gradient(90deg,#6AA84F,#93C47D);
}
.severity-level-moderate{
    background:linear-gradient(90deg,#FBC02D,#FFD54F);
}
.severity-level-high{
    background:linear-gradient(90deg,#E57373,#EF5350);
}

/* 🌿 UPLOADER */
[data-testid="stFileUploader"] > div{
    background:#FFFFFF !important;
    border:2px dashed #93C47D !important;
    border-radius:12px;
}
[data-testid="stFileUploader"] > div:hover{
    border-color:#6AA84F !important;
}

/* 🌿 BUTTON */
button{
    background:#6AA84F !important;
    color:white !important;
    border-radius:8px !important;
}

/* 🌿 TABS */
[data-testid="stTabs"] button{
    color:#7A8B7A;
}
[data-testid="stTabs"] button[aria-selected="true"]{
    color:#6AA84F;
    border-bottom:2px solid #6AA84F;
}

/* 🌿 FOOTER */
.footer{
    color:#7A8B7A;
    text-align:center;
    margin-top:2rem;
}

/* Scrollbar */
::-webkit-scrollbar-thumb{
    background:#93C47D;
}
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
# ─── MODEL LOADER (FIXED) ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model = YOLO("runs/detect/train/weights/best.pt")   # make sure best.pt is in same folder
        return model
    except Exception as e:
        st.error(f"❌ Model Loading Failed: {e}")
        return None

model = load_model()

# 🚨 STOP APP IF MODEL NOT LOADED (prevents blank screen)
if model is None:
    st.stop()

st.success("✅ Model Loaded Successfully")

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
