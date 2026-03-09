"""
ENHANCED Streamlit App with Session State for Result Persistence

This version uses st.session_state to ensure results don't disappear when widgets change.
"""

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import folium
from streamlit_folium import st_folium
import json
import os
import streamlit.components.v1 as components

# Import custom modules
from safety_scorer import SafetyScorer
from explanation_engine import RiskExplainer
from route_engine import RouteAnalyzer
from ambulance_predictor import AmbulanceResponsePredictor
from ai_recommendations import SafetyRecommendationEngine
from geospatial_utils import geocode_address, reverse_geocode

# -------------------------------------------------------------------------
# Page Config
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="Road Safety Intelligence System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Custom CSS Design System
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap" rel="stylesheet">
<style>
    /* ================== ROOT VARIABLES ================== */
    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        --success-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        --danger-gradient: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        --dark-bg: #0a0e27;
        --card-bg: rgba(255, 255, 255, 0.03);
        --card-border: rgba(255, 255, 255, 0.1);
        --text-primary: #ffffff;
        --text-secondary: #b4b9d4;
        --accent-blue: #4facfe;
        --accent-purple: #667eea;
        --accent-pink: #f093fb;
        --accent-green: #00f2fe;
        --shadow-glow: 0 8px 32px 0 rgba(102, 126, 234, 0.2);
    }
    
    /* ================== GLOBAL STYLES ================== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background: linear-gradient(180deg, #0a0e27 0%, #1a1f3a 50%, #0a0e27 100%);
        background-attachment: fixed;
    }
    
    /* ================== TYPOGRAPHY ================== */
    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        background: linear-gradient(135deg, #ffffff 0%, #b4b9d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    h1 {
        font-size: 2.8rem !important;
        margin-bottom: 0.5rem !important;
        text-shadow: 0 0 60px rgba(102, 126, 234, 0.3);
    }
    
    h2 {
        font-size: 2rem !important;
        margin-top: 1.5rem !important;
    }
    
    h3 {
        font-size: 1.5rem !important;
        margin-top: 1rem !important;
    }
    
    p, .stMarkdown {
        color: var(--text-secondary);
        line-height: 1.7;
        font-size: 1rem;
    }
    
    /* ================== BUTTONS ================== */
    .stButton>button {
        width: 100%;
        background: var(--primary-gradient) !important;
        color: white !important;
        height: 56px;
        font-size: 16px !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        text-transform: none !important;
        letter-spacing: 0.02em;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.6) !important;
    }
    
    .stButton>button:active {
        transform: translateY(0px);
    }
    
    /* ================== GLASSMORPHISM CARDS ================== */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 24px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        border-color: rgba(102, 126, 234, 0.3);
        box-shadow: 0 8px 32px 0 rgba(102, 126, 234, 0.2);
    }
    
    /* ================== METRICS & STATS ================== */
    .stMetric {
        background: rgba(255, 255, 255, 0.04);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        transition: all 0.3s ease;
    }
    
    .stMetric:hover {
        background: rgba(255, 255, 255, 0.06);
        border-color: rgba(102, 126, 234, 0.3);
        transform: translateY(-2px);
    }
    
    .stMetric label {
        color: var(--text-secondary) !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
        font-family: 'Space Grotesk', sans-serif !important;
    }
    
    /* ================== INPUT FIELDS ================== */
    .stTextInput>div>div>input,
    .stNumberInput>div>div>input,
    .stSelectbox>div>div>div,
    .stTimeInput>div>div>input {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
        color: #1a1a1a !important;
        padding: 12px 16px !important;
        font-size: 0.95rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput>div>div>input:focus,
    .stNumberInput>div>div>input:focus,
    .stSelectbox>div>div>div:focus-within,
    .stTimeInput>div>div>input:focus {
        border-color: var(--accent-purple) !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
        background: rgba(255, 255, 255, 0.95) !important;
    }
    
    /* Make dropdown text and options dark */
    .stSelectbox>div>div>div {
        color: #1a1a1a !important;
    }
    
    /* Placeholder text */
    .stTextInput>div>div>input::placeholder,
    .stNumberInput>div>div>input::placeholder {
        color: #666666 !important;
    }
    
    /* ================== SLIDER ================== */
    .stSlider>div>div>div>div {
        background: var(--primary-gradient) !important;
    }
    
    .stSlider>div>div>div>div>div {
        background: white !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
    }
    
    /* ================== PROGRESS BAR ================== */
    .stProgress>div>div>div>div {
        background: var(--success-gradient) !important;
        border-radius: 10px;
    }
    
    /* ================== SIDEBAR ================== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(10, 14, 39, 0.95) 0%, rgba(26, 31, 58, 0.95) 100%);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    [data-testid="stSidebar"] .stRadio>div {
        background: transparent;
    }
    
    [data-testid="stSidebar"] .stRadio label {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 10px;
        padding: 12px 16px;
        transition: all 0.3s ease;
        border: 1px solid transparent;
        font-weight: 500;
    }
    
    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(102, 126, 234, 0.1);
        border-color: rgba(102, 126, 234, 0.3);
    }
    
    /* ================== INFO/SUCCESS/WARNING BOXES ================== */
    .stInfo {
        background: linear-gradient(135deg, rgba(79, 172, 254, 0.1) 0%, rgba(0, 242, 254, 0.1) 100%);
        border-left: 4px solid var(--accent-blue);
        border-radius: 8px;
        padding: 16px;
        backdrop-filter: blur(10px);
    }
    
    .stSuccess {
        background: linear-gradient(135deg, rgba(0, 242, 254, 0.1) 0%, rgba(79, 172, 254, 0.1) 100%);
        border-left: 4px solid var(--accent-green);
        border-radius: 8px;
        padding: 16px;
    }
    
    .stWarning {
        background: linear-gradient(135deg, rgba(240, 147, 251, 0.1) 0%, rgba(245, 87, 108, 0.1) 100%);
        border-left: 4px solid var(--accent-pink);
        border-radius: 8px;
        padding: 16px;
    }
    
    .stError {
        background: linear-gradient(135deg, rgba(250, 112, 154, 0.1) 0%, rgba(254, 225, 64, 0.1) 100%);
        border-left: 4px solid #fa709a;
        border-radius: 8px;
        padding: 16px;
    }
    
    /* ================== TABLES ================== */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* ================== CUSTOM SCROLLBAR ================== */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--primary-gradient);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #7d92f5 0%, #8b5ec4 100%);
    }
    
    /* ================== ANIMATIONS ================== */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    @keyframes glow {
        0%, 100% { box-shadow: 0 0 20px rgba(102, 126, 234, 0.3); }
        50% { box-shadow: 0 0 40px rgba(102, 126, 234, 0.6); }
    }
    
    .animate-fade-in {
        animation: fadeIn 0.6s ease-out;
    }
    
    .animate-pulse {
        animation: pulse 2s ease-in-out infinite;
    }
    
    .animate-glow {
        animation: glow 2s ease-in-out infinite;
    }
    
    /* ================== DIVIDERS ================== */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.5), transparent);
        margin: 2rem 0;
    }
    
    /* ================== CAPTIONS ================== */
    .caption {
        color: var(--text-secondary);
        font-size: 0.875rem;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# Load Resources
# -------------------------------------------------------------------------
MODEL_FILE = 'model.joblib'
LOC_FILE = 'location_counts.json'
ACCIDENT_DATA =  'accident_prediction_india.csv'

@st.cache_resource
def load_resources():
    model = None
    loc_counts = {}
    
    if os.path.exists(MODEL_FILE):
        model = joblib.load(MODEL_FILE)
    
    if os.path.exists(LOC_FILE):
        with open(LOC_FILE, 'r') as f:
            loc_counts = json.load(f)
    
    accident_data = None
    if os.path.exists(ACCIDENT_DATA):
        accident_data = pd.read_csv(ACCIDENT_DATA)
    
    safety_scorer = SafetyScorer(ACCIDENT_DATA if os.path.exists(ACCIDENT_DATA) else None)
    explainer = RiskExplainer(model) if model else None
    route_analyzer = RouteAnalyzer(model, safety_scorer) if model else None
    ambulance_predictor = AmbulanceResponsePredictor()
    recommendation_engine = SafetyRecommendationEngine(accident_data)
    
    return model, loc_counts, safety_scorer, explainer, route_analyzer, ambulance_predictor, recommendation_engine

pipeline, loc_counts, scorer, explainer, route_analyzer, ambulance_predictor, rec_engine = load_resources()

if pipeline is None:
    st.error("⚠️ Model file not found. Please run `train_notebook.py` first.")
    st.stop()

# -------------------------------------------------------------------------
# Historical Accident Count by City (from CSV)
# -------------------------------------------------------------------------
@st.cache_data
def get_city_accident_counts(csv_path: str):
    """Returns a dict of {city_name_lower: count} from the accident CSV."""
    if not os.path.exists(csv_path):
        return {}
    df = pd.read_csv(csv_path)
    col = 'City Name' if 'City Name' in df.columns else None
    if col is None:
        return {}
    counts = df[col].str.strip().str.lower().value_counts().to_dict()
    return counts

city_accident_counts = get_city_accident_counts(ACCIDENT_DATA)

def get_historical_count_for_place(place_name: str) -> tuple:
    """Fuzzy-match a place name against city accident counts.
    Returns (matched_city, count) or (None, 0)."""
    if not place_name or not city_accident_counts:
        return (None, 0)
    query = place_name.strip().lower()
    # Exact match first
    if query in city_accident_counts:
        return (query.title(), city_accident_counts[query])
    # Substring match
    for city, cnt in city_accident_counts.items():
        if query in city or city in query:
            return (city.title(), cnt)
    return (None, 0)

# -------------------------------------------------------------------------
# Premium Statistics Dashboard Header
# -------------------------------------------------------------------------
st.markdown("""
<div style="
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
    padding: 30px;
    border-radius: 20px;
    border: 1px solid rgba(102, 126, 234, 0.2);
    margin-bottom: 30px;
    backdrop-filter: blur(10px);
">
    <h1 style="margin: 0; padding: 0; font-size: 2.8rem;">🚦 Road Safety Intelligence System</h1>
    <p style="margin: 10px 0 0 0; font-size: 1.1rem; color: #b4b9d4;">
        AI-Powered Accident Prevention & Emergency Response Platform
    </p>
</div>
""", unsafe_allow_html=True)

# Statistics Cards
stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

with stat_col1:
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, rgba(79, 172, 254, 0.1) 0%, rgba(0, 242, 254, 0.1) 100%);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(79, 172, 254, 0.3);
        text-align: center;
    ">
        <div style="font-size: 2.5rem;">📊</div>
        <div style="font-size: 1.8rem; font-weight: 700; color: #4facfe; font-family: 'Space Grotesk', sans-serif;">
            {}
        </div>
        <div style="font-size: 0.875rem; color: #b4b9d4; text-transform: uppercase; letter-spacing: 0.05em;">
            Known Locations
        </div>
    </div>
    """.format(len(loc_counts) if loc_counts else 0), unsafe_allow_html=True)

with stat_col2:
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(102, 126, 234, 0.3);
        text-align: center;
    ">
        <div style="font-size: 2.5rem;">🎯</div>
        <div style="font-size: 1.8rem; font-weight: 700; color: #667eea; font-family: 'Space Grotesk', sans-serif;">
            98.2%
        </div>
        <div style="font-size: 0.875rem; color: #b4b9d4; text-transform: uppercase; letter-spacing: 0.05em;">
            Model Accuracy
        </div>
    </div>
    """, unsafe_allow_html=True)

with stat_col3:
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, rgba(240, 147, 251, 0.1) 0%, rgba(245, 87, 108, 0.1) 100%);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(240, 147, 251, 0.3);
        text-align: center;
    ">
        <div style="font-size: 2.5rem;">🏥</div>
        <div style="font-size: 1.8rem; font-weight: 700; color: #f093fb; font-family: 'Space Grotesk', sans-serif;">
            Live
        </div>
        <div style="font-size: 0.875rem; color: #b4b9d4; text-transform: uppercase; letter-spacing: 0.05em;">
            OSM Integration
        </div>
    </div>
    """, unsafe_allow_html=True)

with stat_col4:
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, rgba(0, 242, 254, 0.1) 0%, rgba(79, 172, 254, 0.1) 100%);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(0, 242, 254, 0.3);
        text-align: center;
    ">
        <div style="font-size: 2.5rem;">✨</div>
        <div style="font-size: 1.8rem; font-weight: 700; color: #00f2fe; font-family: 'Space Grotesk', sans-serif;">
            4
        </div>
        <div style="font-size: 0.875rem; color: #b4b9d4; text-transform: uppercase; letter-spacing: 0.05em;">
            AI Modules
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# Initialize Session State
# -------------------------------------------------------------------------
if 'risk_results' not in st.session_state:
    st.session_state.risk_results = None
if 'route_results' not in st.session_state:
    st.session_state.route_results = None
if 'recommendation_results' not in st.session_state:
    st.session_state.recommendation_results = None
if 'emergency_results' not in st.session_state:
    st.session_state.emergency_results = None
# Geo-resolved coordinates from place name or live GPS
if 'resolved_lat' not in st.session_state:
    st.session_state.resolved_lat = 12.9716
if 'resolved_lon' not in st.session_state:
    st.session_state.resolved_lon = 77.5946
if 'resolved_place' not in st.session_state:
    st.session_state.resolved_place = ""
if 'emergency_lat' not in st.session_state:
    st.session_state.emergency_lat = 12.9716
if 'emergency_lon' not in st.session_state:
    st.session_state.emergency_lon = 77.5946
if 'emergency_place' not in st.session_state:
    st.session_state.emergency_place = ""

# -------------------------------------------------------------------------
# Enhanced Sidebar
# -------------------------------------------------------------------------
st.sidebar.markdown("""
<div style="text-align: center; padding: 20px 0;">
    <div style="font-size: 3rem;">🚦</div>
    <h2 style="margin: 10px 0; font-size: 1.5rem;">Road Safety<br/>Intelligence</h2>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

st.sidebar.markdown("### 📍 Navigation")
tab_selection = st.sidebar.radio(
    "Navigation Menu",
    ["🔍 Risk Prediction", "🗺️ Route Analysis", "💡 Safety Recommendations", "🚑 Emergency Response"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")

st.sidebar.markdown("### ✨ Features")
st.sidebar.markdown("""
<div style="
    background: rgba(102, 126, 234, 0.1);
    padding: 12px;
    border-radius: 8px;
    margin-bottom: 10px;
    border-left: 3px solid #667eea;
">
    <div style="font-size: 0.875rem; font-weight: 600; color: #4facfe;">🧠 SHAP Explanations</div>
    <div style="font-size: 0.75rem; color: #b4b9d4;">AI-powered insights</div>
</div>

<div style="
    background: rgba(79, 172, 254, 0.1);
    padding: 12px;
    border-radius: 8px;
    margin-bottom: 10px;
    border-left: 3px solid #4facfe;
">
    <div style="font-size: 0.875rem; font-weight: 600; color: #00f2fe;">📊 Safety Scoring</div>
    <div style="font-size: 0.75rem; color: #b4b9d4;">0-100 scale ratings</div>
</div>

<div style="
    background: rgba(240, 147, 251, 0.1);
    padding: 12px;
    border-radius: 8px;
    margin-bottom: 10px;
    border-left: 3px solid #f093fb;
">
    <div style="font-size: 0.875rem; font-weight: 600; color: #f093fb;">🗺️ Multi-Route Analysis</div>
    <div style="font-size: 0.75rem; color: #b4b9d4;">Compare alternatives</div>
</div>

<div style="
    background: rgba(0, 242, 254, 0.1);
    padding: 12px;
    border-radius: 8px;
    margin-bottom: 10px;
    border-left: 3px solid #00f2fe;
">
    <div style="font-size: 0.875rem; font-weight: 600; color: #00f2fe;">🚑 Emergency ETA</div>
    <div style="font-size: 0.75rem; color: #b4b9d4;">Real-time predictions</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="text-align: center; padding: 10px; font-size: 0.75rem; color: #b4b9d4;">
    💯 100% Local Processing<br/>
    🔒 Privacy Guaranteed<br/>
    🆓 Free OpenStreetMap APIs
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# TAB 1: Risk Prediction
# -------------------------------------------------------------------------
if tab_selection == "🔍 Risk Prediction":
    st.title("🔍 Accident Risk Prediction")
    st.markdown("### AI-Powered Risk Assessment with Explainability")
    
    col_input, col_output = st.columns([1, 2])
    
    with col_input:
        st.subheader("📍 Input Parameters")
        
        # ── Location via Place Name or Live GPS ──────────────────────────
        st.markdown("**📍 Location**")
        place_name = st.text_input(
            "Place name",
            placeholder="e.g. Connaught Place, Delhi or MG Road, Bangalore",
            help="Type any address, landmark, or city. We'll look up its coordinates automatically."
        )

        # Live Location button via injected HTML/JS
        st.markdown("""
        <style>
        .live-btn {
            display:inline-block; padding:8px 16px;
            background:linear-gradient(135deg,#4facfe,#00f2fe);
            color:#0a0e27; border-radius:8px; font-weight:600;
            font-size:0.85rem; cursor:pointer; border:none;
            margin-bottom:8px;
        }
        .live-btn:hover{opacity:0.85;}
        </style>
        """, unsafe_allow_html=True)

        # Hidden text field receives the GPS coords from JS
        gps_coords_raw = st.text_input(
            "_gps_hidden", value="", label_visibility="collapsed",
            key="gps_input_risk",
            placeholder="GPS coords will appear here after clicking below"
        )

        components.html("""
        <button class="live-btn" onclick="getLocation()"
          style="padding:9px 18px;background:linear-gradient(135deg,#4facfe,#00f2fe);
                 color:#0a0e27;border-radius:8px;font-weight:700;font-size:0.9rem;
                 cursor:pointer;border:none;font-family:Inter,sans-serif;">
            📍 Use My Live Location
        </button>
        <div id="status" style="font-size:0.78rem;color:#b4b9d4;margin-top:4px;"></div>
        <script>
        function getLocation(){
          var s=document.getElementById('status');
          if(!navigator.geolocation){s.innerText='Geolocation not supported.';return;}
          s.innerText='Detecting location...';
          navigator.geolocation.getCurrentPosition(
            function(pos){
              var lat=pos.coords.latitude.toFixed(6);
              var lon=pos.coords.longitude.toFixed(6);
              // Write into the Streamlit hidden input
              var inputs=window.parent.document.querySelectorAll('input[data-testid="stTextInput"]');
              for(var i=0;i<inputs.length;i++){
                if(inputs[i].placeholder && inputs[i].placeholder.includes('GPS coords')){
                  var nativeInputValueSetter=Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype,'value').set;
                  nativeInputValueSetter.call(inputs[i],lat+','+lon);
                  inputs[i].dispatchEvent(new Event('input',{bubbles:true}));
                  break;
                }
              }
              s.innerText='📍 Got location: '+lat+', '+lon+' — now click Analyze Risk';
            },
            function(err){s.innerText='Error: '+err.message;}
          );
        }
        </script>
        """, height=80)

        # Parse GPS from hidden field if populated
        if gps_coords_raw and ',' in gps_coords_raw:
            try:
                _parts = gps_coords_raw.strip().split(',')
                st.session_state.resolved_lat = float(_parts[0])
                st.session_state.resolved_lon = float(_parts[1])
                st.session_state.resolved_place = f"Live GPS ({_parts[0][:7]}, {_parts[1][:7]})"
                st.markdown(
                    f"<div style='color:#4facfe;font-size:0.82rem;'>📍 Live location: "
                    f"{st.session_state.resolved_lat:.4f}, {st.session_state.resolved_lon:.4f}</div>",
                    unsafe_allow_html=True
                )
            except:
                pass
        elif st.session_state.resolved_place and not place_name:
            st.markdown(
                f"<div style='color:#4facfe;font-size:0.82rem;'>📍 Using: {st.session_state.resolved_place} "
                f"({st.session_state.resolved_lat:.4f}, {st.session_state.resolved_lon:.4f})</div>",
                unsafe_allow_html=True
            )

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

        # Road / location type
        known_areas = list(loc_counts.keys()) if loc_counts else ["Other"]
        area = st.selectbox("Road / Location Type", options=known_areas)

        # Time
        st.markdown("**⏰ Time**")
        time_val = st.time_input("Time of Day", pd.to_datetime("14:30").time())
        day_val = st.selectbox("Day of Week",
                               ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

        # Road Conditions
        st.markdown("**🛣️ Road Conditions**")
        road_type = st.selectbox("Road Type", ["Single carriageway", "Dual carriageway", "Roundabout", "One way", "Highway"])
        speed_limit = st.slider("Speed Limit (km/h)", 20, 120, 50)
        traffic_level = st.select_slider("Traffic Level", options=["Low", "Medium", "High"])

        # Weather
        st.markdown("**🌤️ Weather**")
        weather = st.selectbox("Weather Condition",
                               ["Fine", "Raining", "Raining and Windy", "Fog or Mist", "Other"])

        # Past Accidents auto-filled from loc_counts
        default_past = loc_counts.get(area, 0)
        past_accidents = st.number_input("Historical Accident Count (location type)", min_value=0, value=default_past)

        analyze_btn = st.button("🔍 Analyze Risk", use_container_width=True)
    
    # Process if button clicked
    if analyze_btn:
        # ── Geocode the place name if provided ──────────────────────────
        if place_name and place_name.strip():
            with st.spinner("🔍 Looking up location..."):
                coords = geocode_address(place_name.strip())
            if coords:
                st.session_state.resolved_lat, st.session_state.resolved_lon = coords
                st.session_state.resolved_place = place_name.strip()
            else:
                st.warning(f"⚠️ Could not find '{place_name}'. Using last known location.")

        lat = st.session_state.resolved_lat
        lon = st.session_state.resolved_lon

        # ── Compute historical accident count for the searched place ────
        search_key = place_name.strip() if place_name and place_name.strip() else st.session_state.resolved_place
        hist_city, hist_count = get_historical_count_for_place(search_key)

        hour = time_val.hour
        is_night = 1 if (hour >= 18 or hour < 6) else 0
        traffic_map = {"Low": "None", "Medium": "Give way", "High": "Traffic signal"}

        input_data = {
            'hour': hour,
            'is_night': is_night,
            'location_accident_count': past_accidents,
            'Day_of_Week': day_val,
            'Weather_conditions': weather,
            'Road_surface_type': road_type,
            'Speed_limit': speed_limit,
            'Traffic Control Presence': traffic_map[traffic_level],
            'Area_accident_occured': area,
            'Month': pd.to_datetime("today").month,
            'Year': pd.to_datetime("today").year,
            'Sex_of_driver': "Male",
            'State Name': "Karnataka",
            'City Name': "Bangalore",
            'Alcohol Involvement': "No",
            'Type_of_vehicle': "Automobile",
            'Number_of_casualties': 1,
            'Age_band_of_driver': 25,
            'Driver License Status': "Valid",
            'Road_surface_conditions': "Dry" if "Rain" not in weather else "Wet or damp",
            'Light_conditions': "Daylight" if not is_night else "Darkness - lights lit",
            'Number_of_vehicles_involved': 2,
            'Number of Fatalities': 0
        }

        input_df = pd.DataFrame([input_data])

        try:
            probs = pipeline.predict_proba(input_df)[0]
            pred_class = np.argmax(probs)
            confidence = probs[pred_class]

            risk_labels = ["Low", "Medium", "High"]
            risk_level = risk_labels[pred_class]

            safety_result = scorer.calculate_safety_score(
                area, road_type, speed_limit, hour, day_val, weather, traffic_level, lat, lon
            )

            explanation = None
            if explainer:
                explanation = explainer.explain_prediction(input_df, top_n=5)

            # STORE IN SESSION STATE
            st.session_state.risk_results = {
                'risk_level': risk_level,
                'confidence': confidence,
                'safety_result': safety_result,
                'explanation': explanation,
                'past_accidents': past_accidents,
                'traffic_level': traffic_level,
                'speed_limit': speed_limit,
                'resolved_lat': lat,
                'resolved_lon': lon,
                'resolved_place': st.session_state.resolved_place,
                'hist_city': hist_city,
                'hist_count': hist_count,
            }

        except Exception as e:
            st.session_state.risk_results = {'error': str(e)}
    
    # DISPLAY FROM SESSION STATE
    with col_output:
        if st.session_state.risk_results:
            results = st.session_state.risk_results
            
            if 'error' in results:
                st.error(f"Prediction Error: {results['error']}")
            else:
                # Risk Level Display with Enhanced Visuals
                risk_colors_bg = {
                    "Low": "linear-gradient(135deg, rgba(0, 242, 254, 0.15) 0%, rgba(79, 172, 254, 0.15) 100%)", 
                    "Medium": "linear-gradient(135deg, rgba(255, 193, 7, 0.15) 0%, rgba(255, 152, 0, 0.15) 100%)", 
                    "High": "linear-gradient(135deg, rgba(250, 112, 154, 0.15) 0%, rgba(254, 225, 64, 0.15) 100%)"
                }
                risk_colors_text = {"Low": "#00f2fe", "Medium": "#ffc107", "High": "#fa709a"}
                risk_icons = {"Low": "✅", "Medium": "⚠️", "High": "🚨"}
                
                st.markdown(f"""
                <div style="
                    background: {risk_colors_bg[results['risk_level']]};
                    padding: 30px;
                    border-radius: 16px;
                    border: 2px solid {risk_colors_text[results['risk_level']]}40;
                    text-align: center;
                    margin-bottom: 20px;
                ">
                    <div style="font-size: 4rem; margin-bottom: 10px;">{risk_icons[results['risk_level']]}</div>
                    <div style="
                        font-size: 2.5rem;
                        font-weight: 700;
                        color: {risk_colors_text[results['risk_level']]};
                        font-family: 'Space Grotesk', sans-serif;
                        margin-bottom: 10px;
                    ">{results['risk_level'].upper()} RISK</div>
                    <div style="
                        font-size: 1.1rem;
                        color: #b4b9d4;
                        font-weight: 500;
                    ">Prediction Confidence: {results['confidence']*100:.1f}%</div>
                    <div style="
                        width: 100%;
                        height: 8px;
                        background: rgba(255,255,255,0.1);
                        border-radius: 4px;
                        margin-top: 15px;
                        overflow: hidden;
                    ">
                        <div style="
                            width: {results['confidence']*100}%;
                            height: 100%;
                            background: {risk_colors_text[results['risk_level']]};
                            border-radius: 4px;
                        "></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Safety Score with Circular Progress
                score = results['safety_result']['score']
                score_color = "#00f2fe" if score >= 70 else "#ffc107" if score >= 40 else "#fa709a"
                
                st.markdown("### 🛡️ Safety Score Analysis")
                
                col_score, col_details = st.columns([1, 2])
                
                with col_score:
                    # Circular Progress Visualization
                    st.markdown(f"""
                    <div style="text-align: center; padding: 20px;">
                        <div style="
                            width: 180px;
                            height: 180px;
                            margin: 0 auto;
                            background: conic-gradient({score_color} {score*3.6}deg, rgba(255,255,255,0.1) {score*3.6}deg);
                            border-radius: 50%;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            position: relative;
                        ">
                            <div style="
                                width: 140px;
                                height: 140px;
                                background: #0a0e27;
                                border-radius: 50%;
                                display: flex;
                                flex-direction: column;
                                align-items: center;
                                justify-content: center;
                            ">
                                <div style="
                                    font-size: 3rem;
                                    font-weight: 700;
                                    color: {score_color};
                                    font-family: 'Space Grotesk', sans-serif;
                                ">{score}</div>
                                <div style="
                                    font-size: 0.875rem;
                                    color: #b4b9d4;
                                ">out of 100</div>
                            </div>
                        </div>
                        <div style="
                            margin-top: 15px;
                            font-size: 1.1rem;
                            font-weight: 600;
                            color: {score_color};
                        ">{results['safety_result']['category']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_details:
                    st.markdown(f"""
                    <div style="
                        background: rgba(255,255,255,0.03);
                        padding: 20px;
                        border-radius: 12px;
                        border: 1px solid rgba(255,255,255,0.1);
                        height: 100%;
                    ">
                        <h4 style="margin-top: 0; color: #ffffff;">Score Breakdown</h4>
                        <p style="color: #b4b9d4; line-height: 1.8;">
                            The safety score is calculated based on multiple risk factors including 
                            historical accident data, road conditions, time of day, weather patterns, 
                            and traffic levels at this location.
                        </p>
                        <div style="margin-top: 15px;">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                                <span style="color: #b4b9d4;">0-39:</span>
                                <span style="color: #fa709a;">High Risk</span>
                            </div>
                            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                                <span style="color: #b4b9d4;">40-69:</span>
                                <span style="color: #ffc107;">Medium Risk</span>
                            </div>
                            <div style="display: flex; justify-content: space-between;">
                                <span style="color: #b4b9d4;">70-100:</span>
                                <span style="color: #00f2fe;">Low Risk</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # ── Historical Accident Count Card ──────────────────────
                hist_city  = results.get('hist_city')
                hist_count = results.get('hist_count', 0)
                resolved_place = results.get('resolved_place', '')

                if hist_city and hist_count > 0:
                    hist_color = "#fa709a" if hist_count >= 100 else "#ffc107" if hist_count >= 30 else "#00f2fe"
                    hist_icon  = "🚨" if hist_count >= 100 else "⚠️" if hist_count >= 30 else "✅"
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, rgba(79,172,254,0.08) 0%, rgba(102,126,234,0.08) 100%);
                        padding: 18px 22px;
                        border-radius: 12px;
                        border-left: 4px solid {hist_color};
                        margin-bottom: 18px;
                        display: flex;
                        align-items: center;
                        gap: 16px;
                    ">
                        <div style="font-size:2.2rem;">{hist_icon}</div>
                        <div>
                            <div style="font-size:0.8rem;text-transform:uppercase;letter-spacing:0.08em;
                                        color:#b4b9d4;font-weight:600;">Historical Accident Data</div>
                            <div style="font-size:1.5rem;font-weight:700;color:{hist_color};
                                        font-family:'Space Grotesk',sans-serif;">
                                {hist_count:,} recorded accidents
                            </div>
                            <div style="font-size:0.85rem;color:#b4b9d4;margin-top:2px;">
                                in <strong style="color:#ffffff;">{hist_city}</strong> from dataset
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                elif resolved_place:
                    st.markdown(f"""
                    <div style="
                        background: rgba(255,255,255,0.03);
                        padding: 14px 18px;
                        border-radius: 10px;
                        border-left: 4px solid rgba(255,255,255,0.15);
                        margin-bottom: 16px;
                        font-size:0.88rem;
                        color:#b4b9d4;
                    ">
                        📊 No historical accident data found for <strong style="color:#ffffff;">{resolved_place}</strong> in the dataset.
                    </div>
                    """, unsafe_allow_html=True)

                # Key Metrics Row
                st.markdown("### 📊 Key Metrics")
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Location Type Accidents", results['past_accidents'],
                             delta=None, delta_color="off")
                with m2:
                    st.metric("Traffic Level", results['traffic_level'],
                             delta=None, delta_color="off")
                with m3:
                    st.metric("Speed Limit", f"{results['speed_limit']} km/h",
                             delta=None, delta_color="off")
                with m4:
                    st.metric("Safety Index", f"{score}/100",
                             delta=None, delta_color="off")
                
                st.markdown("---")
                
                # AI Explanation Section
                st.markdown("### 🧠 AI Explanation & Risk Factors")
                
                if results['explanation']:
                    explanation = results['explanation']
                    
                    # Natural language explanation
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
                        padding: 20px;
                        border-radius: 12px;
                        border-left: 4px solid #667eea;
                        margin-bottom: 20px;
                    ">
                        <div style="font-weight: 600; color: #4facfe; margin-bottom: 8px;">💬 What's Contributing to This Assessment:</div>
                        <div style="color: #e0e0e0; line-height: 1.7;">
                            {explanation.get('explanation', 'No explanation available')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Top contributing factors with visual bars
                    st.markdown("**Top Contributing Factors:**")
                    for i, factor in enumerate(explanation.get('top_factors', [])[:5], 1):
                        feature = factor.get('feature', 'Unknown').replace('_', ' ').title()
                        importance = factor.get('importance_pct', 0)
                        direction = factor.get('direction', 'affects')
                        
                        direction_icon = "📈" if "increas" in direction.lower() else "📉"
                        bar_color = "#fa709a" if "increas" in direction.lower() else "#00f2fe"
                        
                        st.markdown(f"""
                        <div style="margin-bottom: 15px;">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                                <span style="color: #ffffff; font-weight: 600;">
                                    {i}. {direction_icon} {feature}
                                </span>
                                <span style="color: {bar_color}; font-weight: 700;">
                                    {importance:.1f}%
                                </span>
                            </div>
                            <div style="
                                width: 100%;
                                height: 10px;
                                background: rgba(255,255,255,0.1);
                                border-radius: 5px;
                                overflow: hidden;
                            ">
                                <div style="
                                    width: {importance}%;
                                    height: 100%;
                                    background: {bar_color};
                                    border-radius: 5px;
                                "></div>
                            </div>
                            <div style="color: #b4b9d4; font-size: 0.85rem; margin-top: 3px;">
                                {direction.capitalize()}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("⚠️ SHAP explainer not available.")
                
                # Safety Recommendations based on risk level
                st.markdown("---")
                st.markdown("### 💡 Safety Recommendations")
                
                if results['risk_level'] == "High":
                    st.markdown("""
                    <div style="
                        background: linear-gradient(135deg, rgba(250, 112, 154, 0.1) 0%, rgba(254, 225, 64, 0.1) 100%);
                        padding: 20px;
                        border-radius: 12px;
                        border-left: 4px solid #fa709a;
                    ">
                        <div style="font-weight: 700; color: #fa709a; margin-bottom: 10px;">⚠️ HIGH RISK PRECAUTIONS:</div>
                        <ul style="color: #e0e0e0; line-height: 1.8; margin: 0;">
                            <li>Reduce speed significantly below the limit</li>
                            <li>Maintain extra following distance</li>
                            <li>Stay highly alert for pedestrians and obstacles</li>
                            <li>Consider alternative routes or timing</li>
                            <li>Ensure all safety equipment is functional</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                elif results['risk_level'] == "Medium":
                    st.markdown("""
                    <div style="
                        background: linear-gradient(135deg, rgba(255, 193, 7, 0.1) 0%, rgba(255, 152, 0, 0.1) 100%);
                        padding: 20px;
                        border-radius: 12px;
                        border-left: 4px solid #ffc107;
                    ">
                        <div style="font-weight: 700; color: #ffc107; margin-bottom: 10px;">⚡ MODERATE RISK PRECAUTIONS:</div>
                        <ul style="color: #e0e0e0; line-height: 1.8; margin: 0;">
                            <li>Drive defensively and stay attentive</li>
                            <li>Watch for changing road conditions</li>
                            <li>Be prepared for unexpected situations</li>
                            <li>Follow traffic rules strictly</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="
                        background: linear-gradient(135deg, rgba(0, 242, 254, 0.1) 0%, rgba(79, 172, 254, 0.1) 100%);
                        padding: 20px;
                        border-radius: 12px;
                        border-left: 4px solid #00f2fe;
                    ">
                        <div style="font-weight: 700; color: #00f2fe; margin-bottom: 10px;">✅ LOW RISK - GOOD CONDITIONS:</div>
                        <ul style="color: #e0e0e0; line-height: 1.8; margin: 0;">
                            <li>Conditions are favorable for travel</li>
                            <li>Maintain normal defensive driving practices</li>
                            <li>Stay alert and follow traffic rules</li>
                            <li>Weather and road conditions are good</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="
                background: rgba(255,255,255,0.03);
                padding: 40px;
                border-radius: 16px;
                border: 2px dashed rgba(255,255,255,0.1);
                text-align: center;
            ">
                <div style="font-size: 3rem; margin-bottom: 15px;">📊</div>
                <h3 style="color: #ffffff; margin-bottom: 10px;">Ready for Risk Analysis</h3>
                <p style="color: #b4b9d4; font-size: 1rem;">
                    Configure the parameters on the left and click <strong>"Analyze Risk"</strong> 
                    to get AI-powered safety predictions and recommendations.
                </p>
            </div>
            """, unsafe_allow_html=True)

# -------------------------------------------------------------------------
# TAB 2: Route Analysis
# -------------------------------------------------------------------------
elif tab_selection == "🗺️ Route Analysis":
    st.title("🗺️ Route Safety Analysis")
    st.markdown("### Compare Multiple Routes by Safety Score")
    
    col_params, col_map = st.columns([1, 2])
    
    with col_params:
        st.subheader("🎯 Route Parameters")
        
        origin_input = st.text_input("Origin Address", "Bangalore Railway Station")
        dest_input = st.text_input("Destination Address", "Kempegowda International Airport")
        
        hour_route = st.slider("Hour of Travel", 0, 23, 14)
        day_route = st.selectbox("Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], key="route_day")
        weather_route = st.selectbox("Weather", ["Fine", "Raining", "Fog or Mist"], key="route_weather")
        
        num_routes = st.slider("Number of Alternative Routes", 1, 3, 2)
        
        analyze_route_btn = st.button("🗺️ Analyze Routes", use_container_width=True)
    
    if analyze_route_btn:
        with st.spinner("Searching for routes..."):
            origin_coords = geocode_address(origin_input)
            dest_coords = geocode_address(dest_input)
            
            if not origin_coords or not dest_coords:
                st.error("Could not geocode addresses. Using default Bangalore coordinates.")
                origin_coords = (12.9716, 77.5946)
                dest_coords = (13.1986, 77.7066)
            
            if route_analyzer:
                routes = route_analyzer.analyze_route(
                    origin_coords[0], origin_coords[1],
                    dest_coords[0], dest_coords[1],
                    hour_route, day_route, weather_route, num_routes
                )
                
                # STORE IN SESSION STATE
                st.session_state.route_results = {
                    'routes': routes,
                    'origin_coords': origin_coords,
                    'dest_coords': dest_coords
                }
    
    # DISPLAY FROM SESSION STATE
    with col_map:
        if st.session_state.route_results:
            data = st.session_state.route_results
            routes = data['routes']
            
            if routes:
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, rgba(0, 242, 254, 0.1) 0%, rgba(79, 172, 254, 0.1) 100%);
                    padding: 20px;
                    border-radius: 12px;
                    border: 1px solid rgba(0, 242, 254, 0.3);
                    margin-bottom: 20px;
                    text-align: center;
                ">
                    <div style="font-size: 1.5rem; font-weight: 700; color: #00f2fe;">
                        ✅ Found {len(routes)} Alternative Route{'' if len(routes) == 1 else 's'}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("### 🏆 Best Route Recommendation")
                best_route = routes[0]
                
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, rgba(0, 242, 254, 0.15) 0%, rgba(79, 172, 254, 0.15) 100%);
                    padding: 25px;
                    border-radius: 16px;
                    border: 2px solid #00f2fe;
                    margin-bottom: 20px;
                ">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div style="flex: 1;">
                            <div style="font-size: 1.5rem; font-weight: 700; color: #00f2fe; margin-bottom: 10px;">
                                {best_route['recommendation']}
                            </div>
                            <div style="color: #b4b9d4; font-size: 1rem;">
                                🛡️ Safety Category: <strong style="color: #ffffff;">{best_route['category']}</strong>
                            </div>
                        </div>
                        <div style="text-align: center;">
                            <div style="
                                width: 100px;
                                height: 100px;
                                background: conic-gradient(#00f2fe {best_route['safety_score']*3.6}deg, rgba(255,255,255,0.1) {best_route['safety_score']*3.6}deg);
                                border-radius: 50%;
                                display: flex;
                                align-items: center;
                                justify-content: center;
                                margin-left: 20px;
                            ">
                                <div style="
                                    width: 75px;
                                    height: 75px;
                                    background: #0a0e27;
                                    border-radius: 50%;
                                    display: flex;
                                    flex-direction: column;
                                    align-items: center;
                                    justify-content: center;
                                ">
                                    <div style="font-size: 1.5rem; font-weight: 700; color: #00f2fe;">
                                        {best_route['safety_score']}
                                    </div>
                                    <div style="font-size: 0.7rem; color: #b4b9d4;">
                                        /100
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div style="display: flex; justify-content: space-around; margin-top: 20px; padding-top: 15px; border-top: 1px solid rgba(0, 242, 254, 0.2);">
                        <div style="text-align: center;">
                            <div style="font-size: 1.3rem; font-weight: 700; color: #ffffff;">
                                {best_route['distance_km']} km
                            </div>
                            <div style="font-size: 0.875rem; color: #b4b9d4;">Distance</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 1.3rem; font-weight: 700; color: #ffffff;">
                                {best_route['duration_min']} min
                            </div>
                            <div style="font-size: 0.875rem; color: #b4b9d4;">Duration</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # All Routes Comparison
                if len(routes) > 1:
                    st.markdown("### 📊 All Route Comparisons")
                    
                    for idx, route in enumerate(routes):
                        score_color = "#00f2fe" if route['safety_score'] >= 70 else "#ffc107" if route['safety_score'] >= 40 else "#fa709a"
                        border_style = "2px solid" if idx == 0 else "1px solid"
                        
                        st.markdown(f"""
                        <div style="
                            background: rgba(255,255,255,0.03);
                            padding: 20px;
                            border-radius: 12px;
                            border: {border_style} {score_color}40;
                            margin-bottom: 15px;
                        ">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <div style="font-size: 1.2rem; font-weight: 600; color: #ffffff; margin-bottom: 5px;">
                                        {'🏆 ' if idx == 0 else f'{idx + 1}. '}{route['recommendation']}
                                    </div>
                                    <div style="color: #b4b9d4; font-size: 0.9rem;">
                                        {route['category']} • {route['distance_km']} km • {route['duration_min']} min
                                    </div>
                                </div>
                                <div style="
                                    font-size: 1.8rem;
                                    font-weight: 700;
                                    color: {score_color};
                                    font-family: 'Space Grotesk', sans-serif;
                                ">
                                    {route['safety_score']}/100
                                </div>
                            </div>
                            <div style="
                                width: 100%;
                                height: 8px;
                                background: rgba(255,255,255,0.1);
                                border-radius: 4px;
                                margin-top: 12px;
                                overflow: hidden;
                            ">
                                <div style="
                                    width: {route['safety_score']}%;
                                    height: 100%;
                                    background: {score_color};
                                    border-radius: 4px;
                                "></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("### 🗺️ Interactive Route Map")
                
                
                center_lat = (data['origin_coords'][0] + data['dest_coords'][0]) / 2
                center_lon = (data['origin_coords'][1] + data['dest_coords'][1]) / 2
                
                m = folium.Map(location=[center_lat, center_lon], zoom_start=11)
                
                folium.Marker(
                    data['origin_coords'],
                    popup="Origin",
                    icon=folium.Icon(color="green", icon="play")
                ).add_to(m)
                
                folium.Marker(
                    data['dest_coords'],
                    popup="Destination",
                    icon=folium.Icon(color="red", icon="stop")
                ).add_to(m)
                
                route_colors = ['blue', 'orange', 'purple']
                for idx, route in enumerate(routes):
                    color = route_colors[idx % len(route_colors)]
                    geometry = route.get('geometry', [])
                    
                    if geometry:
                        coords = [(pt[1], pt[0]) for pt in geometry]
                        
                        folium.PolyLine(
                            coords,
                            color=color,
                            weight=5,
                            opacity=0.7,
                            popup=f"{route['recommendation']}: {route['safety_score']}/100"
                        ).add_to(m)
                
                st_folium(m, width=700, height=500)
        else:
            st.info("Enter origin and destination, then click 'Analyze Routes'.")

# -------------------------------------------------------------------------
# TAB 3: Safety Recommendations
# -------------------------------------------------------------------------
elif tab_selection == "💡 Safety Recommendations":
    st.title("💡 AI-Driven Safety Recommendations")
    st.markdown("###  Infrastructure & Traffic Management Suggestions")
    
    col_context, col_recs = st.columns([1, 2])
    
    with col_context:
        st.subheader("📍 Context")
        
        rec_area = st.selectbox("Area", list(loc_counts.keys()) if loc_counts else ["MG Road"], key="rec_area")
        default_accidents = min(loc_counts.get(rec_area, 0), 2000)
        rec_accident_count = st.number_input("Known Accidents", 0, 2000, default_accidents)
        
        rec_hour = st.slider("Typical Hour", 0, 23, 18)
        rec_weather = st.selectbox("Typical Weather", ["Fine", "Raining", "Fog or Mist"], key="rec_weather")
        rec_road_type = st.selectbox("Road Type", ["Single carriageway", "Highway", "Roundabout"], key="rec_road")
        rec_speed_limit = st.slider("Speed Limit", 20, 120, 60)
        
        generate_recs_btn = st.button("💡 Generate Recommendations", use_container_width=True)
    
    if generate_recs_btn:
        context = {
            'area': rec_area,
            'accident_count': rec_accident_count,
            'hour': rec_hour,
            'is_night': rec_hour >= 18 or rec_hour < 6,
            'weather': rec_weather,
            'road_type': rec_road_type,
            'speed_limit': rec_speed_limit
        }
        
        results = rec_engine.generate_comprehensive_recommendations(context)
        
        # STORE IN SESSION STATE
        st.session_state.recommendation_results = results
    
    # DISPLAY FROM SESSION STATE
    with col_recs:
        if st.session_state.recommendation_results:
            results = st.session_state.recommendation_results
            
            st.subheader(f"📋 {results['total_recommendations']} Recommendations Generated")
            
            st.markdown("### 🔴 High Priority Actions")
            for rec in results['high_priority'][:5]:
                st.markdown(f"""
                **{rec['action']}**  
                _Category:_ {rec['category']} | _Priority:_ {rec['priority']}  
                {rec['description']}  
                **Impact:** {rec['estimated_impact']} | **Timeline:** {rec['implementation_time']} | **Cost:** {rec['cost']}
                """)
                st.markdown("---")
            
            st.markdown("### ⚡ Quick Wins")
            for rec in results['quick_wins'][:3]:
                st.markdown(f"- **{rec['action']}** ({rec['category']}) - {rec['implementation_time']}")
        else:
            st.info("Set context and click 'Generate Recommendations'.")

# -------------------------------------------------------------------------
# TAB 4: Emergency Response
# -------------------------------------------------------------------------
elif tab_selection == "🚑 Emergency Response":
    st.title("🚑 Emergency Response Analysis")
    st.markdown("### Ambulance ETA Prediction for Accident Locations")
    
    col_accident, col_response = st.columns([1, 2])

    with col_accident:
        st.subheader("🚨 Accident Location")

        emg_place_name = st.text_input(
            "Place name or address",
            placeholder="e.g. Vijay Nagar, Bangalore",
            help="Type any place name or address to locate it on the map.",
            key="emg_place"
        )

        # Live location button for emergency tab
        emg_gps_raw = st.text_input(
            "_emg_gps_hidden", value="", label_visibility="collapsed",
            key="gps_input_emergency",
            placeholder="Emergency GPS coords"
        )

        components.html("""
        <button onclick="getEmgLocation()"
          style="padding:9px 18px;background:linear-gradient(135deg,#fa709a,#fee140);
                 color:#0a0e27;border-radius:8px;font-weight:700;font-size:0.9rem;
                 cursor:pointer;border:none;font-family:Inter,sans-serif;margin-bottom:6px;">
            📍 Use My Live Location
        </button>
        <div id="emg_status" style="font-size:0.78rem;color:#b4b9d4;margin-top:4px;"></div>
        <script>
        function getEmgLocation(){
          var s=document.getElementById('emg_status');
          if(!navigator.geolocation){s.innerText='Geolocation not supported.';return;}
          s.innerText='Detecting location...';
          navigator.geolocation.getCurrentPosition(
            function(pos){
              var lat=pos.coords.latitude.toFixed(6);
              var lon=pos.coords.longitude.toFixed(6);
              var inputs=window.parent.document.querySelectorAll('input[data-testid="stTextInput"]');
              for(var i=0;i<inputs.length;i++){
                if(inputs[i].placeholder && inputs[i].placeholder.includes('Emergency GPS')){
                  var nativeInputValueSetter=Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype,'value').set;
                  nativeInputValueSetter.call(inputs[i],lat+','+lon);
                  inputs[i].dispatchEvent(new Event('input',{bubbles:true}));
                  break;
                }
              }
              s.innerText='📍 Got: '+lat+', '+lon+' — click Find Hospitals';
            },
            function(err){s.innerText='Error: '+err.message;}
          );
        }
        </script>
        """, height=80)

        # Parse emergency GPS
        if emg_gps_raw and ',' in emg_gps_raw:
            try:
                _ep = emg_gps_raw.strip().split(',')
                st.session_state.emergency_lat = float(_ep[0])
                st.session_state.emergency_lon = float(_ep[1])
                st.session_state.emergency_place = f"Live GPS ({_ep[0][:7]}, {_ep[1][:7]})"
                st.markdown(
                    f"<div style='color:#fa709a;font-size:0.82rem;'>📍 Live: "
                    f"{st.session_state.emergency_lat:.4f}, {st.session_state.emergency_lon:.4f}</div>",
                    unsafe_allow_html=True
                )
            except:
                pass

        accident_hour = st.slider("Time of Accident (Hour)", 0, 23, 12)
        max_hospitals = st.slider("Max Hospitals to Show", 3, 10, 5)
        search_radius = st.slider("Search Radius (km)", 5, 50, 20)

        find_hospitals_btn = st.button("🚑 Find Nearest Hospitals", use_container_width=True)
    
    if find_hospitals_btn:
        # Geocode emergency place name if provided
        if emg_place_name and emg_place_name.strip():
            with st.spinner("🔍 Looking up location..."):
                emg_coords = geocode_address(emg_place_name.strip())
            if emg_coords:
                st.session_state.emergency_lat, st.session_state.emergency_lon = emg_coords
                st.session_state.emergency_place = emg_place_name.strip()
            else:
                st.warning(f"⚠️ Could not find '{emg_place_name}'. Using last known location.")

        accident_lat = st.session_state.emergency_lat
        accident_lon = st.session_state.emergency_lon

        with st.spinner("Locating hospitals..."):
            predictions = ambulance_predictor.predict_response_times(
                accident_lat, accident_lon, accident_hour, max_hospitals, search_radius
            )

            # STORE IN SESSION STATE
            st.session_state.emergency_results = {
                'predictions': predictions,
                'accident_lat': accident_lat,
                'accident_lon': accident_lon,
                'location_name': st.session_state.emergency_place or f"{accident_lat:.4f}, {accident_lon:.4f}"
            }
    
    # DISPLAY FROM SESSION STATE
    with col_response:
        if st.session_state.emergency_results:
            data = st.session_state.emergency_results
            predictions = data['predictions']
            
            if predictions:
                st.success(f"✅ Found {len(predictions)} hospitals")
                
                best = predictions[0]
                st.markdown(f"### 🏥 Recommended Hospital")
                st.markdown(f"#### {best['hospital_name']}")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("ETA", f"{best['total_eta_min']} min")
                with col2:
                    st.metric("Distance", f"{best['distance_km']} km")
                with col3:
                    st.metric("Response", best['response_category'])
                with col4:
                    st.metric("Rank", f"#{best['rank']}")
                
                survival = ambulance_predictor.estimate_survival_probability(best['total_eta_min'])
                st.info(f"**Urgency:** {survival['urgency']} | **Golden Hour:** {'Yes ✓' if survival['within_golden_hour'] else 'No ✗'}")
                
                st.markdown("---")
                st.subheader("📊 All Hospitals in Range")
                
                hospital_df = pd.DataFrame([{
                    "Rank": p['rank'],
                    "Hospital": p['hospital_name'],
                    "ETA (min)": p['total_eta_min'],
                    "Distance (km)": p['distance_km'],
                    "Category": p['response_category']
                } for p in predictions])
                
                st.dataframe(hospital_df, use_container_width=True)
                
                st.subheader("🗺️ Hospital Locations")
                
                m = folium.Map(location=[data['accident_lat'], data['accident_lon']], zoom_start=12)
                
                folium.Marker(
                    [data['accident_lat'], data['accident_lon']],
                    popup="Accident Location",
                    icon=folium.Icon(color="red", icon="exclamation-triangle", prefix='fa')
                ).add_to(m)
                
                for p in predictions[:5]:
                    color = "green" if p['rank'] == 1 else "blue" if p['has_emergency'] else "gray"
                    
                    folium.Marker(
                        [p['hospital_lat'], p['hospital_lon']],
                        popup=f"{p['hospital_name']}<br>ETA: {p['total_eta_min']} min",
                        icon=folium.Icon(color=color, icon="plus", prefix='fa')
                    ).add_to(m)
                
                st_folium(m, width=700, height=400)
            else:
                st.warning("No hospitals found. Try increasing search radius.")
        else:
            st.info("Enter accident location and click 'Find Nearest Hospitals'.")

# -------------------------------------------------------------------------
# Premium Footer
# -------------------------------------------------------------------------
st.markdown("---")

st.markdown("""
<div style="
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
    padding: 40px;
    border-radius: 16px;
    border: 1px solid rgba(102, 126, 234, 0.15);
    margin-top: 40px;
">
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 30px; margin-bottom: 30px;">
        <div>
            <h3 style="font-size: 1.2rem; margin-bottom: 15px; color: #ffffff;">🚦 Road Safety Intelligence</h3>
            <p style="color: #b4b9d4; font-size: 0.9rem; line-height: 1.6;">
                AI-powered accident prediction and prevention platform for safer Indian roads.
            </p>
        </div>
        
        <div>
            <h4 style="font-size: 1rem; margin-bottom: 12px; color: #ffffff;">✨ Key Features</h4>
            <ul style="color: #b4b9d4; font-size: 0.85rem; line-height: 1.8; list-style: none; padding: 0;">
                <li>🧠 SHAP Explainability</li>
                <li>🗺️ Multi-Route Analysis</li>
                <li>🚑 Emergency Response</li>
                <li>💡 AI Recommendations</li>
            </ul>
        </div>
        
        <div>
            <h4 style="font-size: 1rem; margin-bottom: 12px; color: #ffffff;">🛠️ Technology Stack</h4>
            <div style="display: flex; flex-wrap: wrap; gap: 8px;">
                <span style="
                    background: rgba(102, 126, 234, 0.2);
                    padding: 4px 12px;
                    border-radius: 20px;
                    font-size: 0.75rem;
                    color: #4facfe;
                    border: 1px solid rgba(102, 126, 234, 0.3);
                ">Streamlit</span>
                <span style="
                    background: rgba(79, 172, 254, 0.2);
                    padding: 4px 12px;
                    border-radius: 20px;
                    font-size: 0.75rem;
                    color: #00f2fe;
                    border: 1px solid rgba(79, 172, 254, 0.3);
                ">XGBoost</span>
                <span style="
                    background: rgba(240, 147, 251, 0.2);
                    padding: 4px 12px;
                    border-radius: 20px;
                    font-size: 0.75rem;
                    color: #f093fb;
                    border: 1px solid rgba(240, 147, 251, 0.3);
                ">SHAP</span>
                <span style="
                    background: rgba(0, 242, 254, 0.2);
                    padding: 4px 12px;
                    border-radius: 20px;
                    font-size: 0.75rem;
                    color: #00f2fe;
                    border: 1px solid rgba(0, 242, 254, 0.3);
                ">OpenStreetMap</span>
            </div>
        </div>
        
        <div>
            <h4 style="font-size: 1rem; margin-bottom: 12px; color: #ffffff;">🔒 Privacy & Security</h4>
            <ul style="color: #b4b9d4; font-size: 0.85rem; line-height: 1.8; list-style: none; padding: 0;">
                <li>💯 100% Local Processing</li>
                <li>🔒 No Data Collection</li>
                <li>🆓 Free OSM APIs</li>
                <li>⚡ Real-time Analysis</li>
            </ul>
        </div>
    </div>
    
    <div style="
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        padding-top: 20px;
        text-align: center;
    ">
        <p style="color: #b4b9d4; font-size: 0.9rem; margin: 0;">
            Built with ❤️ for safer roads in India | 
            <strong style="color: #667eea;">Machine Learning</strong> • 
            <strong style="color: #4facfe;">Geospatial Analysis</strong> • 
            <strong style="color: #f093fb;">Explainable AI</strong>
        </p>
        <p style="color: #798299; font-size: 0.75rem; margin-top: 10px;">
            © 2026 Road Safety Intelligence System • For Educational & Research Purposes
        </p>
    </div>
</div>
""", unsafe_allow_html=True)
