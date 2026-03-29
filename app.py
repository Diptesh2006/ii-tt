import streamlit as st
import os
import torch
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import leafmap.foliumap as leafmap
from datetime import datetime

# Custom modules
from src.inference_engine import load_model, run_tiled_inference, CLASS_NAMES, NUM_CLASSES
from src.analytics import calculate_areas, get_municipal_insights, format_class_data_for_plotly
from src.utils import generate_thumbnail, get_cache_paths, load_analytics_cache, save_analytics_cache

# ─────────────────────────────────────────────────
# CONFIG & STYLING
# ─────────────────────────────────────────────────

st.set_page_config(
    page_title="Geospatial Intelligence Dashboard",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Technical CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;600;700&display=swap');

    .main {
        background: #0B0E14;
    }
    .stApp {
        background-color: #0B0E14;
        font-family: 'Inter', sans-serif;
    }
    [data-testid="stSidebar"] {
        background-color: #0F1219;
        border-right: 1px solid #1E232E;
    }
    .metric-card {
        background: #161B22;
        padding: 24px;
        border-radius: 8px;
        border: 1px solid #30363D;
        margin-bottom: 16px;
        transition: border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: #58A6FF;
    }
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.8rem;
        font-weight: 700;
        color: #F0F6FC;
        letter-spacing: -0.5px;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #8B949E;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 4px;
    }
    .insight-card {
        background: rgba(88, 166, 255, 0.05);
        border-left: 4px solid #58A6FF;
        padding: 20px;
        border-radius: 4px;
        margin-top: 12px;
    }
    .recommendation-item {
        margin-bottom: 10px;
        color: #C9D1D9;
        font-size: 0.95rem;
        display: flex;
        align-items: flex-start;
    }
    .recommendation-item:before {
        content: "•";
        color: #58A6FF;
        font-weight: bold;
        display: inline-block;
        width: 1em;
        margin-left: -1em;
        margin-right: 0.5em;
    }
    h1, h2, h3 {
        color: #F0F6FC;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0 0;
        gap: 0;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: transparent;
        border-bottom: 2px solid #58A6FF !important;
    }
    hr {
        border: 0;
        border-top: 1px solid #30363D;
        margin: 2rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────
# CONSTANTS & SETUP
# ─────────────────────────────────────────────────

MODEL_PATH = r"d:\geo-spatial\best"
DATA_DIR = r"D:\geo-dataste\CG_live-demo\live-demo"
CACHE_DIR = r"d:\geo-spatial\results"

if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# ─────────────────────────────────────────────────
# SIDEBAR / CONTROLS
# ─────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### System Configuration")
    st.markdown("---")
    
    # Village Selection
    all_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.tif')]
    selected_village_file = st.selectbox("Select Target Orthophoto", all_files)
    village_path = os.path.join(DATA_DIR, selected_village_file)
    village_name = os.path.splitext(selected_village_file)[0]
    
    st.markdown("---")
    st.subheader("Compute Parameters")
    alpha_val = st.slider("Segmentation Opacity", 0.0, 1.0, 0.5, step=0.1)
    use_gpu = st.toggle("Hardware Acceleration", value=torch.cuda.is_available())
    
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    
    st.code(f"EXECUTION_DEVICE: {device.type.upper()}", language="bash")
    
    if st.button("Initialize Analysis", type="primary", use_container_width=True):
        st.session_state.force_analyze = True
    else:
        st.session_state.force_analyze = False

# ─────────────────────────────────────────────────
# DATA LOADING / CACHING
# ─────────────────────────────────────────────────

cache_paths = get_cache_paths(village_path, CACHE_DIR)
stats = load_analytics_cache(cache_paths["stats"])

# Check if analysis is needed
should_run = st.session_state.get("force_analyze", False) or stats is None or not os.path.exists(cache_paths["overlay"])

# ─────────────────────────────────────────────────
# MAIN DASHBOARD
# ─────────────────────────────────────────────────

st.title(village_name)
st.caption(f"Source: {village_path}")

col_main, col_stats = st.columns([2, 1])

with col_main:
    if should_run:
        thumb_path = generate_thumbnail(village_path, cache_paths["thumbnail"])
        st.image(thumb_path, caption="Raw Preview (Generating high-resolution analysis...)", use_container_width=True)
        
        progress_bar = st.progress(0, text="Executing Tiled Inference...")
        
        def update_progress(current, total):
            progress = current / total
            progress_bar.progress(progress, text=f"Processing Tile {current} of {total} ({int(progress*100)}%)")
            
        model, processor = load_model(MODEL_PATH, device)
        
        class_counts, transform, crs, total_time, avg_tile_time = run_tiled_inference(
            village_path, model, processor, device, 
            cache_paths["overlay"], mode='overlay', alpha=alpha_val, 
            progress_callback=update_progress
        )
        
        areas = calculate_areas(class_counts, transform, crs)
        insights = get_municipal_insights(areas)
        
        save_analytics_cache(cache_paths["stats"], {
            "areas": areas, 
            "insights": insights,
            "metrics": {
                "total_inference_time": total_time,
                "avg_tile_time": avg_tile_time
            }
        })
        
        progress_bar.empty()
        st.success("Analysis sequence completed successfully.")
        st.rerun()
        
    else:
        st.subheader("Visualization Engine")
        
        tab1, tab2 = st.tabs(["Comparative Analysis", "High-Resolution Overlay"])
        
        with tab1:
            thumb_path = generate_thumbnail(village_path, cache_paths["thumbnail"])
            overlay_thumb = generate_thumbnail(cache_paths["overlay"], cache_paths["overlay"].replace(".tif", "_thumb.png"))
            
            c1, c2 = st.columns(2)
            c1.image(thumb_path, caption="Original RGB Data", use_container_width=True)
            c2.image(overlay_thumb, caption="Semantic Segmentation Output", use_container_width=True)
            
        with tab2:
            st.image(overlay_thumb, caption=f"Composite Layer (Opacity: {alpha_val})", use_container_width=True)

with col_stats:
    if stats:
        areas = stats["areas"]
        insights = stats["insights"]
        metrics = stats.get("metrics", {})
        
        st.subheader("Statistical Summary")
        
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Urban Density</div>
                <div class="metric-value">{insights['urban_density_score']:.1f}%</div>
            </div>""", unsafe_allow_html=True)
            
        with m2:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Built-up Surface</div>
                <div class="metric-value">{areas['2']['hectares']:.2f}<span style="font-size:0.8rem; color:#8B949E; margin-left:4px;">HA</span></div>
            </div>""", unsafe_allow_html=True)
            
        m3, m4 = st.columns(2)
        with m3:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Infrastructure</div>
                <div class="metric-value">{insights['road_connectivity_score']:.1f}%</div>
            </div>""", unsafe_allow_html=True)
            
        with m4:
             st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Hydrographic Ratio</div>
                <div class="metric-value">{insights['water_security_ratio']:.1f}%</div>
            </div>""", unsafe_allow_html=True)

        if metrics:
            st.markdown("---")
            st.subheader("Performance Metrics")
            m_time1, m_time2 = st.columns(2)
            with m_time1:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-label">Total Latency</div>
                    <div class="metric-value">{metrics['total_inference_time']:.2f}<span style="font-size:0.8rem; color:#8B949E; margin-left:4px;">s</span></div>
                </div>""", unsafe_allow_html=True)
            with m_time2:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-label">Avg Tile Latency</div>
                    <div class="metric-value">{metrics['avg_tile_time']:.2f}<span style="font-size:0.8rem; color:#8B949E; margin-left:4px;">s</span></div>
                </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("Municipal Insights")
        st.markdown(f"""<div class="insight-card">
            <strong>Strategic Assessment:</strong> {insights['assessment']}
        </div>""", unsafe_allow_html=True)
        
        st.write("")
        for rec in insights["recommendations"]:
            st.markdown(f'<div class="recommendation-item">{rec}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────
# ANALYTICS BREAKDOWN
# ─────────────────────────────────────────────────

if stats:
    st.markdown("---")
    st.subheader("Land-Use Classification Breakdown")
    
    col_pie, col_bar = st.columns(2)
    
    names, hectares = format_class_data_for_plotly(stats["areas"], CLASS_NAMES)
    
    with col_pie:
        colors = ['#3C3C3C', '#F0B419', '#D73232', '#1973D7']
        fig_pie = px.pie(
            values=hectares, names=names, 
            color=names, color_discrete_map={names[i]: colors[i] for i in range(len(names))},
            hole=0.4, title="Class Distribution Matrix"
        )
        fig_pie.update_layout(
            template="plotly_dark", 
            plot_bgcolor='rgba(0,0,0,0)', 
            paper_bgcolor='rgba(0,0,0,0)',
            font_family="Inter",
            margin=dict(t=40, b=0, l=0, r=0)
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col_bar:
        fig_bar = px.bar(
            x=names, y=hectares, color=names,
            color_discrete_map={names[i]: colors[i] for i in range(len(names))},
            title="Area Quantification (Hectares)",
            labels={'x': 'Classification', 'y': 'Hectares'}
        )
        fig_bar.update_layout(
            template="plotly_dark", 
            plot_bgcolor='rgba(0,0,0,0)', 
            paper_bgcolor='rgba(0,0,0,0)',
            font_family="Inter",
            margin=dict(t=40, b=0, l=0, r=0)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# ─────────────────────────────────────────────────
# DATA EXPORT
# ─────────────────────────────────────────────────

st.markdown("---")
st.subheader("Export Data Matrix")
df_stats = pd.DataFrame({
    "Classification": names,
    "Area (Hectares)": hectares,
    "Area (Acres)": [stats["areas"][str(i)]["acres"] for i in range(len(names))]
})
st.dataframe(df_stats, use_container_width=True)

csv = df_stats.to_csv(index=False).encode('utf-8')
st.download_button(
    "Export Analysis Report (CSV)",
    csv,
    f"{village_name}_analysis.csv",
    "text/csv",
    key='download-csv',
    use_container_width=True
)

st.markdown(f"""
    <div style="text-align: center; color: #484F58; margin-top: 60px; font-size: 0.75rem; font-family: 'JetBrains Mono';">
        GEOSPATIAL INTELLIGENCE CORE | AEGFORMER ENGINE | {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
    </div>
""", unsafe_allow_html=True)

