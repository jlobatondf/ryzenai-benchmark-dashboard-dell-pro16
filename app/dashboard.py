"""
Streamlit Dashboard for Dell Pro 16 RyzenAI Benchmark Analysis

Provides interactive visualizations across 9 tabs:
1. Executive Summary
2. Provider Comparison
3. Model Performance
4. Reliability Analysis
5. Run Comparison
6. System Comparison
7. Power & Thermal Analysis
8. Latency Distribution
9. Use Case Mapping
"""

import warnings
import logging

# Suppress warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
logging.getLogger('streamlit.runtime.scriptrunner.script_runner').setLevel(logging.ERROR)

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
import httpx
import uuid
import os
import hashlib
from datetime import datetime, timedelta
import extra_streamlit_components as stx

# Page configuration
st.set_page_config(
    page_title="Dell Pro 16 RyzenAI Benchmark",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# AUTHENTICATION
# ============================================================================

def check_password():
    """Password protection with 'Remember me' cookie support."""
    # Skip auth if no password is set (local development)
    app_password = os.getenv("APP_PASSWORD", "")
    if not app_password:
        return True

    cookie_manager = stx.CookieManager(key="auth_cookie_manager")
    secret_key = os.getenv("COOKIE_SECRET", "default-dev-secret-change-in-prod")

    # Create a signed token (password + secret = harder to forge)
    valid_token = hashlib.sha256(f"{app_password}{secret_key}".encode()).hexdigest()

    # Check for existing valid cookie
    auth_cookie = cookie_manager.get("auth_token")
    if auth_cookie == valid_token:
        st.session_state.authenticated = True
        return True

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    st.title("🔐 Dell Pro 16 RyzenAI Dashboard")
    st.markdown("Please enter the password to access the dashboard.")

    password = st.text_input("Password", type="password", key="password_input")
    remember_me = st.checkbox("Remember me for 30 days", key="remember_me")

    if st.button("Login", type="primary"):
        if password == app_password:
            st.session_state.authenticated = True
            if remember_me:
                cookie_manager.set(
                    "auth_token",
                    valid_token,
                    expires_at=datetime.now() + timedelta(days=30)
                )
            st.rerun()
        else:
            st.error("Incorrect password")
    return False

# ============================================================================
# DARK MODE SUPPORT
# ============================================================================

def get_dark_mode_css():
    """Return CSS for dark mode that directly targets Streamlit elements."""
    return """
    <style>
        /* Main app and all nested containers */
        .stApp,
        .stApp > div,
        .stApp > div > div,
        .stApp > div > div > div,
        .main,
        .main .block-container,
        section.main,
        section.main > div,
        [data-testid="stAppViewContainer"],
        [data-testid="stAppViewContainer"] > div,
        [data-testid="stAppViewContainer"] > section,
        [data-testid="stAppViewContainer"] > section > div,
        [data-testid="stMain"],
        [data-testid="stMainBlockContainer"],
        [data-testid="stVerticalBlock"],
        .block-container {
            background-color: #0e1117 !important;
        }

        /* Header */
        .stApp > header,
        [data-testid="stHeader"],
        header {
            background-color: #0e1117 !important;
        }

        /* Tab panels need dark background too */
        [role="tabpanel"],
        [data-baseweb="tab-panel"],
        .stTabs > div:last-child {
            background-color: #0e1117 !important;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #262730 !important;
        }

        [data-testid="stSidebar"] > div:first-child {
            background-color: #262730 !important;
        }

        [data-testid="stSidebarContent"] {
            background-color: #262730 !important;
        }

        /* Text colors */
        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
            color: #fafafa !important;
        }

        .stApp p, .stApp span, .stApp label, .stApp div {
            color: #fafafa !important;
        }

        .stApp .stMarkdown {
            color: #fafafa !important;
        }

        /* Metrics */
        [data-testid="stMetricValue"] {
            color: #fafafa !important;
        }

        [data-testid="stMetricLabel"] {
            color: #b0b0b0 !important;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            background-color: #262730 !important;
            gap: 2px;
        }

        .stTabs [data-baseweb="tab"] {
            background-color: #262730 !important;
            color: #fafafa !important;
        }

        .stTabs [aria-selected="true"] {
            background-color: #4a4a5a !important;
        }

        /* Buttons */
        .stButton > button,
        [data-testid="stSidebar"] button,
        [data-testid="stSidebar"] .stButton button,
        [data-testid="baseButton-secondary"],
        [data-testid="baseButton-primary"],
        button[kind="secondary"],
        button[kind="primary"] {
            background-color: #262730 !important;
            color: #fafafa !important;
            border-color: #4a4a5a !important;
        }

        .stButton > button:hover,
        [data-testid="stSidebar"] button:hover {
            background-color: #3a3a4a !important;
            border-color: #6a6a7a !important;
        }

        /* Sidebar specific button containers */
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] button {
            background-color: #262730 !important;
            color: #fafafa !important;
            border-color: #4a4a5a !important;
        }

        /* All button elements in sidebar - catch all */
        [data-testid="stSidebar"] button,
        [data-testid="stSidebarContent"] button,
        section[data-testid="stSidebar"] button {
            background-color: #262730 !important;
            color: #fafafa !important;
            border-color: #4a4a5a !important;
        }

        /* Primary and secondary button variants */
        [data-testid="stSidebar"] [data-testid="stButton"] button,
        [data-testid="stSidebar"] .stButton button,
        [data-testid="stSidebar"] div[data-testid="column"] button {
            background-color: #262730 !important;
            color: #fafafa !important;
            border-color: #4a4a5a !important;
        }

        /* Button inner elements */
        [data-testid="stSidebar"] button p,
        [data-testid="stSidebar"] button span {
            color: #fafafa !important;
        }

        /* Force ALL buttons in sidebar to dark - ultimate catch-all */
        [data-testid="stSidebar"] button,
        [data-testid="stSidebar"] [role="button"],
        [data-testid="stSidebar"] .stButton > button,
        [data-testid="stSidebarContent"] button,
        [data-testid="stSidebarContent"] [role="button"],
        section[data-testid="stSidebar"] button,
        section[data-testid="stSidebar"] [role="button"] {
            background: #262730 !important;
            background-color: #262730 !important;
            color: #fafafa !important;
            border-color: #4a4a5a !important;
        }

        /* Target by element type directly in sidebar */
        [data-testid="stSidebar"] button[kind],
        [data-testid="stSidebar"] button[data-testid] {
            background: #262730 !important;
            background-color: #262730 !important;
        }

        /* Select boxes and inputs */
        .stSelectbox > div > div,
        .stMultiSelect > div > div,
        .stTextInput > div > div > input {
            background-color: #262730 !important;
            color: #fafafa !important;
        }

        /* Text input containers - more specific targeting */
        .stTextInput input,
        .stTextInput > div > div,
        [data-testid="stTextInput"] input,
        [data-testid="stTextInput"] > div,
        [data-testid="stTextInput"] > div > div,
        [data-baseweb="input"],
        [data-baseweb="base-input"] {
            background-color: #262730 !important;
            color: #fafafa !important;
            border-color: #4a4a5a !important;
        }

        /* Input placeholders */
        .stTextInput input::placeholder,
        [data-testid="stTextInput"] input::placeholder {
            color: #808080 !important;
        }

        /* Multiselect dropdown */
        [data-baseweb="select"] > div,
        [data-baseweb="popover"] > div {
            background-color: #262730 !important;
        }

        /* Dropdown menu items */
        [data-baseweb="menu"],
        [data-baseweb="menu"] ul,
        [data-baseweb="menu"] li {
            background-color: #262730 !important;
            color: #fafafa !important;
        }

        [data-baseweb="menu"] li:hover {
            background-color: #3a3a4a !important;
        }

        /* DataFrames - comprehensive styling */
        .stDataFrame,
        [data-testid="stDataFrame"],
        [data-testid="stDataFrame"] > div,
        [data-testid="stDataFrame"] iframe,
        .stDataFrame > div {
            background-color: #0e1117 !important;
        }

        /* DataFrame table styling */
        [data-testid="stDataFrame"] table,
        [data-testid="stDataFrame"] th,
        [data-testid="stDataFrame"] td,
        [data-testid="stDataFrame"] thead,
        [data-testid="stDataFrame"] tbody,
        [data-testid="stDataFrame"] tr {
            background-color: #0e1117 !important;
            color: #fafafa !important;
            border-color: #3a3a4a !important;
        }

        /* HTML tables in markdown or custom HTML */
        table,
        table th,
        table td,
        table thead,
        table tbody,
        table tr {
            background-color: #0e1117 !important;
            color: #fafafa !important;
            border-color: #3a3a4a !important;
        }

        table th {
            background-color: #1a1a2e !important;
        }

        table tr:nth-child(even) {
            background-color: #141420 !important;
        }

        table tr:hover {
            background-color: #262730 !important;
        }

        /* Download buttons */
        .stDownloadButton button,
        [data-testid="stDownloadButton"] button,
        [data-testid="baseButton-secondary"] {
            background-color: #262730 !important;
            color: #fafafa !important;
            border-color: #4a4a5a !important;
        }

        .stDownloadButton button:hover,
        [data-testid="stDownloadButton"] button:hover {
            background-color: #3a3a4a !important;
        }

        /* Containers and cards */
        [data-testid="stHorizontalBlock"],
        [data-testid="column"],
        .element-container {
            background-color: transparent !important;
        }

        /* Metric containers */
        [data-testid="stMetric"],
        [data-testid="metric-container"] {
            background-color: transparent !important;
        }

        /* Alert/Info boxes */
        .stAlert,
        [data-testid="stAlert"],
        [role="alert"] {
            background-color: #1a1a2e !important;
            border-color: #4a4a5a !important;
        }

        /* Expanders */
        .streamlit-expanderHeader,
        [data-testid="stExpander"] summary,
        details summary {
            background-color: #262730 !important;
            color: #fafafa !important;
        }

        [data-testid="stExpander"],
        details {
            background-color: #1a1a2e !important;
            border-color: #4a4a5a !important;
        }

        .streamlit-expanderContent,
        [data-testid="stExpander"] > div:last-child,
        details > div {
            background-color: #1a1a2e !important;
        }

        /* Code blocks */
        code, pre {
            background-color: #1a1a2e !important;
            color: #fafafa !important;
        }

        /* Run status cards - target card-like containers with inline styles */
        [data-testid="stVerticalBlock"] > div > div[style*="background"] {
            background-color: #262730 !important;
        }

        /* Override inline background-color for card divs */
        div[style*="background-color: #f0f2f6"],
        div[style*="background-color:#f0f2f6"],
        div[style*="background: #f0f2f6"] {
            background-color: #262730 !important;
            background: #262730 !important;
        }

        /* Card text elements */
        div[style*="background-color: #f0f2f6"] h4,
        div[style*="background-color: #f0f2f6"] p,
        div[style*="background-color: #f0f2f6"] b,
        div[style*="background-color:#f0f2f6"] h4,
        div[style*="background-color:#f0f2f6"] p,
        div[style*="background-color:#f0f2f6"] b {
            color: #fafafa !important;
        }

        /* Generic div backgrounds in main content */
        .main [data-testid="stVerticalBlock"] > div {
            background-color: transparent !important;
        }

        .streamlit-expanderContent {
            background-color: #1a1a2e !important;
        }

        /* Dividers */
        hr {
            border-color: #4a4a5a !important;
        }

        /* Cards/containers */
        [data-testid="stVerticalBlock"] > div {
            background-color: transparent !important;
        }

        /* Download buttons */
        .stDownloadButton > button {
            background-color: #262730 !important;
            color: #fafafa !important;
            border-color: #4a4a5a !important;
        }

        /* Toggle */
        [data-testid="stCheckbox"] label span {
            color: #fafafa !important;
        }

        /* Plotly charts - make background transparent */
        .js-plotly-plot .plotly .bg {
            fill: #0e1117 !important;
        }
    </style>
    """


def apply_theme(dark_mode: bool):
    """Apply dark or light theme CSS."""
    # Store in session state for chart functions to access
    st.session_state['dark_mode'] = dark_mode
    if dark_mode:
        st.markdown(get_dark_mode_css(), unsafe_allow_html=True)


def apply_dark_mode_to_figure(fig):
    """Apply dark mode styling to a Plotly figure if dark mode is enabled."""
    if st.session_state.get('dark_mode', False):
        fig.update_layout(
            paper_bgcolor='rgba(14, 17, 23, 1)',
            plot_bgcolor='rgba(14, 17, 23, 1)',
            font_color='#fafafa',
            title_font_color='#fafafa',
            legend_font_color='#fafafa',
            xaxis=dict(
                gridcolor='#3a3a4a',
                linecolor='#3a3a4a',
                tickfont=dict(color='#fafafa'),
                title_font=dict(color='#fafafa')
            ),
            yaxis=dict(
                gridcolor='#3a3a4a',
                linecolor='#3a3a4a',
                tickfont=dict(color='#fafafa'),
                title_font=dict(color='#fafafa')
            )
        )
        # Handle faceted charts with multiple axes
        for key in list(fig.layout):
            if key.startswith('xaxis') or key.startswith('yaxis'):
                fig.layout[key].update(
                    gridcolor='#3a3a4a',
                    linecolor='#3a3a4a',
                    tickfont=dict(color='#fafafa'),
                    title_font=dict(color='#fafafa')
                )
    return fig


def plotly_chart_dark(fig, **kwargs):
    """Wrapper for st.plotly_chart that applies dark mode styling."""
    apply_dark_mode_to_figure(fig)
    st.plotly_chart(fig, **kwargs)


def render_dataframe(df, column_config=None, use_container_width=True, hide_index=True, **kwargs):
    """Render a DataFrame with dark mode support.

    When dark mode is enabled, renders as styled HTML table.
    Otherwise uses st.dataframe for full interactivity.
    """
    if st.session_state.get('dark_mode', False):
        # Dark mode: render as styled HTML table
        # Generate HTML table with proper dark styling
        html_table = df.to_html(classes='dark-table', index=not hide_index, escape=False, border=0)

        # Wrap in a div with inline styles to ensure dark mode
        styled_html = f'''<div style="overflow-x: auto;">
<style>
.dark-table {{
    width: 100%;
    border-collapse: collapse;
    background-color: #0e1117;
    color: #fafafa;
    font-size: 14px;
    border-radius: 8px;
    overflow: hidden;
}}
.dark-table th {{
    background-color: #1a1a2e;
    color: #fafafa;
    padding: 10px 12px;
    text-align: left;
    border-bottom: 2px solid #3a3a4a;
    font-weight: 600;
}}
.dark-table td {{
    padding: 8px 12px;
    border-bottom: 1px solid #3a3a4a;
    background-color: #0e1117;
    color: #fafafa;
}}
.dark-table tr:nth-child(even) td {{
    background-color: #141420;
}}
.dark-table tr:hover td {{
    background-color: #262730;
}}
.dark-table thead tr {{
    background-color: #1a1a2e;
}}
</style>
{html_table}
</div>'''
        st.markdown(styled_html, unsafe_allow_html=True)
    else:
        # Light mode: use regular st.dataframe for full interactivity
        st.dataframe(df, column_config=column_config, use_container_width=use_container_width,
                     hide_index=hide_index, **kwargs)


# ============================================================================
# DATA LOADING
# ============================================================================

# Default data directory relative to this script
_DEFAULT_DATA_DIR = str(Path(__file__).parent.parent / 'data' / 'gold')


@st.cache_data
def load_gold_data(data_dir: str = None):
    """Load all gold layer parquet files."""
    if data_dir is None:
        data_dir = _DEFAULT_DATA_DIR
    data_path = Path(data_dir)
    data = {}

    files = {
        'model_summary': 'model_summary.parquet',
        'model_summary_by_run': 'model_summary_by_run.parquet',
        'provider_comparison': 'provider_comparison.parquet',
        'reliability_analysis': 'reliability_analysis.parquet',
        'run_comparison': 'run_comparison.parquet',
        'error_root_causes': 'error_root_causes.parquet',
        'run_status_summary': 'run_status_summary.parquet',
        'power_efficiency': 'power_efficiency.parquet',
        'provider_thermal_profile': 'provider_thermal_profile.parquet',
        'system_summary': 'system_summary.parquet',
        'system_comparison': 'system_comparison.parquet'
    }

    for key, filename in files.items():
        file_path = data_path / filename
        if file_path.exists():
            data[key] = pd.read_parquet(file_path)
        else:
            data[key] = pd.DataFrame()

    return data


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_provider_color(provider: str) -> str:
    """Get consistent color for each provider."""
    colors = {
        'cpu': '#E74C3C',       # Red
        'dml': '#F39C12',       # Orange
        'vitisai': '#27AE60',   # Green
    }
    return colors.get(provider.lower(), '#999999')


def render_system_badge(system_name: str, size: str = 'small') -> str:
    """Render a colored badge for system identification.

    Args:
        system_name: Name of the system to display
        size: 'small' or 'large' for different badge sizes

    Returns:
        HTML string for the badge, or empty string if no system_name
    """
    if not system_name:
        return ""

    # Generate consistent color from system name
    colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe']
    color_idx = hash(system_name) % len(colors)
    color = colors[color_idx]

    if size == 'small':
        return f'<span style="background: {color}; color: white; padding: 2px 6px; border-radius: 4px; font-size: 10px;">{system_name}</span>'
    else:
        return f'<span style="background: {color}; color: white; padding: 4px 10px; border-radius: 6px; font-size: 12px;">{system_name}</span>'


def get_provider_colors() -> dict:
    """Get provider color mapping for plots."""
    return {'cpu': '#E74C3C', 'dml': '#F39C12', 'vitisai': '#27AE60'}


def format_metric(value, suffix='', decimals=2):
    """Format metric value for display."""
    if pd.isna(value):
        return 'N/A'
    return f"{value:.{decimals}f}{suffix}"


def clean_facet_labels(fig):
    """Clean up facet labels by removing 'column_name=' prefix.

    Plotly facet_col creates labels like 'system_name=Dell Pro16 - System 1'.
    This function strips the prefix to show just 'Dell Pro16 - System 1'.
    """
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1] if "=" in a.text else a.text))
    return fig


def render_chart_help(title: str, explanation: str, show_help: bool):
    """Render expandable help section above charts.

    Args:
        title: Chart title for the help section
        explanation: Markdown-formatted explanation text
        show_help: Boolean flag from sidebar checkbox
    """
    if show_help:
        with st.expander(f"ℹ️ About: {title}", expanded=False):
            st.markdown(explanation)


def create_enhanced_hover_template(metric_type: str, include_speedup: bool = False) -> str:
    """Create enhanced hover template for consistent tooltips across charts.

    Args:
        metric_type: Type of metric ('throughput', 'latency', 'power', 'efficiency')
        include_speedup: Whether to include speedup information

    Returns:
        Hover template string for Plotly charts
    """
    templates = {
        'throughput': (
            '<b>%{x}</b><br>'
            '<b>Provider:</b> %{customdata[0]}<br>'
            '<b>Throughput:</b> %{y:.1f} ips<br>'
            '<i>💡 Higher is better</i>'
        ),
        'throughput_speedup': (
            '<b>%{x}</b><br>'
            '<b>Provider:</b> %{customdata[0]}<br>'
            '<b>Throughput:</b> %{y:.1f} ips<br>'
            '<b>Speedup vs CPU:</b> %{customdata[1]:.1f}x<br>'
            '<i>💡 Higher is better</i>'
        ),
        'latency': (
            '<b>%{x}</b><br>'
            '<b>Provider:</b> %{customdata[0]}<br>'
            '<b>Latency:</b> %{y:.2f} ms<br>'
            '<i>💡 Lower is better</i>'
        ),
        'power': (
            '<b>%{x}</b><br>'
            '<b>Provider:</b> %{customdata[0]}<br>'
            '<b>Power:</b> %{y:.1f} W<br>'
            '<i>💡 Lower power = better efficiency</i>'
        ),
        'efficiency': (
            '<b>%{x}</b><br>'
            '<b>Provider:</b> %{customdata[0]}<br>'
            '<b>Efficiency:</b> %{y:.2f} inf/W<br>'
            '<i>💡 Higher = more inferences per watt</i>'
        ),
        'stability': (
            '<b>%{x}</b><br>'
            '<b>Provider:</b> %{customdata[0]}<br>'
            '<b>Stability Score:</b> %{y:.1f}<br>'
            '<i>💡 Higher = more consistent latency</i>'
        ),
    }

    key = f'{metric_type}_speedup' if include_speedup and f'{metric_type}_speedup' in templates else metric_type
    return templates.get(key, templates['throughput']) + '<extra></extra>'


def render_export_buttons(df: pd.DataFrame, filename_prefix: str, key_suffix: str = ""):
    """Render export buttons for downloading data as CSV or JSON.

    Args:
        df: DataFrame to export
        filename_prefix: Prefix for the download filename
        key_suffix: Unique suffix for Streamlit keys to avoid duplicates
    """
    if df.empty:
        return

    col1, col2, col3 = st.columns([1, 1, 4])

    with col1:
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="📥 CSV",
            data=csv_data,
            file_name=f"{filename_prefix}.csv",
            mime="text/csv",
            key=f"csv_{filename_prefix}_{key_suffix}",
            help="Download data as CSV file"
        )

    with col2:
        json_data = df.to_json(orient='records', indent=2)
        st.download_button(
            label="📥 JSON",
            data=json_data,
            file_name=f"{filename_prefix}.json",
            mime="application/json",
            key=f"json_{filename_prefix}_{key_suffix}",
            help="Download data as JSON file"
        )


# ============================================================================
# TAB 1: EXECUTIVE SUMMARY
# ============================================================================

def generate_executive_narrative(model_summary, provider_comparison, run_comparison):
    """Generate dynamic executive summary narrative from benchmark data."""
    if model_summary.empty or provider_comparison.empty:
        return None

    # Calculate key metrics
    total_models = model_summary['model_clean'].nunique()

    # Get provider stats
    vitisai_stats = provider_comparison[provider_comparison['provider'] == 'vitisai']
    dml_stats = provider_comparison[provider_comparison['provider'] == 'dml']
    cpu_stats = provider_comparison[provider_comparison['provider'] == 'cpu']

    vitisai_avg_speedup = vitisai_stats['avg_speedup_vs_cpu'].values[0] if not vitisai_stats.empty else 0
    vitisai_max_speedup = vitisai_stats['max_speedup_vs_cpu'].values[0] if not vitisai_stats.empty else 0
    vitisai_win_rate = vitisai_stats['win_rate_pct'].values[0] if not vitisai_stats.empty else 0
    vitisai_wins = int(vitisai_stats['total_wins'].values[0]) if not vitisai_stats.empty else 0

    dml_avg_speedup = dml_stats['avg_speedup_vs_cpu'].values[0] if not dml_stats.empty else 0
    dml_max_speedup = dml_stats['max_speedup_vs_cpu'].values[0] if not dml_stats.empty else 0
    dml_wins = int(dml_stats['total_wins'].values[0]) if not dml_stats.empty else 0

    # Find top NPU performers (>15x speedup)
    vitisai_models = model_summary[model_summary['provider'] == 'vitisai'].copy()
    top_npu_models = vitisai_models[vitisai_models['speedup_vs_cpu'] > 15]['model_clean'].unique().tolist()

    # Find NPU weak spots (<3x speedup)
    weak_npu_models = vitisai_models[vitisai_models['speedup_vs_cpu'] < 3]['model_clean'].unique().tolist()

    # Find models where each provider wins (by unique model, not configuration)
    winners_by_model = model_summary.loc[
        model_summary.groupby('model_clean')['throughput_mean_ips'].idxmax()
    ]
    dml_winner_models = winners_by_model[winners_by_model['provider'] == 'dml']['model_clean'].tolist()

    # Calculate actual win rate by unique model (more accurate than per-config)
    vitisai_unique_wins = len(winners_by_model[winners_by_model['provider'] == 'vitisai'])
    dml_unique_wins = len(dml_winner_models)
    vitisai_win_rate_actual = (vitisai_unique_wins / total_models * 100) if total_models > 0 else 0

    # Real-time capable models (p50 < 33ms for 30fps)
    realtime_capable = model_summary[
        (model_summary['provider'] == 'vitisai') &
        (model_summary['latency_p50_ms'] < 33)
    ]['model_clean'].nunique()

    # High throughput models (>200 ips on NPU)
    high_throughput_npu = vitisai_models[vitisai_models['throughput_mean_ips'] > 200]['model_clean'].unique().tolist()

    # Run consistency issues (if available)
    consistency_issues = []
    if not run_comparison.empty and 'abs_delta_pct' in run_comparison.columns:
        inconsistent = run_comparison[run_comparison['abs_delta_pct'] > 20]
        if not inconsistent.empty:
            consistency_issues = inconsistent.groupby('provider').size().to_dict()

    return {
        'total_models': total_models,
        'vitisai_avg_speedup': vitisai_avg_speedup,
        'vitisai_max_speedup': vitisai_max_speedup,
        'vitisai_win_rate': vitisai_win_rate_actual,  # Use accurate per-model win rate
        'vitisai_wins': vitisai_unique_wins,  # Use unique model wins
        'dml_avg_speedup': dml_avg_speedup,
        'dml_max_speedup': dml_max_speedup,
        'dml_wins': dml_unique_wins,
        'top_npu_models': top_npu_models,
        'weak_npu_models': weak_npu_models,
        'dml_winner_models': dml_winner_models,
        'realtime_capable': realtime_capable,
        'high_throughput_npu': high_throughput_npu,
        'consistency_issues': consistency_issues,
    }


def render_executive_summary(data, show_chart_help=False):
    """Render executive summary tab."""
    st.header("Executive Summary")

    # Dynamic header based on selected systems
    system_summary = data.get('system_summary', pd.DataFrame())
    if not system_summary.empty:
        system_count = len(system_summary)
        if system_count == 1:
            sys_name = system_summary.iloc[0].get('system_name', 'Dell Pro 16 with AMD RyzenAI')
            st.markdown(f"**{sys_name}** | Benchmark Analysis Dashboard")
        else:
            st.markdown(f"**{system_count} Systems Compared** | Benchmark Analysis Dashboard")
    else:
        st.markdown("**Dell Pro 16 with AMD RyzenAI** | Benchmark Analysis Dashboard")

    model_summary = data['model_summary']
    provider_comparison = data['provider_comparison']
    run_status = data['run_status_summary']
    run_comparison = data.get('run_comparison', pd.DataFrame())

    if model_summary.empty:
        st.warning("""
        **No data available.**

        This could be because:
        1. The data pipeline hasn't been run yet
        2. All selected filters resulted in no matching data
        3. There was an error loading the parquet files

        **To fix:** Run the pipeline from the project directory:
        ```bash
        python scripts/run_pipeline.py --raw-dirs "path/to/your/benchmark/data"
        ```
        """)
        return

    # Generate narrative data
    narrative = generate_executive_narrative(model_summary, provider_comparison, run_comparison)

    # =========================================================================
    # EXECUTIVE NARRATIVE SECTION
    # =========================================================================
    if narrative:
        st.markdown("---")

        # Main Summary Paragraph
        st.markdown("### Key Findings")

        summary_text = f"""
The AMD RyzenAI NPU (VitisAI) demonstrates **strong performance advantages** across the majority of tested workloads,
achieving an average **{narrative['vitisai_avg_speedup']:.1f}x speedup** over CPU execution with peaks of **{narrative['vitisai_max_speedup']:.1f}x**
on optimized models. The NPU wins **{narrative['vitisai_win_rate']:.0f}%** of model comparisons ({narrative['vitisai_wins']} out of {narrative['total_models']} models),
establishing it as the recommended execution provider for most AI inference workloads on this platform.
The DirectML GPU provider shows more modest gains at **{narrative['dml_avg_speedup']:.1f}x** average speedup,
winning only {narrative['dml_wins']} model{'s' if narrative['dml_wins'] != 1 else ''}.
        """
        st.markdown(summary_text)

        # Two-column layout for positive/negative findings
        col_pos, col_neg = st.columns(2)

        with col_pos:
            st.markdown("#### Positive Conclusions")

            positives = []
            if narrative['top_npu_models']:
                models_list = ", ".join(narrative['top_npu_models'][:5])
                positives.append(f"**Excellent NPU acceleration (>15x):** {models_list}")

            if narrative['realtime_capable'] > 0:
                positives.append(f"**Real-time capable:** {narrative['realtime_capable']} models achieve <33ms latency on NPU (suitable for 30fps video)")

            if narrative['high_throughput_npu']:
                models_list = ", ".join(narrative['high_throughput_npu'][:4])
                positives.append(f"**High throughput (>200 ips):** {models_list}")

            positives.append(f"**Consistent NPU stability:** VitisAI shows higher stability scores than DML across most models")

            for p in positives:
                st.markdown(f"- {p}")

        with col_neg:
            st.markdown("#### Areas of Concern")

            negatives = []
            if narrative['weak_npu_models']:
                models_list = ", ".join(narrative['weak_npu_models'][:4])
                negatives.append(f"**Limited NPU benefit (<3x speedup):** {models_list} - these may need model optimization or are better suited for CPU/GPU")

            if narrative['dml_winner_models']:
                models_list = ", ".join(narrative['dml_winner_models'])
                negatives.append(f"**DML outperforms NPU on:** {models_list} - consider GPU for these workloads")

            if narrative['consistency_issues']:
                dml_issues = narrative['consistency_issues'].get('dml', 0)
                if dml_issues > 5:
                    negatives.append(f"**Run-to-run variance:** DML shows >20% throughput variation across {dml_issues} configurations - may affect production reliability")

            if not negatives:
                negatives.append("No significant issues identified in current benchmark data")

            for n in negatives:
                st.markdown(f"- {n}")

        # Best-Fit Applications
        st.markdown("#### Best-Fit Applications for NPU")
        app_col1, app_col2, app_col3 = st.columns(3)

        with app_col1:
            st.markdown("""
**Object Detection**
- YOLO variants (v3, v5, v8, X)
- Real-time video analytics
- Security/surveillance systems
            """)

        with app_col2:
            st.markdown("""
**Image Classification**
- ResNet, EfficientNet, Inception
- Quality inspection
- Medical imaging triage
            """)

        with app_col3:
            st.markdown("""
**Image Enhancement**
- SESR (super-resolution)
- Video upscaling
- Photo enhancement apps
            """)

        # Recommended Actions
        st.markdown("#### Recommended Actions")
        st.markdown("""
1. **Deploy NPU-first strategy** for object detection and image classification workloads - expect 10-35x speedup over CPU
2. **Use DirectML (GPU)** for movenet (pose estimation) where it outperforms NPU
3. **Investigate optimization** for PAN, yolov5s, and mobilenet_v2 which show suboptimal NPU acceleration (<3x)
4. **Validate in production** any DML configurations showing high run-to-run variance before deployment
5. **Consider CPU fallback** only for models where accelerator overhead exceeds benefit (none identified in this benchmark)
        """)

        st.markdown("---")

    # Dynamic Data Quality Note based on actual errors
    error_root_causes = data.get('error_root_causes', pd.DataFrame())
    if not error_root_causes.empty:
        excluded_items = []
        for _, row in error_root_causes.iterrows():
            model = row.get('model_clean', 'unknown')
            provider = row.get('provider', 'unknown')
            error_cat = row.get('error_category', 'unknown')
            excluded_items.append(f"- `{model}` + {provider.upper()} - {error_cat}")

        if excluded_items:
            exclusion_list = '\n    '.join(excluded_items[:10])
            st.info(f"""
    **Data Quality Note:** The following model+provider combinations were excluded due to benchmark failures:
    {exclusion_list}

    These configurations had errors during benchmarking and are not included in performance analysis.
            """)

    # =========================================================================
    # KEY METRICS SECTION
    # =========================================================================
    st.markdown("### Performance Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_models = model_summary['model_clean'].nunique()
        st.metric("Models Tested", total_models, help="Total number of unique AI models benchmarked across all providers")

    with col2:
        total_runs = len(run_status) if not run_status.empty else 0
        st.metric("Benchmark Runs", total_runs, help="Number of independent benchmark executions (run01, run02, etc.) for reproducibility validation")

    with col3:
        if not provider_comparison.empty:
            best_provider = provider_comparison.loc[provider_comparison['avg_throughput_ips'].idxmax(), 'provider']
            st.metric("Fastest Provider", best_provider.upper(), help="Provider with highest average throughput (inferences per second) across all models")
        else:
            st.metric("Fastest Provider", "N/A")

    with col4:
        if not provider_comparison.empty:
            max_speedup = provider_comparison['max_speedup_vs_cpu'].max()
            st.metric("Max Speedup vs CPU", f"{max_speedup:.1f}x", help="Best-case speedup: throughput of fastest accelerator divided by CPU throughput for the same model. A value of 10x means 10 times faster than CPU.")
        else:
            st.metric("Max Speedup vs CPU", "N/A")

    st.divider()

    # Provider performance cards
    st.subheader("Provider Performance Overview")
    st.caption("**Terminology:** *ips* = inferences per second (higher is better) | *Win* = provider with highest throughput for a model | *Win Rate* = % of models where this provider was fastest")

    if not provider_comparison.empty:
        cols = st.columns(len(provider_comparison))

        for idx, (_, row) in enumerate(provider_comparison.iterrows()):
            with cols[idx]:
                provider = row['provider']
                color = get_provider_color(provider)

                st.markdown(f"""
                <div style="background-color: {color}22; padding: 15px; border-radius: 10px; border-left: 4px solid {color};">
                    <h3 style="margin: 0; color: {color};">{provider.upper()}</h3>
                    <p style="margin: 5px 0;"><b>Avg Throughput:</b> {row['avg_throughput_ips']:.1f} ips</p>
                    <p style="margin: 5px 0;"><b>Avg Latency:</b> {row['avg_latency_ms']:.1f} ms</p>
                    <p style="margin: 5px 0;"><b>Model Wins:</b> {row['total_wins']}</p>
                    <p style="margin: 5px 0;"><b>Win Rate:</b> {row['win_rate_pct']:.0f}%</p>
                </div>
                """, unsafe_allow_html=True)

    st.divider()

    # Throughput comparison chart
    st.subheader("Throughput by Model")

    render_chart_help(
        "Throughput by Model",
        """
**What this shows:** Throughput (inferences per second) comparison across all models and providers.

**How to interpret:**
- Higher bars indicate better performance
- Grouped bars allow direct comparison between CPU, GPU (DML), and NPU (VitisAI)
- Look for models where accelerators provide significant speedups over CPU
        """,
        show_chart_help
    )

    if not model_summary.empty:
        # Check if multiple systems for faceting
        has_multiple_systems = 'system_name' in model_summary.columns and model_summary['system_name'].nunique() > 1

        if has_multiple_systems:
            fig = px.bar(
                model_summary,
                x='model_clean',
                y='throughput_mean_ips',
                color='provider',
                facet_col='system_name',
                barmode='group',
                color_discrete_map=get_provider_colors(),
                title='Throughput Comparison (Inferences per Second)',
                labels={'model_clean': 'Model', 'throughput_mean_ips': 'Throughput (ips)'}
            )
            # Update hover to include system
            fig.update_traces(
                hovertemplate='<b>%{x}</b><br>Throughput: %{y:.1f} ips<br><i>(Higher is better)</i><extra></extra>'
            )
            fig.update_layout(height=500)
            clean_facet_labels(fig)
        else:
            fig = px.bar(
                model_summary,
                x='model_clean',
                y='throughput_mean_ips',
                color='provider',
                barmode='group',
                color_discrete_map=get_provider_colors(),
                title='Throughput Comparison (Inferences per Second)',
                labels={'model_clean': 'Model', 'throughput_mean_ips': 'Throughput (ips)'},
                hover_data={'model_clean': True, 'provider': True, 'throughput_mean_ips': ':.1f'}
            )
            fig.update_traces(
                hovertemplate='<b>%{x}</b><br>Provider: %{customdata[1]}<br>Throughput: %{y:.1f} ips<br><i>(Higher is better)</i><extra></extra>',
                customdata=model_summary[['model_clean', 'provider']].values
            )
            fig.update_layout(height=500)

        fig.update_layout(
            xaxis_title='Model',
            yaxis_title='Throughput (inferences per second)',
            xaxis_tickangle=-45
        )
        plotly_chart_dark(fig, use_container_width=True)


# ============================================================================
# TAB 2: PROVIDER COMPARISON
# ============================================================================

def render_provider_comparison(data, show_chart_help=False):
    """Render provider comparison tab."""
    st.header("Provider Comparison")

    model_summary = data['model_summary']
    provider_comparison = data['provider_comparison']

    if model_summary.empty:
        st.warning("No data available.")
        return

    # Check if multiple systems
    has_multiple_systems = 'system_name' in model_summary.columns and model_summary['system_name'].nunique() > 1

    # Provider summary table
    st.subheader("Provider Summary")

    if not provider_comparison.empty:
        # Build display columns, including system if multiple systems
        display_cols = ['provider']
        if has_multiple_systems and 'system_name' in provider_comparison.columns:
            display_cols.append('system_name')
        display_cols.extend([
            'models_tested', 'total_wins', 'win_rate_pct',
            'avg_throughput_ips', 'avg_latency_ms', 'avg_speedup_vs_cpu'
        ])
        display_cols = [c for c in display_cols if c in provider_comparison.columns]

        # Make a copy for display and hide speedup for CPU (it's always ~1.0x by definition)
        display_df = provider_comparison[display_cols].copy()
        if 'avg_speedup_vs_cpu' in display_df.columns:
            display_df.loc[display_df['provider'] == 'cpu', 'avg_speedup_vs_cpu'] = float('nan')

        # Configure columns for better display
        column_config = {
            'provider': st.column_config.TextColumn('Provider', help='Execution provider'),
            'system_name': st.column_config.TextColumn('System', help='Benchmark system'),
            'models_tested': st.column_config.NumberColumn('Models Tested', help='Number of models tested'),
            'total_wins': st.column_config.NumberColumn('Total Wins', help='Number of models where this provider was fastest'),
            'win_rate_pct': st.column_config.NumberColumn('Win Rate %', format='%.0f%%', help='Percentage of models won'),
            'avg_throughput_ips': st.column_config.NumberColumn('Avg Throughput (ips)', format='%.1f', help='Average inferences per second'),
            'avg_latency_ms': st.column_config.NumberColumn('Avg Latency (ms)', format='%.1f', help='Average latency in milliseconds'),
            'avg_speedup_vs_cpu': st.column_config.NumberColumn('Speedup vs CPU', format='%.2fx', help='Average speedup compared to CPU baseline'),
        }

        render_dataframe(
            display_df,
            column_config=column_config,
            use_container_width=True,
            hide_index=True
        )
        st.caption("Click column headers to sort")

        # Export buttons for provider summary
        render_export_buttons(display_df, "provider_comparison", "prov_comp")

    st.divider()

    # Speedup heatmap - only show accelerators (DML, VitisAI), not CPU
    st.subheader("Accelerator Speedup vs CPU Heatmap")

    render_chart_help(
        "Accelerator Speedup vs CPU Heatmap",
        """
**What this shows:** A heatmap visualizing how much faster each accelerator (DML/GPU, VitisAI/NPU) is compared to CPU baseline.

**How to interpret:**
- Values > 1.0 mean faster than CPU (green = good)
- Values < 1.0 mean slower than CPU (red = poor)
- Higher values indicate better acceleration
- The color scale ranges from red (slow) through yellow to green (fast)

**Note:** CPU is excluded as the baseline (it would always show 1.0x).
        """,
        show_chart_help
    )

    # Filter to only accelerators (exclude CPU - it's always 1.0x vs itself)
    accelerator_data = model_summary[model_summary['provider'] != 'cpu']

    # View mode toggle for multiple systems
    if has_multiple_systems:
        view_mode = st.radio(
            "View Mode",
            ["Combined", "By System"],
            horizontal=True,
            key="provider_speedup_view_mode",
            help="Combined: aggregate across systems. By System: separate view per system."
        )
    else:
        view_mode = "Combined"

    if view_mode == "By System" and has_multiple_systems:
        # Create tabbed heatmaps per system
        system_names = accelerator_data['system_name'].unique().tolist()
        system_tabs = st.tabs([f"📊 {s}" for s in system_names])

        for idx, system_name in enumerate(system_names):
            with system_tabs[idx]:
                system_data = accelerator_data[accelerator_data['system_name'] == system_name]
                pivot_speedup = system_data.pivot_table(
                    index='model_clean',
                    columns='provider',
                    values='speedup_vs_cpu',
                    aggfunc='mean'
                )

                if not pivot_speedup.empty:
                    fig = px.imshow(
                        pivot_speedup,
                        labels=dict(x="Accelerator", y="Model", color="Speedup vs CPU"),
                        color_continuous_scale='RdYlGn',
                        aspect='auto',
                        text_auto='.1f'
                    )
                    fig.update_traces(
                        hovertemplate='<b>%{y}</b> on %{x}<br>Speedup: %{z:.2f}x vs CPU<br><i>(>1.0 = faster than CPU)</i><extra></extra>'
                    )
                    fig.update_layout(height=500)
                    plotly_chart_dark(fig, use_container_width=True, key=f"speedup_heatmap_{idx}")
    else:
        # Combined view (original behavior)
        pivot_speedup = accelerator_data.pivot_table(
            index='model_clean',
            columns='provider',
            values='speedup_vs_cpu',
            aggfunc='mean'
        )

        if not pivot_speedup.empty:
            fig = px.imshow(
                pivot_speedup,
                labels=dict(x="Accelerator", y="Model", color="Speedup vs CPU"),
                color_continuous_scale='RdYlGn',
                aspect='auto',
                text_auto='.1f'
            )
            fig.update_traces(
                hovertemplate='<b>%{y}</b> on %{x}<br>Speedup: %{z:.2f}x vs CPU<br><i>(>1.0 = faster than CPU)</i><extra></extra>'
            )
            fig.update_layout(height=600)
            plotly_chart_dark(fig, use_container_width=True)

    st.caption("Values shown are speedup multipliers vs CPU baseline. Green (>1.0) = faster than CPU, Red (<1.0) = slower than CPU. CPU excluded as it's always 1.0x (the baseline).")

    # Latency comparison
    st.subheader("Latency Comparison")

    render_chart_help(
        "Latency Comparison",
        """
**What this shows:** Mean latency (in milliseconds) for each model-provider combination.

**How to interpret:**
- Lower bars indicate faster inference times (better)
- Compare grouped bars to see which provider offers the lowest latency per model
- For real-time applications, lower latency is critical
        """,
        show_chart_help
    )

    # Handle multiple systems with faceting
    if has_multiple_systems:
        fig = px.bar(
            model_summary,
            x='model_clean',
            y='latency_mean_ms',
            color='provider',
            facet_col='system_name',
            barmode='group',
            color_discrete_map=get_provider_colors(),
            title='Mean Latency by Model and Provider',
            labels={'model_clean': 'Model', 'latency_mean_ms': 'Latency (ms)'}
        )
        fig.update_traces(
            hovertemplate='<b>%{x}</b><br>Latency: %{y:.2f} ms<br><i>(Lower is better)</i><extra></extra>'
        )
        clean_facet_labels(fig)
    else:
        fig = px.bar(
            model_summary,
            x='model_clean',
            y='latency_mean_ms',
            color='provider',
            barmode='group',
            color_discrete_map=get_provider_colors(),
            title='Mean Latency by Model and Provider',
            labels={'model_clean': 'Model', 'latency_mean_ms': 'Latency (ms)'}
        )
        fig.update_traces(
            hovertemplate='<b>%{x}</b><br>Provider: %{fullData.name}<br>Latency: %{y:.2f} ms<br><i>(Lower is better)</i><extra></extra>'
        )

    fig.update_layout(
        xaxis_title='Model',
        yaxis_title='Latency (milliseconds)',
        xaxis_tickangle=-45,
        height=500
    )
    plotly_chart_dark(fig, use_container_width=True)
    st.caption("Latency = time to complete one inference. Lower values mean faster response times.")


# ============================================================================
# TAB 3: MODEL PERFORMANCE
# ============================================================================

def render_model_performance(data, show_chart_help=False):
    """Render model performance tab."""
    st.header("Model Performance")

    model_summary = data['model_summary']

    if model_summary.empty:
        st.warning("No data available.")
        return

    # Model selector with export option
    col_select, col_export = st.columns([3, 1])
    with col_select:
        models = sorted(model_summary['model_clean'].unique())
        selected_model = st.selectbox("Select Model", models)
    with col_export:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacing
        render_export_buttons(model_summary, "model_performance_data", "model_perf")

    model_data = model_summary[model_summary['model_clean'] == selected_model]

    if model_data.empty:
        st.warning(f"No data for model: {selected_model}")
        return

    st.divider()

    # Provider metrics cards
    st.subheader(f"Performance Metrics: {selected_model}")
    st.caption("**Metrics Guide:** *Throughput* = inferences/second (higher=better) | *P99 Latency* = worst-case latency for 99% of requests | *Speedup* = how many times faster than CPU")

    # Check if multiple systems for this model
    has_multiple_systems = 'system_name' in model_data.columns and model_data['system_name'].nunique() > 1

    if has_multiple_systems:
        # Group cards by system
        for system_name in model_data['system_name'].unique():
            st.markdown(f"**{system_name}**")
            system_model_data = model_data[model_data['system_name'] == system_name]
            cols = st.columns(len(system_model_data))

            for idx, (_, row) in enumerate(system_model_data.iterrows()):
                with cols[idx]:
                    provider = row['provider']
                    color = get_provider_color(provider)

                    # Only show speedup for accelerators (not CPU vs itself)
                    speedup_line = f"<p><b>Speedup vs CPU:</b> {row['speedup_vs_cpu']:.2f}x</p>" if provider != 'cpu' else ""

                    st.markdown(f"""
                    <div style="background-color: {color}22; padding: 15px; border-radius: 10px; border-left: 4px solid {color};">
                        <h4 style="margin: 0; color: {color};">{provider.upper()}</h4>
                        <p><b>Throughput:</b> {row['throughput_mean_ips']:.1f} ips</p>
                        <p><b>Mean Latency:</b> {row['latency_mean_ms']:.2f} ms</p>
                        <p><b>P99 Latency:</b> {row['latency_p99_ms']:.2f} ms</p>
                        {speedup_line}
                    </div>
                    """, unsafe_allow_html=True)
    else:
        # Original single-row display
        cols = st.columns(len(model_data))

        for idx, (_, row) in enumerate(model_data.iterrows()):
            with cols[idx]:
                provider = row['provider']
                color = get_provider_color(provider)

                # Add system indicator if available (single system case)
                system_indicator = ""
                if 'system_name' in row and pd.notna(row.get('system_name')):
                    system_indicator = f'<p style="font-size: 11px; color: #666;"><b>System:</b> {row["system_name"]}</p>'

                # Only show speedup for accelerators (not CPU vs itself)
                speedup_line = f"<p><b>Speedup vs CPU:</b> {row['speedup_vs_cpu']:.2f}x</p>" if provider != 'cpu' else ""

                st.markdown(f"""
                <div style="background-color: {color}22; padding: 15px; border-radius: 10px; border-left: 4px solid {color};">
                    <h4 style="margin: 0; color: {color};">{provider.upper()}</h4>
                    {system_indicator}
                    <p><b>Throughput:</b> {row['throughput_mean_ips']:.1f} ips</p>
                    <p><b>Mean Latency:</b> {row['latency_mean_ms']:.2f} ms</p>
                    <p><b>P99 Latency:</b> {row['latency_p99_ms']:.2f} ms</p>
                    {speedup_line}
                </div>
                """, unsafe_allow_html=True)

    st.divider()

    # Latency percentiles chart
    st.subheader("Latency Percentiles")

    render_chart_help(
        "Latency Percentiles",
        """
**What this shows:** Latency distribution at key percentiles (P50, P95, P99) for the selected model.

**How to interpret:**
- **P50 (median)**: Half of inferences are faster than this
- **P95**: 95% of inferences are faster than this (typical worst case)
- **P99**: 99% of inferences are faster than this (tail latency)
- A large gap between P50 and P99 indicates inconsistent performance
        """,
        show_chart_help
    )

    latency_data = []
    for _, row in model_data.iterrows():
        for pct, col in [('P50', 'latency_p50_ms'), ('P95', 'latency_p95_ms'), ('P99', 'latency_p99_ms')]:
            if col in row:
                entry = {
                    'provider': row['provider'],
                    'percentile': pct,
                    'latency_ms': row[col]
                }
                # Include system_name if available
                if 'system_name' in row:
                    entry['system_name'] = row['system_name']
                latency_data.append(entry)

    if latency_data:
        latency_df = pd.DataFrame(latency_data)

        # Check for multiple systems
        if has_multiple_systems and 'system_name' in latency_df.columns:
            fig = px.bar(
                latency_df,
                x='percentile',
                y='latency_ms',
                color='provider',
                facet_col='system_name',
                barmode='group',
                color_discrete_map=get_provider_colors(),
                labels={'percentile': 'Percentile', 'latency_ms': 'Latency (ms)'}
            )
            clean_facet_labels(fig)
        else:
            fig = px.bar(
                latency_df,
                x='percentile',
                y='latency_ms',
                color='provider',
                barmode='group',
                color_discrete_map=get_provider_colors(),
                labels={'percentile': 'Percentile', 'latency_ms': 'Latency (ms)'}
            )

        fig.update_layout(
            xaxis_title='Percentile',
            yaxis_title='Latency (ms)',
            height=400
        )
        plotly_chart_dark(fig, use_container_width=True)


# ============================================================================
# TAB 4: RELIABILITY ANALYSIS
# ============================================================================

def render_reliability_analysis(data, show_chart_help=False):
    """Render reliability analysis tab."""
    st.header("Reliability Analysis")
    st.markdown("Analyzes benchmark success rates and identifies error patterns across model+provider combinations.")

    reliability = data['reliability_analysis']
    errors = data['error_root_causes']

    if reliability.empty:
        st.warning("""
        **No reliability data available.**

        This could be because:
        - No models match the current filter selection
        - The reliability analysis wasn't generated during pipeline execution

        Try adjusting your filters in the sidebar or re-running the pipeline.
        """)
        return

    # Check if multiple systems
    has_multiple_systems = 'system_name' in reliability.columns and reliability['system_name'].nunique() > 1

    # Summary metrics with help tooltips
    col1, col2, col3 = st.columns(3)

    with col1:
        reliable_count = reliability['is_reliable'].sum()
        total_count = len(reliability)
        st.metric("Reliable Configs", f"{reliable_count}/{total_count}",
                  help="Configurations with 100% success rate across all benchmark runs. A 'config' is a unique model+provider combination.")

    with col2:
        avg_success = reliability['success_rate'].mean() * 100
        st.metric("Avg Success Rate", f"{avg_success:.1f}%",
                  help="Average percentage of successful benchmark iterations across all configurations. 100% means all inference runs completed without errors.")

    with col3:
        error_count = len(errors) if not errors.empty else 0
        st.metric("Error Patterns", error_count,
                  help="Number of distinct error patterns identified. Each pattern represents a unique failure mode (e.g., model incompatibility, memory issues).")

    # Per-system reliability breakdown (if multiple systems)
    if has_multiple_systems:
        st.divider()
        st.subheader("Reliability by System")

        sys_cols = st.columns(reliability['system_name'].nunique())
        for idx, system_name in enumerate(reliability['system_name'].unique()):
            system_data = reliability[reliability['system_name'] == system_name]
            reliable = system_data['is_reliable'].sum()
            total = len(system_data)
            avg_success_sys = system_data['success_rate'].mean() * 100

            with sys_cols[idx]:
                st.markdown(f"**{system_name}**")
                st.metric(f"Reliable", f"{reliable}/{total}")
                st.progress(avg_success_sys / 100, text=f"{avg_success_sys:.0f}% avg success")

    st.divider()

    # Success rate heatmap
    st.subheader("Success Rate by Model and Provider")

    render_chart_help(
        "Success Rate Heatmap",
        """
**What this shows:** A heatmap displaying the benchmark success rate for each model-provider combination.

**How to interpret:**
- **100% (green)**: All benchmark runs completed successfully
- **< 100% (yellow/red)**: Some runs failed due to errors
- White/missing cells indicate untested combinations
- Look for patterns of provider-specific failures
        """,
        show_chart_help
    )

    if has_multiple_systems:
        # Create tabbed view per system
        system_names = reliability['system_name'].unique().tolist()
        system_tabs = st.tabs([f"📊 {s}" for s in system_names])

        for idx, system_name in enumerate(system_names):
            with system_tabs[idx]:
                system_reliability = reliability[reliability['system_name'] == system_name]
                pivot_success = system_reliability.pivot_table(
                    index='model_clean',
                    columns='provider',
                    values='success_rate',
                    aggfunc='mean'
                )

                if not pivot_success.empty:
                    fig = px.imshow(
                        pivot_success * 100,
                        labels=dict(x="Provider", y="Model", color="Success %"),
                        color_continuous_scale='RdYlGn',
                        aspect='auto',
                        zmin=0,
                        zmax=100,
                        text_auto='.0f'
                    )
                    fig.update_traces(
                        hovertemplate='<b>%{y}</b> on %{x}<br>Success Rate: %{z:.0f}%<extra></extra>'
                    )
                    fig.update_layout(height=500)
                    plotly_chart_dark(fig, use_container_width=True, key=f"reliability_heatmap_{idx}")
    else:
        # Single system view (original)
        pivot_success = reliability.pivot_table(
            index='model_clean',
            columns='provider',
            values='success_rate',
            aggfunc='mean'
        )

        if not pivot_success.empty:
            fig = px.imshow(
                pivot_success * 100,
                labels=dict(x="Provider", y="Model", color="Success %"),
                color_continuous_scale='RdYlGn',
                aspect='auto',
                zmin=0,
                zmax=100,
                text_auto='.0f'
            )
            fig.update_traces(
                hovertemplate='<b>%{y}</b> on %{x}<br>Success Rate: %{z:.0f}%<extra></extra>'
            )
            fig.update_layout(height=600)
            plotly_chart_dark(fig, use_container_width=True)

    st.caption("Green (100%) = all benchmark iterations succeeded. Red (<100%) = some failures occurred. Check Error Details below for root causes.")

    # Error details
    if not errors.empty:
        st.divider()
        st.subheader("Error Details")

        for _, row in errors.iterrows():
            severity_color = '#E74C3C' if row.get('is_persistent', False) else '#F39C12'

            # Add system label if available
            system_label = f" [{row.get('system_name', '')}]" if row.get('system_name') else ""

            with st.expander(f"⚠️ {row['model_clean']} + {row['provider']}{system_label} - {row['error_category']}"):
                system_info = f"\n\n                **System:** {row.get('system_name', 'N/A')}" if row.get('system_name') else ""
                st.markdown(f"""
                **Severity:** {'Persistent' if row.get('is_persistent', False) else 'Intermittent'}{system_info}

                **Occurrences:** {row.get('occurrence_count', 1)}

                **Error Summary:**
                ```
                {row.get('error_summary', 'No details available')[:500]}
                ```

                **Recommendation:**
                {row.get('recommendation', 'Review error logs for more details.')}
                """)

    # Export reliability data
    st.divider()
    st.subheader("Export Reliability Data")
    render_export_buttons(reliability, "reliability_analysis", "reliability")


# ============================================================================
# TAB 5: RUN COMPARISON
# ============================================================================

def render_run_comparison(data, show_chart_help=False):
    """Render run comparison tab."""
    st.header("Run Comparison")
    st.markdown("""
    Compares benchmark results between multiple runs to validate **reproducibility**.
    Consistent results across runs indicate reliable benchmarks.
    """)

    run_comparison = data['run_comparison']
    run_status = data['run_status_summary']
    model_summary_by_run = data.get('model_summary_by_run', pd.DataFrame())

    if run_comparison.empty:
        st.warning("""
        **No run comparison data available.**

        Run comparisons require at least 2 benchmark runs to compare.
        Current filters may have limited data to a single run.

        Try:
        - Selecting more runs in the sidebar filters
        - Ensuring multiple runs (run01, run02, etc.) exist in the benchmark data
        """)
        return

    # Models being compared
    models_in_comparison = sorted(run_comparison['model_clean'].unique().tolist()) if 'model_clean' in run_comparison.columns else []
    providers_in_comparison = sorted(run_comparison['provider'].unique().tolist()) if 'provider' in run_comparison.columns else []

    # Visual confirmation of models being compared
    st.subheader("Comparison Scope")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Models Compared", len(models_in_comparison),
                  help="Number of unique AI models included in this run-to-run comparison")
        with st.expander("View Models", expanded=False):
            for model in models_in_comparison:
                st.markdown(f"- {model}")
    with col2:
        st.metric("Provider Configurations", len(run_comparison),
                  help="Total model+provider combinations being compared (e.g., resnet50+cpu, resnet50+vitisai, etc.)")
        st.caption(f"Providers: {', '.join([p.upper() for p in providers_in_comparison])}")

    st.divider()

    # Run status cards with per-provider breakdown
    if not run_status.empty:
        st.subheader("Run Status")

        # Check if multiple systems exist
        has_multiple_systems = 'system_name' in run_status.columns and run_status['system_name'].nunique() > 1

        if has_multiple_systems:
            # Group runs by system for clearer display
            for system_name in run_status['system_name'].unique():
                st.markdown(f"**{system_name}**")
                system_runs = run_status[run_status['system_name'] == system_name]
                cols = st.columns(len(system_runs))

                for idx, (_, row) in enumerate(system_runs.iterrows()):
                    run_name = row['run_name']
                    # Dark mode aware card colors
                    card_bg = '#262730' if st.session_state.get('dark_mode', False) else '#f0f2f6'
                    card_text = '#fafafa' if st.session_state.get('dark_mode', False) else '#262730'
                    with cols[idx]:
                        st.markdown(f"""
                        <div style="background-color: {card_bg}; padding: 15px; border-radius: 10px; color: {card_text};">
                            <h4 style="margin: 0; color: {card_text};">{run_name}</h4>
                            <p style="color: {card_text};"><b>Date:</b> {row.get('test_date', 'N/A')}</p>
                            <p style="color: {card_text};"><b>Configs:</b> {row['successful_configs']}/{row['total_configs']}</p>
                            <p style="color: {card_text};"><b>Avg Throughput:</b> {row['avg_throughput_ips']:.1f} ips</p>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            # Single system or no system info - show all runs in one row
            cols = st.columns(len(run_status))

            for idx, (_, row) in enumerate(run_status.iterrows()):
                run_name = row['run_name']
                # Add system badge if system_name is available
                system_name = row.get('system_name', '')
                system_badge = render_system_badge(system_name) if system_name else ''
                # Dark mode aware card colors
                card_bg = '#262730' if st.session_state.get('dark_mode', False) else '#f0f2f6'
                card_text = '#fafafa' if st.session_state.get('dark_mode', False) else '#262730'

                with cols[idx]:
                    st.markdown(f"""
                    <div style="background-color: {card_bg}; padding: 15px; border-radius: 10px; color: {card_text};">
                        <h4 style="margin: 0; color: {card_text};">{run_name} {system_badge}</h4>
                        <p style="color: {card_text};"><b>Date:</b> {row.get('test_date', 'N/A')}</p>
                        <p style="color: {card_text};"><b>Configs:</b> {row['successful_configs']}/{row['total_configs']}</p>
                        <p style="color: {card_text};"><b>Avg Throughput:</b> {row['avg_throughput_ips']:.1f} ips</p>
                    </div>
                    """, unsafe_allow_html=True)

        # Per-provider averages for each run
        if not model_summary_by_run.empty and 'run_name' in model_summary_by_run.columns:
            st.markdown("#### Per-Provider Averages (Aggregate)")
            st.caption("⚠️ **These are AVERAGED values** across all models for each provider. The 'Throughput Change' shows how the provider's average performance changed between runs.")

            # Calculate per-provider stats for each run
            provider_stats = model_summary_by_run.groupby(['run_name', 'provider']).agg({
                'throughput_mean_ips': 'mean',
                'latency_mean_ms': 'mean'
            }).reset_index()

            # Create a comparison table
            if not provider_stats.empty:
                runs = sorted(provider_stats['run_name'].unique())
                providers = sorted(provider_stats['provider'].unique())

                # Build comparison data
                comparison_rows = []
                for provider in providers:
                    row_data = {'Provider': provider.upper()}
                    for run in runs:
                        run_data = provider_stats[(provider_stats['run_name'] == run) & (provider_stats['provider'] == provider)]
                        if not run_data.empty:
                            throughput = run_data['throughput_mean_ips'].values[0]
                            latency = run_data['latency_mean_ms'].values[0]
                            row_data[f'{run} Throughput (ips)'] = f"{throughput:.1f}"
                            row_data[f'{run} Latency (ms)'] = f"{latency:.1f}"
                    comparison_rows.append(row_data)

                if comparison_rows:
                    comparison_df = pd.DataFrame(comparison_rows)

                    # Calculate deltas if we have both runs
                    if len(runs) == 2 and f'{runs[0]} Throughput (ips)' in comparison_df.columns and f'{runs[1]} Throughput (ips)' in comparison_df.columns:
                        comparison_df['Throughput Change'] = comparison_df.apply(
                            lambda r: f"{((float(r[f'{runs[1]} Throughput (ips)']) - float(r[f'{runs[0]} Throughput (ips)'])) / float(r[f'{runs[0]} Throughput (ips)']) * 100):+.1f}%" if r[f'{runs[0]} Throughput (ips)'] != '0.0' else 'N/A',
                            axis=1
                        )

                    render_dataframe(comparison_df, use_container_width=True, hide_index=True)
                    st.caption("*Throughput Change = (Run2 - Run1) / Run1 × 100%. Positive = improved in Run2.*")

    st.divider()

    # Performance delta chart
    st.subheader("Throughput Change (Run1 → Run2) - Per Model")
    st.info("""
    📊 **This chart shows INDIVIDUAL model changes**, not aggregates. Each bar represents a single model+provider combination.
    Compare this to the "Per-Provider Averages" table above which shows averaged values across all models.
    """)

    render_chart_help(
        "Throughput Change Between Runs",
        """
**What this shows:** Percentage change in throughput between benchmark runs for EACH model+provider combination.

**How to interpret:**
- **Each bar** = one model on one provider (e.g., "resnet50 on VitisAI")
- **Positive values (above 0)**: That specific model improved in Run2
- **Negative values (below 0)**: That specific model was slower in Run2
- **Values near zero**: Consistent performance between runs (good!)
- Large variations (>10%) may indicate system instability

**Why values differ from the aggregate table:**
The "Per-Provider Averages" table above shows the AVERAGE change across all models.
This chart shows the change for EACH individual model, which can vary significantly.
        """,
        show_chart_help
    )

    if 'throughput_delta_pct' in run_comparison.columns:
        # Get valid comparison data
        sorted_data = run_comparison.dropna(subset=['throughput_delta_pct']).copy()

        if not sorted_data.empty:
            # Check if we have system info for faceting
            has_system_info = 'system_name' in sorted_data.columns and sorted_data['system_name'].nunique() > 1

            if has_system_info:
                # Get all unique models across both systems for consistent x-axis
                all_models = sorted(sorted_data['model_clean'].unique())

                # Use facet_col for consistent formatting
                fig = px.bar(
                    sorted_data,
                    x='model_clean',
                    y='throughput_delta_pct',
                    color='provider',
                    facet_col='system_name',
                    barmode='group',
                    color_discrete_map=get_provider_colors(),
                    title='Per-Model Throughput Change: (Run2 - Run1) / Run1 × 100%',
                    labels={'model_clean': 'Model', 'throughput_delta_pct': 'Change (%)'},
                    category_orders={'model_clean': all_models}  # Consistent order
                )

                # Clean facet labels
                clean_facet_labels(fig)

                # Update hover template
                fig.update_traces(
                    hovertemplate='<b>%{x}</b><br>Change: %{y:+.1f}%<extra></extra>'
                )

                # Add reference lines
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                fig.add_hline(y=10, line_dash="dot", line_color="orange")
                fig.add_hline(y=-10, line_dash="dot", line_color="orange")

                fig.update_layout(
                    height=550,
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                    yaxis_title='Throughput Change (%)'
                )

                # Better x-axis labels
                fig.update_xaxes(tickangle=-60, tickfont=dict(size=9))

            else:
                sorted_data = sorted_data.sort_values('throughput_delta_pct')
                fig = px.bar(
                    sorted_data,
                    x='model_clean',
                    y='throughput_delta_pct',
                    color='provider',
                    color_discrete_map=get_provider_colors(),
                    title='Per-Model Throughput Change: (Run2 - Run1) / Run1 × 100%',
                    barmode='group'
                )
                fig.update_traces(
                    hovertemplate='<b>%{x}</b><br>Change: %{y:+.1f}%<extra></extra>'
                )
                fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="No change")
                fig.add_hline(y=10, line_dash="dot", line_color="orange", annotation_text="+10% threshold")
                fig.add_hline(y=-10, line_dash="dot", line_color="orange", annotation_text="-10% threshold")
                fig.update_layout(
                    xaxis_title='Model',
                    yaxis_title='Throughput Change (%)',
                    xaxis_tickangle=-60,
                    height=550
                )

            apply_dark_mode_to_figure(fig)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Each bar = one model+provider. Bars close to 0% indicate consistent, reproducible performance.")

    # Summary table
    st.subheader("Comparison Details (Individual Models)")
    st.caption("Each row shows how a specific model+provider performed between runs. This is the raw data behind the chart above.")

    # Build display columns, including system if available
    display_cols = ['model_clean', 'provider']
    if 'system_name' in run_comparison.columns:
        display_cols.append('system_name')
    display_cols.extend(['performance_change', 'throughput_delta_pct'])
    display_cols = [c for c in display_cols if c in run_comparison.columns]

    if display_cols:
        # Configure columns for better display with detailed help
        column_config = {
            'model_clean': st.column_config.TextColumn('Model',
                help='AI model name (e.g., resnet50, yolov5s)'),
            'provider': st.column_config.TextColumn('Provider',
                help='Execution provider: CPU (x86 baseline), DML (GPU via DirectML), or VitisAI (AMD NPU)'),
            'system_name': st.column_config.TextColumn('System',
                help='Benchmark system name'),
            'performance_change': st.column_config.TextColumn('Direction',
                help='Simple indicator: "improved" = Run2 was faster, "degraded" = Run2 was slower, "stable" = minimal change'),
            'throughput_delta_pct': st.column_config.NumberColumn('Throughput Delta %',
                format='%+.1f%%',
                help='Percentage change = (Run2 - Run1) / Run1 × 100. Positive = faster in Run2. Example: +47.7% means Run2 was 47.7% faster than Run1 for THIS specific model+provider.'),
        }

        render_dataframe(
            run_comparison[display_cols],
            column_config=column_config,
            use_container_width=True,
            hide_index=True
        )
        st.caption("💡 Click column headers to sort | **Tip:** Sort by 'Throughput Delta %' to find models with largest changes")


# ============================================================================
# TAB 6: SYSTEM COMPARISON
# ============================================================================

def render_system_comparison(data, show_chart_help=False):
    """Render system comparison tab for cross-system analysis."""
    st.header("System Comparison")
    st.markdown("""
    Compares benchmark performance **across different systems**.
    Useful for validating consistency and identifying hardware-specific behavior.
    """)

    system_summary = data.get('system_summary', pd.DataFrame())
    system_comparison = data.get('system_comparison', pd.DataFrame())
    model_summary = data.get('model_summary', pd.DataFrame())

    # Check if we have any system data
    has_system_data = (
        (not system_summary.empty) or
        (not model_summary.empty and 'system_name' in model_summary.columns)
    )

    if not has_system_data:
        st.info("""
        **No system data available.**

        To enable this feature, run the data pipeline:
        ```
        python run_pipeline.py --raw-dirs "/path/to/system1"
        ```

        For multi-system comparison:
        ```
        python run_pipeline.py --raw-dirs "/path/to/system1" "/path/to/system2"
        ```
        """)
        return

    # System summary cards
    if not system_summary.empty:
        st.subheader("System Overview")

        render_chart_help(
            "System Overview",
            """
**What this shows:** High-level summary of each benchmark system's performance.

**Metrics explained:**
- **Models Tested**: Number of unique AI models benchmarked
- **Providers Tested**: Number of execution providers (CPU, DML, VitisAI)
- **Total Configs**: Total model+provider combinations tested
- **Avg Throughput**: Average inferences per second across all tests
- **Max Throughput**: Best throughput achieved
- **Avg Latency**: Average inference time in milliseconds
            """,
            show_chart_help
        )

        for _, row in system_summary.iterrows():
            system_name = row.get('system_name', 'Unknown')
            models = row.get('models_tested', 0)
            providers = row.get('providers_tested', 0)
            configs = row.get('total_configs', 0)
            total_runs = row.get('total_runs', 0)
            avg_thr = row.get('avg_throughput_ips', 0)
            max_thr = row.get('max_throughput_ips', 0)
            avg_lat = row.get('avg_latency_ms', 0)
            min_lat = row.get('min_latency_ms', 0)
            perf_score = row.get('avg_performance_score', 0)
            eff_score = row.get('avg_efficiency_score', 0)
            stab_score = row.get('avg_stability_score', 0)

            st.markdown(f"""
            <div style="padding: 20px; border-radius: 10px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; margin-bottom: 20px;">
                <h3 style="margin: 0 0 15px 0; color: white;">{system_name}</h3>
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;">
                    <div style="background: rgba(255,255,255,0.15); padding: 10px; border-radius: 8px;">
                        <div style="font-size: 24px; font-weight: bold;">{models}</div>
                        <div style="font-size: 12px; opacity: 0.9;">Models Tested</div>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); padding: 10px; border-radius: 8px;">
                        <div style="font-size: 24px; font-weight: bold;">{providers}</div>
                        <div style="font-size: 12px; opacity: 0.9;">Providers</div>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); padding: 10px; border-radius: 8px;">
                        <div style="font-size: 24px; font-weight: bold;">{configs}</div>
                        <div style="font-size: 12px; opacity: 0.9;">Configurations</div>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); padding: 10px; border-radius: 8px;">
                        <div style="font-size: 24px; font-weight: bold;">{total_runs}</div>
                        <div style="font-size: 12px; opacity: 0.9;">Benchmark Runs</div>
                    </div>
                </div>
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-top: 15px;">
                    <div style="background: rgba(255,255,255,0.15); padding: 10px; border-radius: 8px;">
                        <div style="font-size: 20px; font-weight: bold;">{avg_thr:.1f}</div>
                        <div style="font-size: 12px; opacity: 0.9;">Avg Throughput (ips)</div>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); padding: 10px; border-radius: 8px;">
                        <div style="font-size: 20px; font-weight: bold;">{max_thr:.1f}</div>
                        <div style="font-size: 12px; opacity: 0.9;">Max Throughput (ips)</div>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); padding: 10px; border-radius: 8px;">
                        <div style="font-size: 20px; font-weight: bold;">{avg_lat:.1f}</div>
                        <div style="font-size: 12px; opacity: 0.9;">Avg Latency (ms)</div>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); padding: 10px; border-radius: 8px;">
                        <div style="font-size: 20px; font-weight: bold;">{min_lat:.2f}</div>
                        <div style="font-size: 12px; opacity: 0.9;">Min Latency (ms)</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

    # Per-provider performance breakdown (works with single or multiple systems)
    if not model_summary.empty and 'provider' in model_summary.columns:
        st.subheader("Performance by Provider")

        render_chart_help(
            "Performance by Provider",
            """
**What this shows:** Average performance metrics broken down by execution provider.

**How to interpret:**
- Compare throughput and latency across CPU, DML (GPU), and VitisAI (NPU)
- Higher throughput and lower latency indicate better performance
- Use this to identify which provider offers best performance for your workload
            """,
            show_chart_help
        )

        # Calculate per-provider stats
        provider_stats = model_summary.groupby('provider').agg({
            'throughput_mean_ips': ['mean', 'max', 'std'],
            'latency_mean_ms': ['mean', 'min'],
            'model_clean': 'nunique'
        }).reset_index()
        provider_stats.columns = ['provider', 'avg_throughput', 'max_throughput', 'throughput_std',
                                   'avg_latency', 'min_latency', 'models_count']

        # Provider cards
        cols = st.columns(len(provider_stats))
        for idx, (_, row) in enumerate(provider_stats.iterrows()):
            with cols[idx]:
                provider = row['provider']
                color = get_provider_color(provider)

                st.markdown(f"""
                <div style="background-color: {color}22; padding: 15px; border-radius: 10px; border-left: 4px solid {color};">
                    <h4 style="margin: 0; color: {color};">{provider.upper()}</h4>
                    <p style="margin: 5px 0;"><b>Models:</b> {row['models_count']}</p>
                    <p style="margin: 5px 0;"><b>Avg Throughput:</b> {row['avg_throughput']:.1f} ips</p>
                    <p style="margin: 5px 0;"><b>Max Throughput:</b> {row['max_throughput']:.1f} ips</p>
                    <p style="margin: 5px 0;"><b>Avg Latency:</b> {row['avg_latency']:.1f} ms</p>
                    <p style="margin: 5px 0;"><b>Min Latency:</b> {row['min_latency']:.2f} ms</p>
                </div>
                """, unsafe_allow_html=True)

        # Throughput by provider chart
        fig_provider = px.bar(
            provider_stats,
            x='provider',
            y='avg_throughput',
            color='provider',
            color_discrete_map=get_provider_colors(),
            title='Average Throughput by Provider',
            labels={'provider': 'Provider', 'avg_throughput': 'Avg Throughput (ips)'}
        )
        fig_provider.update_layout(showlegend=False, height=350)
        plotly_chart_dark(fig_provider, use_container_width=True)

        st.divider()

    # Top performing models table
    if not model_summary.empty:
        st.subheader("Top Performing Models")

        render_chart_help(
            "Top Performing Models",
            """
**What this shows:** The best performing model+provider combinations ranked by throughput.

**How to interpret:**
- Higher throughput = more inferences per second = better performance
- Compare providers to see which accelerator works best for each model
            """,
            show_chart_help
        )

        # Get top 10 by throughput
        top_models = model_summary.nlargest(10, 'throughput_mean_ips')

        # Include system_name if available for multi-system comparison
        display_cols = ['model_clean', 'provider']
        if 'system_name' in top_models.columns:
            display_cols.append('system_name')
        display_cols.extend(['throughput_mean_ips', 'latency_mean_ms', 'speedup_vs_cpu'])
        display_cols = [c for c in display_cols if c in top_models.columns]

        if display_cols:
            column_config = {
                'model_clean': st.column_config.TextColumn('Model', help='AI model name'),
                'provider': st.column_config.TextColumn('Provider', help='Execution provider'),
                'system_name': st.column_config.TextColumn('System', help='Benchmark system'),
                'throughput_mean_ips': st.column_config.NumberColumn('Throughput (ips)', format='%.1f',
                    help='Inferences per second (higher is better)'),
                'latency_mean_ms': st.column_config.NumberColumn('Latency (ms)', format='%.2f',
                    help='Average inference time (lower is better)'),
                'speedup_vs_cpu': st.column_config.NumberColumn('Speedup vs CPU', format='%.2fx',
                    help='Performance multiplier compared to CPU baseline'),
            }

            render_dataframe(
                top_models[display_cols],
                column_config=column_config,
                use_container_width=True,
                hide_index=True
            )

        st.divider()

    # Multi-system comparison (only if we have multiple systems)
    if not system_comparison.empty:
        st.subheader("Cross-System Comparison")
        st.markdown("**Comparing performance across multiple benchmark systems**")

        render_chart_help(
            "Cross-System Comparison",
            """
**What this shows:** Side-by-side throughput comparison for each model across systems.

**How to interpret:**
- Bars grouped by model show performance on each system
- Larger differences may indicate hardware-specific optimizations
- Consistent results validate benchmark reproducibility
            """,
            show_chart_help
        )

        # Create grouped bar chart
        fig = px.bar(
            system_comparison,
            x='model_clean',
            y='throughput_ips',
            color='system_name',
            barmode='group',
            title='Throughput by Model and System',
            labels={
                'model_clean': 'Model',
                'throughput_ips': 'Throughput (ips)',
                'system_name': 'System'
            }
        )
        fig.update_layout(
            xaxis_tickangle=-45,
            height=500
        )
        plotly_chart_dark(fig, use_container_width=True)

        # Relative performance chart (if multiple systems)
        if 'throughput_vs_baseline_pct' in system_comparison.columns:
            st.subheader("Relative Performance vs Baseline")

            render_chart_help(
                "Relative Performance",
                """
**What this shows:** Performance difference relative to the first system (baseline).

**How to interpret:**
- 0% = same as baseline system
- Positive % = faster than baseline
- Negative % = slower than baseline
                """,
                show_chart_help
            )

            # Filter to non-baseline systems
            non_baseline = system_comparison[system_comparison['throughput_vs_baseline_pct'].notna()].copy()

            if not non_baseline.empty:
                fig2 = px.bar(
                    non_baseline,
                    x='model_clean',
                    y='throughput_vs_baseline_pct',
                    color='system_name',
                    title='Throughput Difference vs Baseline System (%)',
                    labels={
                        'model_clean': 'Model',
                        'throughput_vs_baseline_pct': 'Difference (%)',
                        'system_name': 'System'
                    }
                )
                fig2.add_hline(y=0, line_dash="dash", line_color="gray")
                fig2.update_layout(
                    xaxis_tickangle=-45,
                    height=400
                )
                plotly_chart_dark(fig2, use_container_width=True)

        st.divider()

        # Detailed comparison table
        st.subheader("Detailed System Comparison Data")
        st.caption("Performance metrics for each model across all systems.")

        display_cols = ['model_clean', 'provider', 'system_name', 'throughput_ips', 'latency_mean_ms']
        display_cols = [c for c in display_cols if c in system_comparison.columns]

        if display_cols:
            column_config = {
                'model_clean': st.column_config.TextColumn('Model', help='AI model name'),
                'provider': st.column_config.TextColumn('Provider', help='Execution provider'),
                'system_name': st.column_config.TextColumn('System', help='Benchmark system'),
                'throughput_ips': st.column_config.NumberColumn('Throughput (ips)', format='%.1f',
                    help='Inferences per second'),
                'latency_mean_ms': st.column_config.NumberColumn('Latency (ms)', format='%.2f',
                    help='Average inference time'),
            }

            render_dataframe(
                system_comparison[display_cols],
                column_config=column_config,
                use_container_width=True,
                hide_index=True
            )
    elif len(system_summary) == 1:
        # Single system - show helpful message about adding more
        st.info("""
        **Single System Detected**

        To enable cross-system comparison, run the pipeline with multiple system directories:
        ```
        python run_pipeline.py --raw-dirs "/path/to/system1" "/path/to/system2"
        ```
        """)


# ============================================================================
# TAB 7: POWER & THERMAL ANALYSIS (Issue #14 - renumbered from duplicate 7)
# ============================================================================

def render_power_thermal(data, show_chart_help=False):
    """Render power and thermal analysis tab."""
    st.header("Power & Thermal Analysis")
    st.markdown("""
    Analyzes **power consumption** and **thermal behavior** during inference.
    Key for battery-powered devices and thermal-constrained deployments.
    """)

    power_efficiency = data['power_efficiency']
    thermal_profile = data['provider_thermal_profile']

    if power_efficiency.empty:
        st.warning("No power/thermal data available. Sensor data may not have been processed.")
        return

    # System selector for power analysis (if multiple systems)
    has_multiple_systems = 'system_name' in power_efficiency.columns and power_efficiency['system_name'].nunique() > 1

    if has_multiple_systems:
        systems = power_efficiency['system_name'].unique().tolist()
        selected_system = st.selectbox(
            "Select System for Analysis",
            options=["All Systems"] + systems,
            key="power_system_selector",
            help="Power/thermal data is system-specific. Select a system for detailed analysis."
        )

        if selected_system != "All Systems":
            power_efficiency = power_efficiency[power_efficiency['system_name'] == selected_system]
            if not thermal_profile.empty and 'system_name' in thermal_profile.columns:
                thermal_profile = thermal_profile[thermal_profile['system_name'] == selected_system]
            st.info(f"Showing power/thermal data for: **{selected_system}**")

    # Inferences per watt (if available)
    if 'inferences_per_watt' in power_efficiency.columns:
        st.subheader("Energy Efficiency (Inferences per Watt)")

        render_chart_help(
            "Energy Efficiency",
            """
**What this shows:** Power efficiency measured as inferences per watt (inf/W) for each model-provider combination.

**How to interpret:**
- Higher values indicate better power efficiency (more work per unit of energy)
- Useful for battery-powered or power-constrained deployments
- Compare providers to identify the most energy-efficient option per model
            """,
            show_chart_help
        )

        valid_data = power_efficiency.dropna(subset=['inferences_per_watt'])
        if not valid_data.empty:
            # Use faceting for multiple systems when showing all
            show_all_systems = has_multiple_systems and (selected_system == "All Systems" if has_multiple_systems else False)

            if show_all_systems:
                fig = px.bar(
                    valid_data,
                    x='model_clean',
                    y='inferences_per_watt',
                    color='provider',
                    facet_col='system_name',
                    barmode='group',
                    color_discrete_map=get_provider_colors(),
                    labels={'model_clean': 'Model', 'inferences_per_watt': 'Inferences per Watt (inf/W)'}
                )
                fig.update_traces(
                    hovertemplate='<b>%{x}</b><br>Efficiency: %{y:.1f} inf/W<br><i>(Higher = more efficient)</i><extra></extra>'
                )
                clean_facet_labels(fig)
            else:
                fig = px.bar(
                    valid_data,
                    x='model_clean',
                    y='inferences_per_watt',
                    color='provider',
                    barmode='group',
                    color_discrete_map=get_provider_colors(),
                    labels={'model_clean': 'Model', 'inferences_per_watt': 'Inferences per Watt (inf/W)'}
                )
                fig.update_traces(
                    hovertemplate='<b>%{x}</b><br>Provider: %{fullData.name}<br>Efficiency: %{y:.1f} inf/W<br><i>(Higher = more efficient)</i><extra></extra>'
                )

            fig.update_layout(
                xaxis_title='Model',
                yaxis_title='Inferences per Watt (inf/W)',
                xaxis_tickangle=-45,
                height=500
            )
            plotly_chart_dark(fig, use_container_width=True)
            st.caption("Higher bars = better efficiency. A model with 100 inf/W can perform 100 inferences using 1 watt of power.")
        else:
            st.info("No power consumption data available for efficiency calculations.")

    # Provider thermal profile
    if not thermal_profile.empty:
        st.divider()
        st.subheader("Provider Thermal Profile")

        render_chart_help(
            "Provider Thermal Profile",
            """
**What this shows:** Average power consumption and CPU utilization metrics per provider.

**How to interpret:**
- **Avg Power**: Average CPU package power during inference (lower = more efficient)
- **Max Power**: Peak power observed (important for thermal design)
- **CPU Usage**: Average core utilization (shows compute load distribution)
- Compare providers to understand thermal/power tradeoffs vs performance
            """,
            show_chart_help
        )

        # Check if we should group by system
        show_all_systems = has_multiple_systems and (selected_system == "All Systems" if has_multiple_systems else False)

        if show_all_systems and 'system_name' in thermal_profile.columns:
            # Show thermal profile per system
            for system_name in thermal_profile['system_name'].unique():
                st.markdown(f"#### {system_name}")
                system_thermal = thermal_profile[thermal_profile['system_name'] == system_name]
                cols = st.columns(len(system_thermal))

                for idx, (_, row) in enumerate(system_thermal.iterrows()):
                    with cols[idx]:
                        provider = row['provider']
                        color = get_provider_color(provider)

                        # Extract power and usage metrics
                        avg_power = row.get('avg_s_cpu_package_power_mean_w', None)
                        max_power = row.get('max_s_cpu_package_power_max_w', None)
                        p95_power = row.get('avg_s_cpu_package_power_p95_w', None)
                        cpu_usage = row.get('avg_s_core_usage_avg_mean_pct', None)

                        # Format values
                        avg_power_str = f"{avg_power:.1f} W" if pd.notna(avg_power) else "N/A"
                        max_power_str = f"{max_power:.1f} W" if pd.notna(max_power) else "N/A"
                        p95_power_str = f"{p95_power:.1f} W" if pd.notna(p95_power) else "N/A"
                        cpu_usage_str = f"{cpu_usage:.1f}%" if pd.notna(cpu_usage) else "N/A"

                        st.markdown(f"""
                        <div style="background-color: {color}22; padding: 15px; border-radius: 10px; border-left: 4px solid {color};">
                            <h4 style="margin: 0; color: {color};">{provider.upper()}</h4>
                            <p><b>Test Count:</b> {row.get('test_count', 'N/A')}</p>
                            <p><b>Avg Power:</b> {avg_power_str}</p>
                            <p><b>P95 Power:</b> {p95_power_str}</p>
                            <p><b>Max Power:</b> {max_power_str}</p>
                            <p><b>Avg CPU Usage:</b> {cpu_usage_str}</p>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            # Original single-row display
            cols = st.columns(len(thermal_profile))

            for idx, (_, row) in enumerate(thermal_profile.iterrows()):
                with cols[idx]:
                    provider = row['provider']
                    color = get_provider_color(provider)

                    # Extract power and usage metrics
                    avg_power = row.get('avg_s_cpu_package_power_mean_w', None)
                    max_power = row.get('max_s_cpu_package_power_max_w', None)
                    p95_power = row.get('avg_s_cpu_package_power_p95_w', None)
                    cpu_usage = row.get('avg_s_core_usage_avg_mean_pct', None)

                    # Format values
                    avg_power_str = f"{avg_power:.1f} W" if pd.notna(avg_power) else "N/A"
                    max_power_str = f"{max_power:.1f} W" if pd.notna(max_power) else "N/A"
                    p95_power_str = f"{p95_power:.1f} W" if pd.notna(p95_power) else "N/A"
                    cpu_usage_str = f"{cpu_usage:.1f}%" if pd.notna(cpu_usage) else "N/A"

                    st.markdown(f"""
                    <div style="background-color: {color}22; padding: 15px; border-radius: 10px; border-left: 4px solid {color};">
                        <h4 style="margin: 0; color: {color};">{provider.upper()}</h4>
                        <p><b>Test Count:</b> {row.get('test_count', 'N/A')}</p>
                        <p><b>Avg Power:</b> {avg_power_str}</p>
                        <p><b>P95 Power:</b> {p95_power_str}</p>
                        <p><b>Max Power:</b> {max_power_str}</p>
                        <p><b>Avg CPU Usage:</b> {cpu_usage_str}</p>
                    </div>
                    """, unsafe_allow_html=True)

    # Power vs Throughput scatter
    if 'throughput_mean_ips' in power_efficiency.columns:
        power_col = next((c for c in power_efficiency.columns if 'power' in c.lower() and 'mean' in c.lower()), None)

        if power_col:
            st.divider()
            st.subheader("Power vs Throughput")

            render_chart_help(
                "Power vs Throughput",
                """
**What this shows:** Scatter plot showing the relationship between power consumption and throughput (efficiency frontier).

**How to interpret:**
- Points in the upper-left are ideal (high throughput, low power)
- Points in the lower-right are inefficient (low throughput, high power)
- The "efficiency frontier" shows the best trade-off between power and performance
- Hover over points to see which model-provider combination they represent
                """,
                show_chart_help
            )

            valid_data = power_efficiency.dropna(subset=[power_col, 'throughput_mean_ips'])
            if not valid_data.empty:
                # Add system as symbol when showing all systems
                show_all_systems = has_multiple_systems and (selected_system == "All Systems" if has_multiple_systems else False)
                symbol_col = 'system_name' if (show_all_systems and 'system_name' in valid_data.columns) else None

                hover_data = ['model_clean']
                if symbol_col:
                    hover_data.append('system_name')

                fig = px.scatter(
                    valid_data,
                    x=power_col,
                    y='throughput_mean_ips',
                    color='provider',
                    symbol=symbol_col,
                    hover_data=hover_data,
                    color_discrete_map=get_provider_colors(),
                    title='Efficiency Frontier: Power vs Performance Trade-off'
                )

                if symbol_col:
                    fig.update_traces(
                        hovertemplate='<b>%{customdata[0]}</b><br>System: %{customdata[1]}<br>Power: %{x:.1f} W<br>Throughput: %{y:.1f} ips<extra></extra>',
                        customdata=valid_data[['model_clean', 'system_name']].values
                    )
                else:
                    fig.update_traces(
                        hovertemplate='<b>%{customdata[0]}</b><br>Provider: %{fullData.name}<br>Power: %{x:.1f} W<br>Throughput: %{y:.1f} ips<extra></extra>',
                        customdata=valid_data[['model_clean']].values
                    )

                fig.update_layout(
                    xaxis_title='Power Consumption (Watts)',
                    yaxis_title='Throughput (inferences per second)',
                    height=500
                )
                plotly_chart_dark(fig, use_container_width=True)
                st.caption("**Ideal position:** upper-left (high throughput, low power). Points further right use more power; points higher are faster.")

    # Export power/thermal data
    st.divider()
    st.subheader("Export Power & Thermal Data")
    render_export_buttons(power_efficiency, "power_thermal_data", "power_thermal")


# ============================================================================
# TAB 8: LATENCY DISTRIBUTION
# ============================================================================

def render_latency_distribution(data, show_chart_help=False):
    """Render latency distribution tab."""
    st.header("Latency Distribution")
    st.markdown("""
    Analyzes **latency consistency** - critical for real-time applications where predictable response times matter.
    Lower latency and narrower distributions indicate more reliable, real-time capable models.
    """)

    model_summary = data['model_summary']

    if model_summary.empty:
        st.warning("No data available.")
        return

    # Check if multiple systems
    has_multiple_systems = 'system_name' in model_summary.columns and model_summary['system_name'].nunique() > 1

    # Percentile comparison
    st.subheader("Latency Percentiles by Model")

    render_chart_help(
        "Latency Percentiles by Model",
        """
**What this shows:** Faceted bar charts showing P50, P95, and P99 latency for all models by provider.

**How to interpret:**
- **P50 (row 1)**: Median latency - typical performance
- **P95 (row 2)**: 95th percentile - worst case for most users
- **P99 (row 3)**: 99th percentile - tail latency for SLA-critical applications
- Lower values are better across all percentiles
        """,
        show_chart_help
    )

    latency_cols = ['latency_p50_ms', 'latency_p95_ms', 'latency_p99_ms']
    available_cols = [c for c in latency_cols if c in model_summary.columns]

    if available_cols:
        # Melt for plotting
        plot_data = []
        for _, row in model_summary.iterrows():
            for col in available_cols:
                pct = col.replace('latency_', '').replace('_ms', '').upper()
                entry = {
                    'model_clean': row['model_clean'],
                    'provider': row['provider'],
                    'percentile': pct,
                    'latency_ms': row[col]
                }
                # Include system_name if available
                if 'system_name' in row:
                    entry['system_name'] = row['system_name']
                plot_data.append(entry)

        plot_df = pd.DataFrame(plot_data)

        # Add system faceting if multiple systems
        if has_multiple_systems and 'system_name' in plot_df.columns:
            fig = px.bar(
                plot_df,
                x='model_clean',
                y='latency_ms',
                color='provider',
                facet_row='percentile',
                facet_col='system_name',
                barmode='group',
                color_discrete_map=get_provider_colors(),
                height=1000,
                labels={'model_clean': 'Model', 'latency_ms': 'Latency (ms)'}
            )
            clean_facet_labels(fig)
        else:
            fig = px.bar(
                plot_df,
                x='model_clean',
                y='latency_ms',
                color='provider',
                facet_row='percentile',
                barmode='group',
                color_discrete_map=get_provider_colors(),
                height=800,
                labels={'model_clean': 'Model', 'latency_ms': 'Latency (ms)'}
            )
            clean_facet_labels(fig)

        fig.update_layout(xaxis_tickangle=-45)
        plotly_chart_dark(fig, use_container_width=True)

    # Box plots for latency distribution
    st.divider()
    st.subheader("Latency Distribution Box Plots")

    render_chart_help(
        "Latency Distribution Box Plots",
        """
**What this shows:** Box plot visualization of latency distribution for each model-provider combination.

**How to interpret:**
- **Box**: Shows the interquartile range (P25 to P75, approximated by P50 spread)
- **Whiskers**: Extend from min to max latency observed
- **Line inside box**: Median (P50) latency
- **Marker**: Mean latency
- Narrower boxes indicate more consistent performance
- Models with whiskers extending far indicate occasional outliers
        """,
        show_chart_help
    )

    # Check if we have the required columns for box-like visualization
    box_cols = ['latency_min_ms', 'latency_p50_ms', 'latency_p95_ms', 'latency_p99_ms', 'latency_max_ms', 'latency_mean_ms']
    if all(c in model_summary.columns for c in box_cols):
        # Create box plot using min, p50, p95, p99, max as box boundaries
        fig_box = go.Figure()

        providers = model_summary['provider'].unique()
        models = sorted(model_summary['model_clean'].unique())

        for provider in providers:
            provider_data = model_summary[model_summary['provider'] == provider]
            color = get_provider_color(provider)

            for _, row in provider_data.iterrows():
                # Create a box trace for each model-provider pair
                fig_box.add_trace(go.Box(
                    name=provider.upper(),
                    x=[row['model_clean']],
                    lowerfence=[row['latency_min_ms']],
                    q1=[row['latency_p50_ms'] * 0.9],  # Approximate Q1
                    median=[row['latency_p50_ms']],
                    mean=[row['latency_mean_ms']],
                    q3=[row['latency_p95_ms']],  # Use P95 as Q3
                    upperfence=[row['latency_max_ms']],
                    marker_color=color,
                    legendgroup=provider,
                    showlegend=False,
                    boxmean=True
                ))

        # Add legend entries
        for provider in providers:
            color = get_provider_color(provider)
            fig_box.add_trace(go.Box(
                name=provider.upper(),
                x=[None],
                marker_color=color,
                legendgroup=provider,
                showlegend=True
            ))

        fig_box.update_layout(
            xaxis_title='Model',
            yaxis_title='Latency (ms)',
            xaxis_tickangle=-45,
            height=500,
            boxmode='group',
            showlegend=True
        )
        plotly_chart_dark(fig_box, use_container_width=True)
        # Issue #17: Add disclaimer about Q1 approximation
        st.caption("""
        **Note:** Box plot boundaries use P50 (median) and P95 for interquartile range.
        Q1 is approximated as 90% of P50 since exact P25 is not available in the benchmark data.
        For precise percentile values, refer to the Latency Percentiles table above.
        """)
    else:
        st.info("Box plot requires min, p50, p95, p99, and max latency columns.")

    # Tail latency ratio
    st.divider()
    st.subheader("Tail Latency Ratio (P99/P50)")

    render_chart_help(
        "Tail Latency Ratio",
        """
**What this shows:** The ratio of P99 to P50 latency, measuring latency consistency.

**How to interpret:**
- **Ratio near 1.0**: Very consistent latency (P99 close to P50)
- **Ratio > 1.5 (red line)**: Significant tail latency - occasional slow inferences
- **Ratio > 2.0**: Poor consistency - some inferences take 2x+ longer than typical
- Lower ratios indicate more predictable, consistent performance
        """,
        show_chart_help
    )
    st.markdown("*Lower is better - indicates more consistent latency*")

    if 'latency_p99_ms' in model_summary.columns and 'latency_p50_ms' in model_summary.columns:
        model_summary_copy = model_summary.copy()
        model_summary_copy['tail_ratio'] = model_summary_copy['latency_p99_ms'] / model_summary_copy['latency_p50_ms']

        fig = px.bar(
            model_summary_copy,
            x='model_clean',
            y='tail_ratio',
            color='provider',
            barmode='group',
            color_discrete_map=get_provider_colors(),
            labels={'model_clean': 'Model', 'tail_ratio': 'P99/P50 Ratio'}
        )
        fig.add_hline(y=1.5, line_dash="dash", line_color="red", annotation_text="Good threshold")
        fig.update_layout(
            xaxis_title='Model',
            yaxis_title='P99/P50 Ratio',
            xaxis_tickangle=-45,
            height=500
        )
        plotly_chart_dark(fig, use_container_width=True)

    # Real-time capability
    st.divider()
    st.subheader("Real-time Capability Assessment")

    # Always show the FPS threshold reference
    st.info("""
    **FPS Latency Budgets:** To achieve a target frame rate, each inference must complete within a time budget:
    - **30 FPS** → Max 33.33 ms per inference (standard video)
    - **60 FPS** → Max 16.67 ms per inference (gaming/high refresh)
    - **120 FPS** → Max 8.33 ms per inference (ultra-high refresh)

    We use **P99 latency** (worst-case for 99% of inferences) to ensure real-time guarantees.
    """)

    render_chart_help(
        "Real-time Capability Assessment",
        """
**What this shows:** Percentage of models that can achieve various real-time frame rate targets.

**How to interpret:**
- **30 FPS** (33.33ms budget): Standard video rate for smooth perception
- **60 FPS** (16.67ms budget): High frame rate gaming/video
- **120 FPS** (8.33ms budget): Ultra-high refresh rate displays
- Higher percentages mean more models can meet the real-time requirement
- We measure against P99 latency for worst-case real-time guarantees
        """,
        show_chart_help
    )

    fps_thresholds = {30: 33.33, 60: 16.67, 120: 8.33}

    if 'latency_p99_ms' in model_summary.columns:
        model_summary_copy = model_summary.copy()

        for fps, threshold in fps_thresholds.items():
            model_summary_copy[f'meets_{fps}fps'] = model_summary_copy['latency_p99_ms'] <= threshold

        # Summary by provider (and system if multiple)
        summary_data = []

        if has_multiple_systems:
            # Include system breakdown
            for system in model_summary_copy['system_name'].unique():
                sys_data = model_summary_copy[model_summary_copy['system_name'] == system]
                for provider in sys_data['provider'].unique():
                    prov_data = sys_data[sys_data['provider'] == provider]
                    for fps in [30, 60, 120]:
                        count = prov_data[f'meets_{fps}fps'].sum()
                        total = len(prov_data)
                        summary_data.append({
                            'system_name': system,
                            'provider': provider,
                            'fps_target': f'{fps} FPS',
                            'models_capable': count,
                            'percentage': (count / total * 100) if total > 0 else 0
                        })

            summary_df = pd.DataFrame(summary_data)

            fig = px.bar(
                summary_df,
                x='fps_target',
                y='percentage',
                color='provider',
                facet_col='system_name',
                barmode='group',
                color_discrete_map=get_provider_colors(),
                title='Models Meeting Real-time Thresholds (%)',
                labels={'fps_target': 'Target Frame Rate', 'percentage': 'Models Capable (%)'}
            )
            clean_facet_labels(fig)
        else:
            # Original single-system view
            for provider in model_summary_copy['provider'].unique():
                prov_data = model_summary_copy[model_summary_copy['provider'] == provider]
                for fps in [30, 60, 120]:
                    count = prov_data[f'meets_{fps}fps'].sum()
                    total = len(prov_data)
                    summary_data.append({
                        'provider': provider,
                        'fps_target': f'{fps} FPS',
                        'models_capable': count,
                        'percentage': (count / total * 100) if total > 0 else 0
                    })

            summary_df = pd.DataFrame(summary_data)

            fig = px.bar(
                summary_df,
                x='fps_target',
                y='percentage',
                color='provider',
                barmode='group',
                color_discrete_map=get_provider_colors(),
                title='Models Meeting Real-time Thresholds (%)',
                labels={'fps_target': 'Target Frame Rate', 'percentage': 'Models Capable (%)'}
            )

        fig.update_layout(
            xaxis_title='Target Frame Rate',
            yaxis_title='Models Capable (%)',
            height=400
        )
        plotly_chart_dark(fig, use_container_width=True)

    # Export latency data
    st.divider()
    st.subheader("Export Latency Data")
    render_export_buttons(model_summary, "latency_distribution", "latency")


# ============================================================================
# TAB 9: USE CASE MAPPING
# ============================================================================

def _get_models_for_use_case(all_models: list, patterns: list) -> list:
    """
    Dynamically find models matching use case patterns (Issue #16).

    Args:
        all_models: List of all available model names
        patterns: List of pattern strings to match (case-insensitive)

    Returns:
        List of models matching any of the patterns
    """
    matching = []
    for model in all_models:
        model_lower = model.lower()
        if any(p.lower() in model_lower for p in patterns):
            matching.append(model)
    return matching


def render_use_case_mapping(data, show_chart_help=False):
    """Render use case mapping tab."""
    st.header("Use Case Mapping & FAE Guidance")
    st.markdown("""
    **Practical recommendations** for selecting the right model+provider combination based on your application requirements.
    Each use case has specific latency constraints and recommended models.
    """)

    model_summary = data['model_summary']

    if model_summary.empty:
        st.warning("No data available.")
        return

    # Get all available models for dynamic matching
    all_models = model_summary['model_clean'].unique().tolist()

    # Use case definitions with patterns for dynamic matching (Issue #16)
    # Using patterns instead of exact names allows matching variations like
    # 'squeezenet' and 'SqueezeNet_int8' from the same pattern
    use_cases = {
        'Real-time Video (30+ FPS)': {
            'description': 'Video analytics, live streaming, security cameras',
            'latency_req': 33.33,
            'patterns': ['efficientnet', 'mobilenet', 'mnasnet', 'squeezenet']
        },
        'Image Classification': {
            'description': 'Product recognition, document classification',
            'latency_req': 100,
            'patterns': ['resnet', 'inception', 'efficientnet', 'vovnet']
        },
        'Object Detection': {
            'description': 'Industrial inspection, autonomous systems',
            'latency_req': 50,
            'patterns': ['yolo', 'retinaface']
        },
        'Pose Estimation': {
            'description': 'Fitness tracking, gesture recognition',
            'latency_req': 33.33,
            'patterns': ['movenet', 'hrnet', 'highresolution']
        },
        'Super Resolution': {
            'description': 'Image enhancement, upscaling',
            'latency_req': 500,
            'patterns': ['sesr']
        },
        'Semantic Segmentation': {
            'description': 'Scene understanding, autonomous driving',
            'latency_req': 100,
            'patterns': ['fpn', 'semantic', 'pan']
        }
    }

    st.subheader("Use Case Recommendations")

    for use_case, details in use_cases.items():
        with st.expander(f"📋 {use_case}", expanded=False):
            st.markdown(f"**Description:** {details['description']}")
            st.markdown(f"**Latency Requirement:** < {details['latency_req']} ms")

            # Dynamically find models matching use case patterns (Issue #16)
            relevant_models = _get_models_for_use_case(all_models, details['patterns'])

            if relevant_models:
                relevant_data = model_summary[model_summary['model_clean'].isin(relevant_models)]

                # Best provider by throughput
                provider_perf = relevant_data.groupby('provider')['throughput_mean_ips'].mean()

                if not provider_perf.empty:
                    best_provider = provider_perf.idxmax()
                    st.markdown(f"**Recommended Provider:** {best_provider.upper()}")

                    # Configure columns for better display
                    use_case_col_config = {
                        'model_clean': st.column_config.TextColumn('Model', help='Model name'),
                        'provider': st.column_config.TextColumn('Provider', help='Execution provider'),
                        'throughput_mean_ips': st.column_config.NumberColumn('Throughput (ips)', format='%.1f', help='Inferences per second'),
                        'latency_mean_ms': st.column_config.NumberColumn('Latency (ms)', format='%.2f', help='Mean latency in milliseconds'),
                    }

                    # Show table with sorting enabled
                    render_dataframe(
                        relevant_data[['model_clean', 'provider', 'throughput_mean_ips', 'latency_mean_ms']],
                        column_config=use_case_col_config,
                        use_container_width=True,
                        hide_index=True
                    )
            else:
                st.info("No matching models in benchmark data.")

    # Export section
    st.divider()
    st.subheader("Export Data")
    st.caption("Download raw data for further analysis in Excel, Python, or other tools.")

    col1, col2 = st.columns(2)

    with col1:
        csv = model_summary.to_csv(index=False)
        st.download_button(
            label="📥 Download Model Summary CSV",
            data=csv,
            file_name="model_summary.csv",
            mime="text/csv",
            help="Contains: model name, provider, throughput, latency percentiles, speedup, and performance scores for all configurations"
        )
        st.caption("All model+provider performance metrics")

    with col2:
        reliability = data['reliability_analysis']
        if not reliability.empty:
            csv = reliability.to_csv(index=False)
            st.download_button(
                label="📥 Download Reliability Analysis CSV",
                data=csv,
                file_name="reliability_analysis.csv",
                mime="text/csv",
                help="Contains: model name, provider, total runs, successful runs, failed runs, success rate, and error types"
            )
            st.caption("Success rates and error tracking")


# ============================================================================
# FILTER FUNCTIONS
# ============================================================================

def apply_filters(data: dict, selected_models: list, selected_providers: list, selected_runs: list, selected_systems: list = None) -> dict:
    """Apply sidebar filters to all dataframes in the data dict.

    When filtering by runs, uses model_summary_by_run which has per-run data.
    When all runs selected, uses aggregated model_summary.
    """
    filtered = {}

    filterable_tables = [
        'model_summary', 'model_summary_by_run', 'provider_comparison', 'reliability_analysis',
        'run_comparison', 'error_root_causes', 'run_status_summary',
        'power_efficiency', 'provider_thermal_profile',
        'system_summary', 'system_comparison'
    ]

    # Get all available runs from model_summary_by_run
    model_summary_by_run = data.get('model_summary_by_run', pd.DataFrame())
    all_runs = []
    if not model_summary_by_run.empty and 'run_name' in model_summary_by_run.columns:
        all_runs = model_summary_by_run['run_name'].unique().tolist()

    # Determine if we should use per-run data
    # Use per-run data when only a subset of runs is selected
    use_per_run_data = (
        len(selected_runs) > 0 and
        len(selected_runs) < len(all_runs) and
        not model_summary_by_run.empty
    )

    for key, df in data.items():
        if df.empty:
            filtered[key] = df
            continue

        result = df.copy()

        # For model_summary, substitute with per-run data if filtering by runs
        if key == 'model_summary' and use_per_run_data:
            # Use model_summary_by_run filtered by selected runs
            result = model_summary_by_run[model_summary_by_run['run_name'].isin(selected_runs)].copy()
        elif key == 'model_summary_by_run':
            # Always filter model_summary_by_run by selected runs
            if selected_runs and 'run_name' in result.columns:
                result = result[result['run_name'].isin(selected_runs)]
        else:
            # Apply run filter to other tables that have run_name
            if selected_runs and 'run_name' in result.columns and key in filterable_tables:
                result = result[result['run_name'].isin(selected_runs)]

        # Apply model filter
        if selected_models and key in filterable_tables:
            if 'model_clean' in result.columns:
                result = result[result['model_clean'].isin(selected_models)]
            elif 'model' in result.columns:
                result = result[result['model'].isin(selected_models)]

        # Apply provider filter
        if selected_providers and key in filterable_tables:
            if 'provider' in result.columns:
                result = result[result['provider'].isin(selected_providers)]

        # Apply system filter
        if selected_systems and key in filterable_tables:
            if 'system_name' in result.columns:
                result = result[result['system_name'].isin(selected_systems)]
            elif 'system_id' in result.columns:
                result = result[result['system_id'].isin(selected_systems)]

        filtered[key] = result

    return filtered


# ============================================================================
# CHATBOT FUNCTIONS
# ============================================================================

# Chatbot API URL - configurable via environment variable (Issue #18)
CHATBOT_API_URL = os.getenv("CHATBOT_API_URL", "http://localhost:8000")


def check_chatbot_api():
    """Check if chatbot API is available."""
    try:
        with httpx.Client(timeout=2.0) as client:
            response = client.get(f"{CHATBOT_API_URL}/health")
            return response.status_code == 200
    except:
        return False


def call_chatbot_api(message: str, dashboard_context: dict = None) -> dict:
    """Call the FastAPI chatbot backend."""
    api_url = f"{CHATBOT_API_URL}/chat"

    # Prepare conversation history
    conversation_history = []
    for msg in st.session_state.get('chat_messages', [])[:-1]:
        conversation_history.append({
            "role": msg["role"],
            "content": msg["content"]
        })

    payload = {
        "message": message,
        "session_id": st.session_state.get('chat_session_id', str(uuid.uuid4())),
        "conversation_history": conversation_history,
        "dashboard_context": dashboard_context or {}
    }

    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(api_url, json=payload)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API returned status {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}


def render_floating_chatbot(dashboard_context: dict = None):
    """Render floating chatbot interface."""
    from datetime import datetime

    # Initialize session state
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    if 'chat_session_id' not in st.session_state:
        st.session_state.chat_session_id = str(uuid.uuid4())
    if 'chatbot_open' not in st.session_state:
        st.session_state.chatbot_open = False

    # Toggle button
    if not st.session_state.chatbot_open:
        col1, col2, col3 = st.columns([5, 1, 1])
        with col3:
            if st.button("💬 AI Assistant", key="chatbot_open_btn"):
                st.session_state.chatbot_open = True
                st.rerun()
        return

    # Chatbot interface
    with st.container():
        # Header
        col1, col2 = st.columns([5, 1])
        with col1:
            st.markdown("### 🤖 AI Assistant")
        with col2:
            if st.button("✕ Close", key="chatbot_close_btn"):
                st.session_state.chatbot_open = False
                st.rerun()

        # Check API
        api_available = check_chatbot_api()

        if not api_available:
            st.warning("""
            **AI Assistant Offline** - To enable:
            1. Set `ANTHROPIC_API_KEY` in `.env`
            2. The chatbot API should be running on port 8000

            The chatbot was started automatically by the launcher. Check if it's running.
            """)
            return

        st.success("AI Assistant Online")

        # Suggestions
        if not st.session_state.chat_messages:
            st.markdown("**Try asking:**")
            suggestions = [
                "Which provider has the best throughput?",
                "What models work best on VitisAI?",
                "Compare DML vs CPU performance",
                "Which models have the lowest latency?"
            ]

            cols = st.columns(2)
            for i, suggestion in enumerate(suggestions):
                with cols[i % 2]:
                    if st.button(suggestion, key=f"suggestion_{i}"):
                        st.session_state.chat_messages.append({
                            "role": "user",
                            "content": suggestion,
                            "timestamp": datetime.now().isoformat()
                        })
                        with st.spinner("Thinking..."):
                            response = call_chatbot_api(suggestion, dashboard_context)
                        if response and "error" not in response:
                            assistant_msg = {
                                "role": "assistant",
                                "content": response.get("response", "Sorry, I couldn't process that."),
                                "timestamp": datetime.now().isoformat()
                            }
                            # Store executed code and result
                            if response.get("executed_code"):
                                assistant_msg["executed_code"] = response["executed_code"]
                            if response.get("code_result"):
                                assistant_msg["code_result"] = response["code_result"]
                            st.session_state.chat_messages.append(assistant_msg)
                        st.rerun()

        # Display chat history
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                # For assistant messages with code, strip the code block from display
                content = msg["content"]
                if msg["role"] == "assistant" and msg.get("executed_code"):
                    # Remove code blocks from the response text (it's shown in expander)
                    import re
                    content = re.sub(r'```python\n.*?```', '', content, flags=re.DOTALL).strip()
                    content = re.sub(r'```\n.*?```', '', content, flags=re.DOTALL).strip()

                if content:
                    st.markdown(content)

                # Show code result (collapsed by default)
                if msg.get("code_result"):
                    result = msg["code_result"]
                    if result.get("success"):
                        with st.expander("📊 Query Result"):
                            preview = result.get("preview", "")
                            if preview:
                                st.text(preview)
                            else:
                                st.info("Query executed successfully (no output)")
                    else:
                        with st.expander("⚠️ Execution Error", expanded=True):
                            st.error(result.get("error", "Unknown error"))

                # Show executed code (collapsed by default)
                if msg.get("executed_code"):
                    with st.expander("📝 View Code"):
                        st.code(msg["executed_code"], language="python")

        # Chat input
        user_input = st.chat_input("Ask about your benchmark data...", key="chatbot_input")

        if user_input:
            st.session_state.chat_messages.append({
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now().isoformat()
            })

            with st.spinner("Thinking..."):
                response = call_chatbot_api(user_input, dashboard_context)

            if response and "error" not in response:
                assistant_msg = {
                    "role": "assistant",
                    "content": response.get("response", "Sorry, I couldn't process that."),
                    "timestamp": datetime.now().isoformat()
                }
                # Store executed code and result
                if response.get("executed_code"):
                    assistant_msg["executed_code"] = response["executed_code"]
                if response.get("code_result"):
                    assistant_msg["code_result"] = response["code_result"]
                st.session_state.chat_messages.append(assistant_msg)
            else:
                error_msg = response.get("error", "Connection failed") if response else "Connection failed"
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": f"Sorry, there was an error: {error_msg}",
                    "timestamp": datetime.now().isoformat()
                })
            st.rerun()

        # Clear chat button
        if st.session_state.chat_messages:
            if st.button("🗑️ Clear Chat", key="clear_chat_btn"):
                st.session_state.chat_messages = []
                st.session_state.chat_session_id = str(uuid.uuid4())
                st.rerun()


# ============================================================================
# SIDEBAR
# ============================================================================

def get_url_params():
    """Get filter parameters from URL query string."""
    params = st.query_params
    return {
        'runs': params.get_all('runs') if 'runs' in params else None,
        'models': params.get_all('models') if 'models' in params else None,
        'providers': params.get_all('providers') if 'providers' in params else None,
    }


def set_url_params(runs: list, models: list, providers: list, all_runs: list, all_models: list, all_providers: list):
    """Set URL query params for shareable links (only if not default)."""
    params = {}
    # Only set params if they differ from defaults (all selected)
    if set(runs) != set(all_runs):
        params['runs'] = runs
    if set(models) != set(all_models):
        params['models'] = models
    if set(providers) != set(all_providers):
        params['providers'] = providers
    st.query_params.update(params)


def render_sidebar(data_raw: dict) -> tuple:
    """Render sidebar with filters. Returns (filtered_data, selected_models, selected_providers, selected_runs, selected_systems, show_help, dark_mode)."""
    st.sidebar.title("🚀 Dell Pro 16")
    st.sidebar.markdown("RyzenAI Benchmark Analysis")

    # Data freshness indicator
    import datetime
    model_summary_raw = data_raw.get('model_summary', pd.DataFrame())
    if not model_summary_raw.empty and 'timestamp' in model_summary_raw.columns:
        try:
            latest_ts = pd.to_datetime(model_summary_raw['timestamp']).max()
            age = datetime.datetime.now() - latest_ts.to_pydatetime().replace(tzinfo=None)
            if age.days > 0:
                freshness = f"📅 Data from {age.days}d ago"
                freshness_color = "orange" if age.days < 7 else "red"
            elif age.seconds // 3600 > 0:
                freshness = f"🕐 Data from {age.seconds // 3600}h ago"
                freshness_color = "green"
            else:
                freshness = "✅ Data is fresh"
                freshness_color = "green"
            st.sidebar.markdown(f"<small style='color: {freshness_color};'>{freshness}</small>", unsafe_allow_html=True)
        except Exception:
            pass

    # Action buttons row
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🔄 Refresh", help="Clear cache and reload data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    with col2:
        reset_clicked = st.button("↩️ Reset", help="Reset all filters to default", use_container_width=True)

    st.sidebar.divider()

    # Get reference dataframes for filter options
    model_summary_by_run = data_raw.get('model_summary_by_run', pd.DataFrame())
    model_summary_raw = data_raw.get('model_summary', pd.DataFrame())

    if model_summary_raw.empty and model_summary_by_run.empty:
        st.sidebar.error("No data loaded")
        return data_raw, [], [], [], [], False, False

    # Display Options
    st.sidebar.subheader("📊 Display Options")
    show_chart_help = st.sidebar.checkbox("Show chart explanations", value=True)

    # Dark mode toggle
    dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=False, help="Toggle dark/light theme")

    # Apply theme
    apply_theme(dark_mode)

    st.sidebar.divider()

    # Filters - Hierarchy: System → Model → Provider → Runs
    st.sidebar.subheader("🔍 Filters")

    # Get URL params for initial state
    url_params = get_url_params()

    # Helper function for inline All/None toggle
    def render_filter_with_toggle(label: str, options: list, default: list, key: str, help_text: str = ""):
        """Render a multiselect with inline All/None toggle links."""
        if not options:
            return []

        # Session state key for filter override
        state_key = f'_filter_{key}'

        # Use session state override if exists (set by previous button click)
        if state_key in st.session_state:
            current_default = st.session_state.pop(state_key)
        else:
            current_default = default

        # Render label
        st.sidebar.markdown(f"**{label}**")

        # Compact All/None buttons below label
        btn_col1, btn_col2, btn_spacer = st.sidebar.columns([1, 1, 1])
        with btn_col1:
            if st.button("Select All", key=f'{key}_select_all', type="tertiary",
                        use_container_width=True, help=f"Select all {label.lower()}"):
                st.session_state[state_key] = options.copy()
                st.rerun()
        with btn_col2:
            if st.button("Clear", key=f'{key}_select_none', type="tertiary",
                        use_container_width=True, help=f"Clear all {label.lower()}"):
                st.session_state[state_key] = []
                st.rerun()

        # Multiselect
        selected = st.sidebar.multiselect(
            label,
            options=options,
            default=current_default,
            help=help_text,
            label_visibility="collapsed"
        )
        return selected

    # -------------------------------------------------------------------------
    # 1. SYSTEM FILTER (Top of hierarchy)
    # -------------------------------------------------------------------------
    all_systems = []
    if not model_summary_by_run.empty and 'system_name' in model_summary_by_run.columns:
        all_systems = sorted(model_summary_by_run['system_name'].unique().tolist())

    if reset_clicked:
        default_systems = all_systems
    else:
        default_systems = all_systems

    if all_systems:
        selected_systems = render_filter_with_toggle(
            "Systems",
            all_systems,
            default_systems,
            "systems",
            "Filter by benchmark system. Multiple systems enable cross-system comparison."
        )
    else:
        selected_systems = []

    # Filter data based on selected systems for cascading filters
    if selected_systems and not model_summary_by_run.empty and 'system_name' in model_summary_by_run.columns:
        filtered_by_system = model_summary_by_run[model_summary_by_run['system_name'].isin(selected_systems)]
    elif not model_summary_by_run.empty:
        filtered_by_system = model_summary_by_run
    else:
        filtered_by_system = model_summary_raw

    # -------------------------------------------------------------------------
    # 2. MODEL FILTER (with search)
    # -------------------------------------------------------------------------
    all_models = sorted(filtered_by_system['model_clean'].unique().tolist()) if 'model_clean' in filtered_by_system.columns else []

    # Model search filter
    model_search = st.sidebar.text_input(
        "🔍 Search Models",
        placeholder="Type to filter...",
        key="model_search",
        help="Filter the model list by name (e.g., 'resnet', 'yolo')"
    )

    # Filter models based on search
    if model_search:
        filtered_models = [m for m in all_models if model_search.lower() in m.lower()]
    else:
        filtered_models = all_models

    if reset_clicked:
        default_models = filtered_models
    elif url_params['models']:
        default_models = [m for m in url_params['models'] if m in filtered_models]
    else:
        default_models = filtered_models

    if filtered_models:
        selected_models = render_filter_with_toggle(
            "Models",
            filtered_models,
            default_models,
            "models",
            f"Filter by AI model ({len(filtered_models)} available)"
        )
    else:
        if model_search:
            st.sidebar.warning(f"No models match '{model_search}'")
        selected_models = []

    # Filter by selected models for cascading
    if selected_models and 'model_clean' in filtered_by_system.columns:
        filtered_by_model = filtered_by_system[filtered_by_system['model_clean'].isin(selected_models)]
    else:
        filtered_by_model = filtered_by_system

    # -------------------------------------------------------------------------
    # 3. PROVIDER FILTER
    # -------------------------------------------------------------------------
    all_providers = sorted(filtered_by_model['provider'].unique().tolist()) if 'provider' in filtered_by_model.columns else []

    if reset_clicked:
        default_providers = all_providers
    elif url_params['providers']:
        default_providers = [p for p in url_params['providers'] if p in all_providers]
    else:
        default_providers = all_providers

    if all_providers:
        selected_providers = render_filter_with_toggle(
            "Providers",
            all_providers,
            default_providers,
            "providers",
            "Filter by execution provider (CPU, DML, VitisAI)"
        )
    else:
        selected_providers = []

    # Filter by selected providers for cascading
    if selected_providers and 'provider' in filtered_by_model.columns:
        filtered_by_provider = filtered_by_model[filtered_by_model['provider'].isin(selected_providers)]
    else:
        filtered_by_provider = filtered_by_model

    # -------------------------------------------------------------------------
    # 4. RUN FILTER (Cascades from System only, not Model/Provider)
    # -------------------------------------------------------------------------
    # Use filtered_by_system (not filtered_by_provider) so runs aren't hidden
    # when models from that run aren't selected. Different runs may have different models.
    run_to_display = {}
    display_to_run = {}
    if 'run_name' in filtered_by_system.columns:
        # Get unique run + system combinations
        if 'system_name' in filtered_by_system.columns:
            run_system_pairs = filtered_by_system[['run_name', 'system_name']].drop_duplicates()
            for _, row in run_system_pairs.iterrows():
                run_name = row['run_name']
                system_name = row['system_name']
                # Extract short system identifier (e.g., "System 1" from "Dell Pro16 - System 1")
                system_short = system_name.split(' - ')[-1] if ' - ' in system_name else system_name
                display_name = f"{run_name} ({system_short})"
                run_to_display[(run_name, system_name)] = display_name
                display_to_run[display_name] = (run_name, system_name)
        else:
            # No system info, use run names directly
            for run_name in filtered_by_system['run_name'].unique():
                run_to_display[(run_name, None)] = run_name
                display_to_run[run_name] = (run_name, None)

    all_run_displays = sorted(display_to_run.keys())

    if reset_clicked:
        default_run_displays = all_run_displays
    elif url_params['runs']:
        # Map URL run names to display names
        default_run_displays = [d for d in all_run_displays if display_to_run.get(d, (None,))[0] in url_params['runs']]
    else:
        default_run_displays = all_run_displays

    if all_run_displays:
        selected_run_displays = render_filter_with_toggle(
            "Runs",
            all_run_displays,
            default_run_displays,
            "runs",
            "Filter by benchmark run. Select a single run to view per-run metrics."
        )
        # Map display names back to run names for filtering
        selected_runs = [display_to_run[d][0] for d in selected_run_displays if d in display_to_run]
        # Store the full mapping in session state for other parts of the app
        st.session_state['run_display_mapping'] = display_to_run
    else:
        selected_runs = []
        selected_run_displays = []

    # Keep track of all runs (raw names) for URL params
    all_runs = sorted(filtered_by_system['run_name'].unique().tolist()) if 'run_name' in filtered_by_system.columns else []

    # Update URL params for shareable links
    set_url_params(selected_runs, selected_models, selected_providers, all_runs, all_models, all_providers)

    # Apply filters
    filtered_data = apply_filters(data_raw, selected_models, selected_providers, selected_runs, selected_systems)

    st.sidebar.divider()

    # Filter summary
    run_indicator = ""
    if len(selected_run_displays) == 1:
        run_indicator = f" ({selected_run_displays[0]})"
    elif len(selected_runs) < len(all_runs):
        run_indicator = f" ({len(selected_runs)} runs)"

    st.sidebar.caption(f"Showing: {len(selected_models)} models, {len(selected_providers)} providers{run_indicator}")

    st.sidebar.divider()

    # Quick stats from filtered data
    model_summary = filtered_data.get('model_summary', pd.DataFrame())
    if not model_summary.empty:
        st.sidebar.markdown("**Filtered Stats:**")
        st.sidebar.markdown(f"- Configurations: {len(model_summary)}")
        st.sidebar.markdown(f"- Models: {model_summary['model_clean'].nunique()}")
        st.sidebar.markdown(f"- Providers: {model_summary['provider'].nunique()}")
        if 'run_name' in model_summary.columns:
            st.sidebar.markdown(f"- Runs: {model_summary['run_name'].nunique()}")

    st.sidebar.divider()

    # AI Assistant button in sidebar (more discoverable)
    if st.sidebar.button("🤖 AI Assistant", use_container_width=True, help="Open AI-powered benchmark assistant"):
        st.session_state.chatbot_open = True
        st.rerun()

    st.sidebar.divider()

    # Info
    st.sidebar.markdown("**Execution Providers:**")
    st.sidebar.markdown("""
    - **CPU** - AMD Ryzen x86 processor (baseline)
    - **DML** - GPU acceleration via DirectML
    - **VitisAI** - AMD NPU (AI Engine) acceleration

    **Key Metrics:**
    - *Throughput (ips)* - Inferences per second (higher = better)
    - *Latency (ms)* - Time per inference (lower = better)
    - *Speedup* - Performance vs CPU (e.g., 5x = 5× faster)

    **Tip:** Select a single run to see per-run metrics.
    """)

    return filtered_data, selected_models, selected_providers, selected_runs, selected_systems, show_chart_help, dark_mode


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main application entry point."""
    # Check password before showing dashboard
    if not check_password():
        st.stop()

    # Load raw data with loading spinner
    with st.spinner("Loading benchmark data..."):
        data_raw = load_gold_data()

    # Render sidebar with filters and get filtered data
    data, selected_models, selected_providers, selected_runs, selected_systems, show_help, dark_mode = render_sidebar(data_raw)

    # Create tabs
    tabs = st.tabs([
        "📊 Executive Summary",
        "🔄 Provider Comparison",
        "📈 Model Performance",
        "✅ Reliability",
        "🔀 Run Comparison",
        "🏢 System Comparison",
        "⚡ Power & Thermal",
        "⏱️ Latency Distribution",
        "🎯 Use Case Mapping"
    ])

    with tabs[0]:
        with st.spinner("Loading Executive Summary..."):
            render_executive_summary(data, show_help)

    with tabs[1]:
        with st.spinner("Loading Provider Comparison..."):
            render_provider_comparison(data, show_help)

    with tabs[2]:
        with st.spinner("Loading Model Performance..."):
            render_model_performance(data, show_help)

    with tabs[3]:
        with st.spinner("Loading Reliability Analysis..."):
            render_reliability_analysis(data, show_help)

    with tabs[4]:
        with st.spinner("Loading Run Comparison..."):
            render_run_comparison(data, show_help)

    with tabs[5]:
        with st.spinner("Loading System Comparison..."):
            render_system_comparison(data, show_help)

    with tabs[6]:
        with st.spinner("Loading Power & Thermal Analysis..."):
            render_power_thermal(data, show_help)

    with tabs[7]:
        with st.spinner("Loading Latency Distribution..."):
            render_latency_distribution(data, show_help)

    with tabs[8]:
        with st.spinner("Loading Use Case Mapping..."):
            render_use_case_mapping(data, show_help)

    # Build dashboard context for chatbot
    dashboard_context = {
        "selected_models": selected_models,
        "selected_providers": selected_providers,
        "selected_runs": selected_runs,
        "selected_systems": selected_systems,
        "total_configurations": len(data.get('model_summary', [])),
    }

    # Floating chatbot (appears at bottom of all tabs)
    st.divider()
    render_floating_chatbot(dashboard_context)


if __name__ == '__main__':
    main()
