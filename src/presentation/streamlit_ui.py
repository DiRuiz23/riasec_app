"""
app.py
Interfaz Streamlit del Perfil de Personalidad Vocacional RIASEC.
Dashboard profesional estilo CoreUI — Light Mode.

Pestañas: Dashboard | Carga y visualización | Estadística descriptiva |
Entrenamiento | Resultados | Comparativa entre clústeres | Metadatos del modelo | Descargas.

Ejecutar con: streamlit run app.py
"""

import json
import io
import datetime
import unicodedata
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.infrastructure.database.db import (
    init_db, get_session, Usuario, Pregunta, OpcionPregunta, Respuesta,
    VectorRiasec, ModeloEntrenado, ResultadoClustering
)
from src.application.cuestionario_service import PREGUNTAS, DIMENSIONES, NOMBRES_DIMENSION, agregar_vector
from src.application.estadistica_service import media, desviacion_estandar, moda, mediana, distribucion_por_categoria, resumen_dimensiones
from src.application.entrenamiento_service import entrenar_modelo, cargar_modelo_activo, obtener_matriz_entrenamiento, proyeccion_pca_2d, etiquetar_clusters, obtener_historial_modelos, activar_modelo_historico

# Importar nuestro generador PDF
from src.infrastructure.services.pdf_service import crear_reporte_pdf

# --------------------------------------------------------------------------
# Configuración de página
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="RIASEC · Dashboard Vocacional",
    layout="wide",
    initial_sidebar_state="expanded",
)
init_db()

# --------------------------------------------------------------------------
# CSS Global — Estilo CoreUI Light
# --------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css');

/* ── Reset y base ── */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* ── Animación de entrada ── */
@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Fondo principal con gradiente sutil ── */
.main .block-container {
    background: linear-gradient(135deg, #EEF2FF 0%, #F9FAFB 50%, #F0F9FF 100%);
    padding: 1.5rem 2rem 2rem 2rem;
    max-width: 1400px;
}
.stApp {
    background: linear-gradient(135deg, #EEF2FF 0%, #F9FAFB 50%, #F0F9FF 100%);
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B1629 0%, #1B2A4A 55%, #243452 100%) !important;
    border-right: none;
    box-shadow: 4px 0 24px rgba(0,0,0,0.2);
}
[data-testid="stSidebar"] * {
    color: #C8D3E8 !important;
}
[data-testid="stSidebar"] .stRadio label {
    color: #A8B8D8 !important;
    font-size: 0.875rem !important;
    font-weight: 500;
    padding: 0.45rem 0.75rem 0.45rem 1rem;
    cursor: pointer;
    transition: all 0.2s ease;
    border-radius: 8px;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    border-left: 2px solid transparent;
    margin-bottom: 0.1rem;
}
[data-testid="stSidebar"] .stRadio label::before {
    content: "";
    display: inline-block;
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: #4A6B9A;
    flex-shrink: 0;
    transition: all 0.2s ease;
}
[data-testid="stSidebar"] .stRadio label:hover {
    color: #FFFFFF !important;
    background: rgba(255,255,255,0.07) !important;
    border-left-color: #3B82F6;
}
[data-testid="stSidebar"] .stRadio label:hover::before {
    background: #3B82F6;
    box-shadow: 0 0 6px rgba(59,130,246,0.5);
}
[data-testid="stSidebar"] .stRadio [aria-checked="true"] + label,
[data-testid="stSidebar"] .stRadio label[data-selected="true"] {
    color: #FFFFFF !important;
    background: rgba(59,130,246,0.12) !important;
    border-left-color: #3B82F6 !important;
    font-weight: 600;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.08) !important;
    margin: 0.75rem 0;
}

/* ── Logo sidebar ── */
.sidebar-logo {
    text-align: center;
    padding: 1.75rem 1rem 1.25rem 1rem;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 1rem;
}
.sidebar-logo h1 {
    color: #FFFFFF !important;
    font-size: 1.9rem !important;
    font-weight: 800 !important;
    letter-spacing: 5px;
    margin: 0;
}
.sidebar-logo .sidebar-subtitle {
    color: #7B97C7 !important;
    font-size: 0.68rem !important;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin: 0.3rem 0 0 0;
    display: block;
}
.sidebar-logo .sidebar-tagline {
    color: #4A6B9A !important;
    font-size: 0.62rem !important;
    margin: 0.5rem 0 0 0;
    display: block;
    font-style: italic;
}

/* ── KPI Cards ── */
.kpi-card {
    background: #FFFFFF;
    border-radius: 16px;
    padding: 1.5rem 1.6rem;
    box-shadow: 0 4px 20px rgba(15, 29, 53, 0.07);
    border: 1px solid #EEF2FF;
    border-left: 4px solid #3B82F6;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    height: 100%;
    animation: fadeSlideIn 0.4s ease;
}
.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 30px rgba(15, 29, 53, 0.13);
}
.kpi-card.green  { border-left-color: #10B981; }
.kpi-card.orange { border-left-color: #F59E0B; }
.kpi-card.violet { border-left-color: #8B5CF6; }
.kpi-card.blue   { border-left-color: #3B82F6; }
.kpi-card.red    { border-left-color: #EF4444; }
.kpi-icon {
    width: 2.2rem;
    height: 2.2rem;
    margin-bottom: 0.55rem;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 10px;
    background: rgba(59,130,246,0.08);
}
.kpi-icon svg {
    width: 1.15rem;
    height: 1.15rem;
    stroke: #3B82F6;
    stroke-width: 2;
    fill: none;
    stroke-linecap: round;
    stroke-linejoin: round;
}
.kpi-card.green  .kpi-icon { background: rgba(16,185,129,0.08); }
.kpi-card.green  .kpi-icon svg { stroke: #10B981; }
.kpi-card.orange .kpi-icon { background: rgba(245,158,11,0.08); }
.kpi-card.orange .kpi-icon svg { stroke: #F59E0B; }
.kpi-card.violet .kpi-icon { background: rgba(139,92,246,0.08); }
.kpi-card.violet .kpi-icon svg { stroke: #8B5CF6; }
.kpi-card.red    .kpi-icon { background: rgba(239,68,68,0.08); }
.kpi-card.red    .kpi-icon svg { stroke: #EF4444; }
.kpi-value {
    font-size: 2.1rem;
    font-weight: 800;
    color: #0B1629;
    line-height: 1.1;
}
.kpi-label {
    font-size: 0.73rem;
    font-weight: 700;
    color: #6B7280;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-top: 0.3rem;
}
.kpi-sub {
    font-size: 0.7rem;
    color: #9CA3AF;
    margin-top: 0.25rem;
    font-weight: 400;
}

/* ── Títulos de sección ── */
.section-header {
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
    margin-bottom: 1.75rem;
    padding-bottom: 1rem;
    border-bottom: 2px solid #E5E7EB;
    animation: fadeSlideIn 0.3s ease;
}
.section-header h2 {
    font-size: 1.4rem !important;
    font-weight: 800 !important;
    color: #0B1629 !important;
    margin: 0 0 0.15rem 0 !important;
}
.section-header-sub {
    font-size: 0.8rem;
    color: #6B7280;
    font-weight: 400;
    margin: 0;
}
.section-badge {
    background: #EEF2FF;
    color: #4F46E5;
    border-radius: 8px;
    padding: 0.2rem 0.75rem;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    border: 1px solid #C7D2FE;
    white-space: nowrap;
}

/* ── Panel/Card contenedor ── */
.panel-card {
    background: #FFFFFF;
    border-radius: 16px;
    padding: 1.75rem;
    box-shadow: 0 2px 16px rgba(15, 29, 53, 0.06);
    margin-bottom: 1.25rem;
    border: 1px solid #EEF2FF;
    animation: fadeSlideIn 0.4s ease;
    transition: box-shadow 0.2s ease;
}
.panel-card:hover {
    box-shadow: 0 4px 24px rgba(15, 29, 53, 0.10);
}
.panel-card h4 {
    font-size: 0.9rem !important;
    font-weight: 700 !important;
    color: #1B2A4A !important;
    margin-bottom: 0.75rem !important;
    letter-spacing: 0.2px;
}

/* ── Explain Box ── */
.explain-box {
    background: linear-gradient(135deg, #EFF6FF 0%, #F0F9FF 100%);
    border: 1px solid #BFDBFE;
    border-left: 4px solid #3B82F6;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 1.5rem;
    animation: fadeSlideIn 0.4s ease;
}
.explain-box .eb-title {
    font-size: 0.85rem;
    font-weight: 700;
    color: #1E40AF;
    margin-bottom: 0.35rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.explain-box .eb-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 1.1rem;
    height: 1.1rem;
    flex-shrink: 0;
}
.explain-box .eb-icon svg {
    width: 1rem;
    height: 1rem;
    stroke: #1E40AF;
    stroke-width: 2;
    fill: none;
    stroke-linecap: round;
    stroke-linejoin: round;
}
.explain-box .eb-body {
    font-size: 0.82rem;
    color: #374151;
    line-height: 1.65;
    margin: 0;
}

/* ── Semáforos de estado ── */
.semaforo {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    border-radius: 20px;
    padding: 0.22rem 0.85rem;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.3px;
}
.semaforo svg {
    width: 0.75rem;
    height: 0.75rem;
    stroke-width: 2.5;
    fill: none;
    stroke-linecap: round;
    stroke-linejoin: round;
    flex-shrink: 0;
}
.semaforo-bueno  { background: #D1FAE5; color: #065F46; }
.semaforo-bueno svg  { stroke: #065F46; }
.semaforo-medio  { background: #FEF3C7; color: #92400E; }
.semaforo-medio svg  { stroke: #92400E; }
.semaforo-malo   { background: #FEE2E2; color: #991B1B; }
.semaforo-malo svg   { stroke: #991B1B; }
.semaforo-neutro { background: #F3F4F6; color: #6B7280; }
.semaforo-neutro svg { stroke: #6B7280; }

/* ── Badges RIASEC por dimensión ── */
.riasec-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 26px; height: 26px;
    border-radius: 7px;
    font-size: 0.75rem;
    font-weight: 800;
}
.riasec-R { background: #DBEAFE; color: #1D4ED8; }
.riasec-I { background: #D1FAE5; color: #065F46; }
.riasec-A { background: #FEF3C7; color: #92400E; }
.riasec-S { background: #FCE7F3; color: #9D174D; }
.riasec-E { background: #EDE9FE; color: #5B21B6; }
.riasec-C { background: #CFFAFE; color: #0E7490; }

/* ── Métricas de Streamlit ── */
[data-testid="metric-container"] {
    background: #FFFFFF;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    box-shadow: 0 2px 10px rgba(15, 29, 53, 0.07);
    border: 1px solid #EEF2FF;
}
[data-testid="metric-container"] label {
    color: #6B7280 !important;
    font-size: 0.78rem !important;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #0B1629 !important;
    font-weight: 800 !important;
}

/* ── Tablas ── */
.stDataFrame {
    border-radius: 12px !important;
    overflow: hidden;
    box-shadow: 0 2px 12px rgba(15, 29, 53, 0.07);
    border: 1px solid #E5E7EB !important;
}

/* ── Botones ── */
.stButton > button {
    background: linear-gradient(135deg, #1D4ED8, #3B82F6) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 0.875rem !important;
    padding: 0.6rem 1.75rem !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 14px rgba(29, 78, 216, 0.3) !important;
    letter-spacing: 0.3px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1E40AF, #1D4ED8) !important;
    box-shadow: 0 6px 22px rgba(29, 78, 216, 0.45) !important;
    transform: translateY(-2px) !important;
}

/* ── Inputs y Selects ── */
.stSelectbox > div > div,
.stMultiSelect > div > div,
.stNumberInput > div > div > input,
.stTextInput > div > div > input {
    border-radius: 10px !important;
    border-color: #D1D5DB !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
.stSelectbox > div > div:focus-within,
.stTextInput > div > div:focus-within {
    border-color: #3B82F6 !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.15) !important;
}

/* ── Alertas ── */
.stInfo {
    background: #EFF6FF !important;
    border-color: #BFDBFE !important;
    color: #1E40AF !important;
    border-radius: 10px !important;
}
.stSuccess { border-radius: 10px !important; }
.stError   { border-radius: 10px !important; }

/* ── Spinner ── */
.stSpinner > div { border-color: #3B82F6 !important; }

/* ── Badge de estado ── */
.badge-active {
    display: inline-block;
    background: linear-gradient(135deg, #D1FAE5, #A7F3D0);
    color: #065F46;
    border-radius: 20px;
    padding: 0.2rem 0.85rem;
    font-size: 0.72rem;
    font-weight: 700;
}
.badge-inactive {
    display: inline-block;
    background: #F3F4F6;
    color: #6B7280;
    border-radius: 20px;
    padding: 0.2rem 0.85rem;
    font-size: 0.72rem;
    font-weight: 600;
}

/* ── Dividers ── */
hr {
    border-color: #E5E7EB !important;
    margin: 1.5rem 0 !important;
}

/* ── Títulos principales ── */
h1 { color: #0B1629 !important; font-weight: 800 !important; }
h2 { color: #0B1629 !important; font-weight: 700 !important; }
h3 { color: #1B2A4A !important; font-weight: 600 !important; }

/* ── Download button ── */
[data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg, #059669, #10B981) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 14px rgba(5, 150, 105, 0.3) !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: linear-gradient(135deg, #047857, #059669) !important;
    box-shadow: 0 6px 22px rgba(5, 150, 105, 0.45) !important;
    transform: translateY(-2px) !important;
}

/* ── Plotly charts ── */
.js-plotly-plot .plotly .main-svg { border-radius: 10px; }

/* ── Ocultar menú hamburguesa y footer de Streamlit ── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)







# --------------------------------------------------------------------------
# Utilidades de datos
# --------------------------------------------------------------------------
def usuarios_a_dataframe():
    session = get_session()
    try:
        registros = []
        usuarios = session.query(Usuario).all()
        for u in usuarios:
            fila = {
                "usuario_id": u.id, "sexo": u.sexo,
                "fecha_registro": u.fecha_registro,
            }
            if u.vector:
                fila.update({"R": u.vector.r, "I": u.vector.i, "A": u.vector.a,
                             "S": u.vector.s, "E": u.vector.e, "C": u.vector.c})
                puntajes = {'R': u.vector.r, 'I': u.vector.i, 'A': u.vector.a,
                            'S': u.vector.s, 'E': u.vector.e, 'C': u.vector.c}
                fila["dominante"] = max(puntajes, key=puntajes.get)
            else:
                fila["dominante"] = None
            registros.append(fila)
        return pd.DataFrame(registros)
    finally:
        session.close()


def resultados_a_dataframe(modelo_id=None):
    session = get_session()
    try:
        query = session.query(ResultadoClustering)
        if modelo_id:
            query = query.filter_by(modelo_id=modelo_id)
        registros = []
        for r in query.all():
            registros.append({
                "usuario_id": r.usuario_id, "cluster_id": r.cluster_id,
                "etiqueta_riasec": r.etiqueta_riasec,
                "probabilidades": json.loads(r.probabilidades_json),
            })
        return pd.DataFrame(registros)
    finally:
        session.close()


def kpi_card(value, label, sub="", color="blue", icon=""):
    icon_html = f'<div class="kpi-icon">{icon}</div>' if icon else ""
    return f"""
    <div class="kpi-card {color}">
        {icon_html}
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
        {"<div class='kpi-sub'>" + sub + "</div>" if sub else ""}
    </div>
    """


def section_header(title, badge="", subtitle=""):
    badge_html = f'<span class="section-badge">{badge}</span>' if badge else ""
    sub_html = f'<p class="section-header-sub">{subtitle}</p>' if subtitle else ""
    st.markdown(f"""
    <div class="section-header">
        <div style="flex:1">
            <h2>{title}</h2>
            {sub_html}
        </div>
        {badge_html}
    </div>
    """, unsafe_allow_html=True)


_ICON_INFO = '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>'
_ICON_CHECK = '<svg viewBox="0 0 24 24"><polyline points="20 6 9 17 4 12"/></svg>'
_ICON_WARN  = '<svg viewBox="0 0 24 24"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>'
_ICON_X     = '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>'
_ICON_USERS = '<svg viewBox="0 0 24 24"><path d="M17 21v-2a4 4 0 00-4-4H5a4 4 0 00-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 00-3-3.87"/><path d="M16 3.13a4 4 0 010 7.75"/></svg>'
_ICON_TARGET= '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>'
_ICON_CHART = '<svg viewBox="0 0 24 24"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>'
_ICON_BRAIN = '<svg viewBox="0 0 24 24"><path d="M9.5 2A2.5 2.5 0 017 4.5v0A2.5 2.5 0 014.5 7H4a2 2 0 00-2 2v0a2 2 0 002 2h.5A2.5 2.5 0 017 13.5v1A2.5 2.5 0 019.5 17h0A2.5 2.5 0 0112 14.5V14a2 2 0 012-2h0a2 2 0 012 2v.5a2.5 2.5 0 002.5 2.5h0A2.5 2.5 0 0021 14.5v-1A2.5 2.5 0 0018.5 11H18a2 2 0 01-2-2v0a2 2 0 012-2h.5A2.5 2.5 0 0021 4.5v0A2.5 2.5 0 0018.5 2h0A2.5 2.5 0 0016 4.5V5a2 2 0 01-2 2h0a2 2 0 01-2-2v-.5A2.5 2.5 0 009.5 2z"/></svg>'


def explain_box(title, body):
    """Cuadro azul con explicación contextual para el usuario."""
    st.markdown(f"""
    <div class="explain-box">
        <div class="eb-title"><span class="eb-icon">{_ICON_INFO}</span>{title}</div>
        <p class="eb-body">{body}</p>
    </div>
    """, unsafe_allow_html=True)


def semaforo_html(valor, bueno_min=0.5, medio_min=0.3, label=""):
    """Genera un badge semafórico (verde/amarillo/rojo) según el valor."""
    lbl = label or ""
    if valor is None:
        return f'<span class="semaforo semaforo-neutro">— Sin dato {lbl}</span>'
    if valor >= bueno_min:
        return f'<span class="semaforo semaforo-bueno">{_ICON_CHECK} Bueno · {valor:.3f} {lbl}</span>'
    elif valor >= medio_min:
        return f'<span class="semaforo semaforo-medio">{_ICON_WARN} Aceptable · {valor:.3f} {lbl}</span>'
    else:
        return f'<span class="semaforo semaforo-malo">{_ICON_X} Bajo · {valor:.3f} {lbl}</span>'


# --------------------------------------------------------------------------
# Barra lateral
# --------------------------------------------------------------------------
st.sidebar.markdown("""
<div class="sidebar-logo">
    <h1>RIASEC</h1>
    <span class="sidebar-subtitle">Análisis Vocacional · Unidad IV</span>
    <span class="sidebar-tagline">Conoce tu perfil · Descubre tu camino</span>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("<p style='font-size:0.7rem;color:#4A6B9A;text-transform:uppercase;letter-spacing:1px;padding:0 0.5rem;margin-bottom:0.4rem;'>Navegación</p>", unsafe_allow_html=True)

pestana = st.sidebar.radio(
    "Navegación",
    [
        "Dashboard",
        "Carga y Visualización",
        "Estadística Descriptiva",
        "Entrenamiento del Modelo",
        "Resultados",
        "Comparativa de Clústeres",
        "Metadatos del Modelo",
        "Descargas",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="padding: 0.5rem 0; color: #7B97C7; font-size: 0.68rem; text-align: center; letter-spacing: 0.5px; line-height: 1.7;">
    Gaussian Mixture Model<br>
    <span style="color: #4A6B9A;">Aprendizaje No Supervisado</span>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Utilidad: Parsear CSV exportado de Google Forms
# --------------------------------------------------------------------------
def es_csv_google_forms(df_raw: pd.DataFrame) -> bool:
    """Detecta si el DataFrame proviene de un formulario de Google."""
    primera = str(df_raw.columns[0]).strip().lower()
    return "marca temporal" in primera or "timestamp" in primera


def _score_respuesta(respuesta: str) -> int:
    """
    Convierte una respuesta cualitativa a puntaje 0/3/6.
    Respuestas positivas -> 6, neutrales -> 3, negativas -> 0.
    """
    if pd.isna(respuesta):
        return 3  # valor neutral por defecto

    def _norm(s):
        """Normaliza acentos y convierte a minusculas para comparacion robusta."""
        return ''.join(
            c for c in unicodedata.normalize('NFD', str(s).strip().lower())
            if unicodedata.category(c) != 'Mn'
        )

    r = _norm(respuesta)

    # Palabras clave de alta afinidad -> 6
    alto = [
        "si, lo prefiero", "si, me identifica mucho", "me interesa mucho",
        "siempre", "mucho", "si, frecuentemente", "si, es una de mis formas favoritas",
        "casi siempre", "siempre que puedo", "si, con facilidad", "si, definitivamente",
        "si, disfruto liderar", "si, sin problema", "si, siempre",
        "si, me siento mas comodo(a)", "si, de manera constante",
    ]
    # Palabras clave neutras -> 3
    medio = [
        "a veces me identifica", "depende de la situacion", "me interesa un poco",
        "algunas veces", "de vez en cuando", "en algunas ocasiones",
        "solo cuando el tema me interesa", "prefiero soluciones tradicionales",
        "solo en algunas ocasiones", "depende de la persona", "depende de la actividad",
        "solo cuando es necesario", "solo si el riesgo es razonable", "tal vez",
        "prefiero tener libertad para decidir",
    ]

    if r in alto:
        return 6
    if r in medio:
        return 3
    # Cualquier otra respuesta negativa -> 0
    return 0


# Mapeo de dimensiones RIASEC -> indice de columna en el CSV de Google Forms
# col[0]=Marca temporal, col[1]=Sexo, col[2-4]=R, col[5-7]=I,
# col[8-10]=A, col[11-13]=S, col[14-16]=E, col[17-19]=C
_GF_DIMENSION_COLS = {
    "R": [2, 3, 4],
    "I": [5, 6, 7],
    "A": [8, 9, 10],
    "S": [11, 12, 13],
    "E": [14, 15, 16],
    "C": [17, 18, 19],
}


def parsear_csv_google_forms(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Transforma el CSV de Google Forms al formato estandar:
    sexo, R, I, A, S, E, C.
    """
    registros = []
    for _, fila in df_raw.iterrows():
        # Sexo
        sexo_raw = str(fila.iloc[1]).strip() if len(fila) > 1 else ""
        sexo = "M" if "hombre" in sexo_raw.lower() else "F" if "mujer" in sexo_raw.lower() else sexo_raw

        # Calcular puntaje por dimension (promedio de 3 preguntas, rango 0-6)
        scores = {}
        for dim, cols in _GF_DIMENSION_COLS.items():
            puntos = []
            for col_idx in cols:
                if col_idx < len(fila):
                    puntos.append(_score_respuesta(fila.iloc[col_idx]))
            scores[dim] = round(sum(puntos) / len(puntos)) if puntos else 0

        registros.append({
            "sexo": sexo,
            "R": scores["R"],
            "I": scores["I"],
            "A": scores["A"],
            "S": scores["S"],
            "E": scores["E"],
            "C": scores["C"],
        })
    return pd.DataFrame(registros)


# --------------------------------------------------------------------------
# 0. DASHBOARD
# --------------------------------------------------------------------------
if pestana == "Dashboard":
    section_header("Dashboard", "Resumen General", "Vista general del estado del análisis vocacional RIASEC")

    df = usuarios_a_dataframe()
    modelo, registro = cargar_modelo_activo()

    explain_box(
        "¿Qué ves aquí?",
        "Esta pantalla muestra el resumen del sistema. Los <b>números de arriba</b> indican cuántos estudiantes hay registrados "
        "y cómo quedó el último análisis de grupos. Las <b>gráficas</b> muestran cómo se distribuyen los perfiles vocacionales."
    )

    #  KPI Cards 
    c1, c2, c3, c4 = st.columns(4)

    total_usuarios = len(df)
    with c1:
        st.markdown(kpi_card(
            str(total_usuarios), "Estudiantes Registrados",
            sub="Total en base de datos", color="blue", icon=_ICON_USERS
        ), unsafe_allow_html=True)

    if modelo is not None:
        n_componentes_activo = registro.n_componentes
        silhouette_val = registro.silhouette_score
        algo = registro.algoritmo
        n_registros_ent = registro.n_registros_entrenamiento

        with c2:
            st.markdown(kpi_card(
                f"{n_componentes_activo}", "Grupos Vocacionales",
                sub=f"Algoritmo: {algo}", color="violet", icon=_ICON_TARGET
            ), unsafe_allow_html=True)

        with c3:
            sil_str = f"{silhouette_val:.3f}" if silhouette_val is not None else "N/A"
            if silhouette_val and silhouette_val > 0.5:
                sil_sub = "Separación excelente"
                sil_color = "green"
            elif silhouette_val and silhouette_val > 0.3:
                sil_sub = "Separación aceptable"
                sil_color = "orange"
            else:
                sil_sub = "Separación baja — reentrenar"
                sil_color = "red" if silhouette_val else "green"
            st.markdown(kpi_card(
                sil_str, "Calidad de Grupos",
                sub=sil_sub, color=sil_color, icon=_ICON_CHART
            ), unsafe_allow_html=True)

        with c4:
            st.markdown(kpi_card(
                str(n_registros_ent), "Analizados en Último Modelo",
                sub=f"Modelo #{registro.id}", color="orange", icon=_ICON_BRAIN
            ), unsafe_allow_html=True)
    else:
        with c2:
            st.markdown(kpi_card("—", "Grupos Vocacionales", sub="Sin modelo entrenado", color="violet", icon=_ICON_TARGET), unsafe_allow_html=True)
        with c3:
            st.markdown(kpi_card("—", "Calidad de Grupos", sub="Entrenar modelo primero", color="green", icon=_ICON_CHART), unsafe_allow_html=True)
        with c4:
            st.markdown(kpi_card("—", "Analizados en Último Modelo", sub="—", color="orange", icon=_ICON_BRAIN), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    #  Gráficas principales 
    if not df.empty and modelo is not None:
        col_left, col_right = st.columns(2)

        # PCA Scatter
        with col_left:
            st.markdown('<div class="panel-card"><h4>Proyección PCA 2D — Distribución de Perfiles</h4>', unsafe_allow_html=True)
            session = get_session()
            usuario_ids, X = obtener_matriz_entrenamiento(session)
            session.close()
            proyeccion, varianza = proyeccion_pca_2d(X)
            df_resultados = resultados_a_dataframe(modelo_id=registro.id)
            df_plot = pd.DataFrame({
                "usuario_id": usuario_ids,
                "PCA_1": proyeccion[:, 0],
                "PCA_2": proyeccion[:, 1],
            }).merge(df_resultados[["usuario_id", "etiqueta_riasec"]], on="usuario_id", how="left")
            fig_pca = px.scatter(
                df_plot, x="PCA_1", y="PCA_2", color="etiqueta_riasec",
                hover_data=["usuario_id"],
                color_discrete_sequence=px.colors.qualitative.Set2,
                labels={"PCA_1": "Componente 1", "PCA_2": "Componente 2", "etiqueta_riasec": "Perfil"},
            )
            fig_pca.update_traces(marker=dict(size=8, opacity=0.8, line=dict(width=0.5, color="white")))
            fig_pca.update_layout(
                height=360,
                margin=dict(l=10, r=10, t=20, b=10),
                paper_bgcolor="white",
                plot_bgcolor="#F8FAFC",
                legend=dict(font=dict(size=10), orientation="v", x=1.01, y=1),
                font=dict(family="Inter", size=11),
            )
            st.plotly_chart(fig_pca, width='stretch')
            st.caption(
                f"Cada punto representa un estudiante · El color indica su perfil vocacional predominante · "
                f"Puntos cercanos = perfiles similares · Varianza explicada: **{round(sum(varianza)*100, 1)}%**"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        # Radar de Centroides
        with col_right:
            st.markdown('<div class="panel-card"><h4>Radar de Grupos — Perfil Promedio</h4>', unsafe_allow_html=True)
            from src.application.entrenamiento_service import etiquetar_clusters
            centroides = modelo.means_
            etiquetas = etiquetar_clusters(centroides)
            fig_radar = go.Figure()
            colores_radar = ["#3B82F6", "#22C55E", "#F59E0B", "#8B5CF6", "#EF4444", "#06B6D4"]

            def _hex_rgba(hex_color, alpha=0.15):
                h = hex_color.lstrip("#")
                r2, g2, b2 = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
                return f"rgba({r2},{g2},{b2},{alpha})"

            for idx, centro in enumerate(centroides):
                color = colores_radar[idx % len(colores_radar)]
                fig_radar.add_trace(go.Scatterpolar(
                    r=list(centro) + [centro[0]],
                    theta=DIMENSIONES + [DIMENSIONES[0]],
                    fill="toself",
                    name=f"C{idx}: {etiquetas[idx].replace('Predominantemente ', '')}",
                    line=dict(color=color, width=2),
                    fillcolor=_hex_rgba(color),
                ))
            fig_radar.update_layout(
                height=360,
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 6], tickfont=dict(size=9)),
                    angularaxis=dict(tickfont=dict(size=11, family="Plus Jakarta Sans", color="#374151")),
                    bgcolor="#F8FAFC",
                ),
                margin=dict(l=40, r=40, t=20, b=20),
                paper_bgcolor="white",
                legend=dict(font=dict(size=9), x=1.02, y=1),
                font=dict(family="Plus Jakarta Sans"),
            )
            st.plotly_chart(fig_radar, width='stretch')
            st.caption("Cada eje (R, I, A, S, E, C) va de 0 a 6 · Mientras mayor el área, más fuerte es ese perfil en el grupo")
            st.markdown('</div>', unsafe_allow_html=True)

    elif df.empty:
        st.info("No hay datos cargados todavía. Ve a la sección **Carga y Visualización** para subir tu primer archivo CSV.")
    else:
        st.info("Hay datos disponibles. Ve a **Entrenamiento del Modelo** para agrupar los perfiles vocacionales.")

    st.markdown("<br>", unsafe_allow_html=True)

    #  Fila inferior: distribuciones + últimos registros 
    if not df.empty:
        col_a, col_b, col_c = st.columns([1, 1, 2])

        with col_a:
            st.markdown('<div class="panel-card"><h4>Distribución por Sexo</h4>', unsafe_allow_html=True)
            conteo_sexo, _ = distribucion_por_categoria(df.to_dict("records"), "sexo")
            if conteo_sexo:
                fig_pie = px.pie(
                    values=list(conteo_sexo.values()),
                    names=list(conteo_sexo.keys()),
                    color_discrete_sequence=["#3B82F6", "#EC4899"],
                    hole=0.45,
                )
                fig_pie.update_traces(textposition="outside", textinfo="percent+label",
                                      textfont=dict(size=11, family="Inter"))
                fig_pie.update_layout(
                    height=240, margin=dict(l=0, r=0, t=10, b=0),
                    paper_bgcolor="white", showlegend=False,
                    font=dict(family="Inter"),
                )
                st.plotly_chart(fig_pie, width='stretch')
            st.markdown('</div>', unsafe_allow_html=True)

        with col_b:
            st.markdown('<div class="panel-card"><h4>Distribución de Géneros</h4>', unsafe_allow_html=True)
            if "sexo" in df.columns:
                dist_sexo = (
                    df["sexo"].dropna()
                    .value_counts()
                    .reset_index()
                )
                dist_sexo.columns = ["sexo", "count"]
                if not dist_sexo.empty:
                    fig_bar = px.bar(
                        dist_sexo, x="count", y="sexo",
                        orientation="h",
                        color="count",
                        color_continuous_scale=["#BFDBFE", "#2563EB"],
                        labels={"count": "", "sexo": ""},
                    )
                    fig_bar.update_layout(
                        height=240, margin=dict(l=0, r=10, t=10, b=0),
                        paper_bgcolor="white", plot_bgcolor="white",
                        coloraxis_showscale=False,
                        yaxis=dict(tickfont=dict(size=9), autorange="reversed"),
                        xaxis=dict(showgrid=False, showticklabels=False),
                        font=dict(family="Inter"),
                    )
                    fig_bar.update_traces(marker_line_width=0)
                    st.plotly_chart(fig_bar, width='stretch')
                else:
                    st.caption("Sin datos de sexo disponibles.")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_c:
            st.markdown('<div class="panel-card"><h4>Últimos Registros</h4>', unsafe_allow_html=True)
            cols_mostrar = [c for c in ["usuario_id", "sexo", "fecha_registro"] if c in df.columns]
            ultimos = df.sort_values("fecha_registro", ascending=False).head(8)[cols_mostrar]
            if "fecha_registro" in ultimos.columns:
                ultimos["fecha_registro"] = pd.to_datetime(ultimos["fecha_registro"]).dt.strftime("%Y-%m-%d")
            st.dataframe(ultimos, width='stretch', hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)



# --------------------------------------------------------------------------
# 1. CARGA Y VISUALIZACIÓN
# --------------------------------------------------------------------------
elif pestana == "Carga y Visualización":
    section_header("Carga de Datos y Visualización", "Importación · Filtros",
                   "Sube el archivo CSV con las respuestas del cuestionario RIASEC y explóralos aquí")

    explain_box(
        "Cómo cargar datos",
        "<b>Paso 1:</b> Sube tu archivo CSV con las respuestas del cuestionario. "
        "<b>Paso 2:</b> Revisa la vista previa. "
        "<b>Paso 3:</b> Haz clic en <em>Confirmar e insertar</em> para guardarlos. "
        "Formatos aceptados: CSV estándar o exportación directa de Google Forms."
    )
    st.caption(
        "Formatos soportados: **CSV estándar** con columnas `sexo`, `R`, `I`, `A`, `S`, `E`, `C` "
        "— o directamente el **CSV exportado de Google Forms** (se convierte automáticamente)."
    )
    archivo = st.file_uploader("Selecciona un archivo CSV", type=["csv"])

    if archivo is not None:
        df_raw = pd.read_csv(archivo)
        columnas_estandar = {"sexo", "R", "I", "A", "S", "E", "C"}
        es_forms = es_csv_google_forms(df_raw)

        if es_forms:
            # ── Formato Google Forms ──────────────────────────────────────
            st.info(
                "**Formato Google Forms detectado** — Las respuestas de texto se convierten "
                "automáticamente a puntajes RIASEC (0–6)."
            )
            df_nuevo = parsear_csv_google_forms(df_raw)
            st.success(f"{len(df_nuevo)} registros convertidos correctamente.")
            with st.expander("Vista previa de los datos convertidos (primeros 5)", expanded=True):
                st.dataframe(df_nuevo.head(5), use_container_width=True, hide_index=True)

        elif columnas_estandar.issubset(set(df_raw.columns)):
            # ── Formato estándar ──────────────────────────────────────────
            df_nuevo = df_raw
            st.success(f"Archivo válido — {len(df_nuevo)} registros detectados.")
            with st.expander("Vista previa (primeros 5)", expanded=True):
                st.dataframe(df_nuevo.head(5), use_container_width=True, hide_index=True)

        else:
            # ── Error: formato desconocido ────────────────────────────────
            st.error(
                "Formato de archivo no reconocido.\n\n"
                "**Opciones válidas:**\n"
                "- CSV estándar con columnas: `sexo`, `R`, `I`, `A`, `S`, `E`, `C`\n"
                "- CSV exportado directamente de Google Forms (detección automática)"
            )
            with st.expander("Columnas encontradas en el archivo"):
                st.write(list(df_raw.columns))
            df_nuevo = None

        if 'df_nuevo' in dir() and df_nuevo is not None:
            if st.button("Confirmar e insertar en la base de datos", type="primary"):
                session = get_session()
                try:
                    insertados = 0
                    for _, fila in df_nuevo.iterrows():
                        usuario = Usuario(
                            sexo=fila.get("sexo"),
                            fecha_registro=datetime.datetime.utcnow(),
                        )
                        session.add(usuario)
                        session.flush()
                        session.add(VectorRiasec(
                            usuario_id=usuario.id,
                            r=int(fila["R"]), i=int(fila["I"]),
                            a=int(fila["A"]), s=int(fila["S"]),
                            e=int(fila["E"]), c=int(fila["C"]),
                        ))
                        insertados += 1
                    session.commit()
                    st.success(f"Se insertaron **{insertados}** registros nuevos correctamente.")
                    st.balloons()
                except Exception as ex:
                    session.rollback()
                    st.error(f"Error al insertar: {ex}")
                finally:
                    session.close()
    st.markdown('</div>', unsafe_allow_html=True)

    #  Tabla y filtros 
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown("#### Explorador de Datos Actuales")
    df = usuarios_a_dataframe()

    if df.empty:
        st.info("Aún no hay registros. Carga un CSV o ejecuta `seed.py` desde consola.")
    else:
        df_filtrado = df.copy()

        # Obtener resultados del modelo activo para filtrar por cluster
        modelo_activo, _ = cargar_modelo_activo()
        if modelo_activo:
            df_resultados = resultados_a_dataframe(modelo_id=modelo_activo.id)
            if not df_resultados.empty:
                # Hacer merge left para añadir la etiqueta_riasec
                df = df.merge(df_resultados[["usuario_id", "etiqueta_riasec"]], on="usuario_id", how="left")
                df_filtrado = df.copy()

        col1, col2 = st.columns(2)
        with col1:
            filtro_sexo = st.multiselect("Filtrar por sexo", options=df["sexo"].dropna().unique().tolist())
            opciones_dom = df["dominante"].dropna().unique().tolist() if "dominante" in df.columns else []
            filtro_dom = st.multiselect("Dimensión Dominante", options=opciones_dom)
            
        with col2:
            fecha_min = df["fecha_registro"].min()
            fecha_max = df["fecha_registro"].max()
            rango_fecha = st.date_input("Rango de fecha de registro", (fecha_min.date(), fecha_max.date()))
            if "etiqueta_riasec" in df.columns:
                opciones_cluster = df["etiqueta_riasec"].dropna().unique().tolist()
                filtro_cluster = st.multiselect("Clúster Asignado", options=opciones_cluster)
            else:
                filtro_cluster = []

        if filtro_sexo:
            df_filtrado = df_filtrado[df_filtrado["sexo"].isin(filtro_sexo)]
        if filtro_dom:
            df_filtrado = df_filtrado[df_filtrado["dominante"].isin(filtro_dom)]
        if filtro_cluster:
            df_filtrado = df_filtrado[df_filtrado["etiqueta_riasec"].isin(filtro_cluster)]

        if isinstance(rango_fecha, tuple) and len(rango_fecha) == 2:
            df_filtrado = df_filtrado[
                (df_filtrado["fecha_registro"].dt.date >= rango_fecha[0]) &
                (df_filtrado["fecha_registro"].dt.date <= rango_fecha[1])
            ]

        st.markdown(f"<p style='color:#6B7280; font-size:0.85rem; margin-bottom:0.5rem;'>Mostrando <b>{len(df_filtrado)}</b> de <b>{len(df)}</b> registros</p>", unsafe_allow_html=True)
        st.dataframe(df_filtrado, width='stretch', hide_index=True)
        st.session_state["df_filtrado"] = df_filtrado

        # Mini distribución visual de vectores RIASEC filtrados
        if all(c in df_filtrado.columns for c in ["R", "I", "A", "S", "E", "C"]):
            st.markdown("---")
            st.markdown("#### Distribución Media de Dimensiones RIASEC (datos filtrados)")
            promedios = df_filtrado[["R", "I", "A", "S", "E", "C"]].mean().reset_index()
            promedios.columns = ["Dimensión", "Promedio"]
            promedios["Nombre"] = promedios["Dimensión"].map(NOMBRES_DIMENSION)
            fig_dim = px.bar(
                promedios, x="Dimensión", y="Promedio",
                text="Promedio",
                color="Promedio",
                color_continuous_scale=["#BFDBFE", "#1D4ED8"],
                labels={"Dimensión": "Dimensión RIASEC", "Promedio": "Media"},
                custom_data=["Nombre"],
            )
            fig_dim.update_traces(
                texttemplate="%{text:.2f}",
                textposition="outside",
                hovertemplate="<b>%{customdata[0]}</b><br>Media: %{y:.2f}<extra></extra>",
                marker_line_width=0,
            )
            fig_dim.update_layout(
                height=300, margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="white", plot_bgcolor="white",
                coloraxis_showscale=False,
                yaxis=dict(range=[0, 7], showgrid=True, gridcolor="#F0F2F6"),
                font=dict(family="Inter"),
            )
            st.plotly_chart(fig_dim, width='stretch')
    st.markdown('</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------
# 2. ESTADÍSTICA DESCRIPTIVA
# --------------------------------------------------------------------------
elif pestana == "Estadística Descriptiva":
    section_header("Estadística Descriptiva", "Algoritmos Propios",
                   "Resumen numérico de los puntajes RIASEC de todos los estudiantes registrados")

    df = usuarios_a_dataframe()
    if df.empty:
        st.info("No hay datos suficientes. Carga información primero en la sección Carga y Visualización.")
    else:
        explain_box(
            "¿Qué muestra esta sección?",
            "Aquí ves cómo se distribuyen los puntajes de cada dimensión RIASEC. "
            "El <b>promedio</b> indica el nivel típico del grupo. La <b>dispersión</b> indica qué tan diferentes son los estudiantes entre sí. "
            "Los <b>gráficos de caja</b> muestran dónde se concentra la mayoría de los valores (rango del 25% al 75%)."
        )
        vectores = df[["R", "I", "A", "S", "E", "C"]].rename(columns=str.lower).to_dict("records")
        resumen = resumen_dimensiones(vectores)

        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown("#### Resumen Estadístico por Dimensión RIASEC")
        st.caption(
            "Calculado con algoritmos propios · "
            "**Promedio**: valor típico del grupo · **Mediana**: valor del centro · "
            "**Moda**: valor más repetido · **Dispersión**: qué tanto varían los valores (0 = todos igual)"
        )
        df_resumen = pd.DataFrame(resumen).T
        df_resumen.index.name = "Dimensión"
        df_resumen = df_resumen.rename(columns={
            "media": "Promedio",
            "mediana": "Mediana",
            "moda": "Moda",
            "desviacion_estandar": "Dispersión",
            "minimo": "Mínimo",
            "maximo": "Máximo",
        })
        st.dataframe(df_resumen.style.format({
            "Promedio": "{:.2f}",
            "Mediana": "{:.2f}",
            "Dispersión": "{:.2f}",
        }), width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown("#### Distribución de Puntajes por Dimensión (Box Plot)")
        st.caption(
            "La caja = rango donde está el 50% central de los estudiantes · La línea del medio = mediana · "
            "Los puntos fuera = estudiantes con perfiles atípicos"
        )
        dim_cols = [c for c in ["R", "I", "A", "S", "E", "C"] if c in df.columns]
        df_melt = df[dim_cols].melt(var_name="Dimensión", value_name="Puntuación")
        df_melt["Nombre"] = df_melt["Dimensión"].map(NOMBRES_DIMENSION)
        colores_box = {"R": "#3B82F6", "I": "#22C55E", "A": "#F59E0B", "S": "#EC4899", "E": "#8B5CF6", "C": "#06B6D4"}
        fig_box = px.box(
            df_melt, x="Dimensión", y="Puntuación",
            color="Dimensión",
            color_discrete_map=colores_box,
            points="all",
            hover_data=["Nombre"],
            labels={"Puntuación": "Puntuación (0–6)", "Dimensión": "Dimensión RIASEC"},
        )
        fig_box.update_traces(marker=dict(size=4, opacity=0.5))
        fig_box.update_layout(
            height=360, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="white", plot_bgcolor="#F8FAFC",
            showlegend=False,
            yaxis=dict(range=[-0.5, 6.5], showgrid=True, gridcolor="#E5E7EB"),
            font=dict(family="Inter"),
        )
        st.plotly_chart(fig_box, width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)

        #  Distribución por categoría 
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown("#### Distribución por Campo Categórico")
        campo = st.selectbox("Selecciona campo", ["sexo"])
        conteo, porcentaje = distribucion_por_categoria(df.to_dict("records"), campo)

        col1, col2 = st.columns(2)
        with col1:
            df_conteo = pd.DataFrame({"Categoría": list(conteo.keys()), "Cantidad": list(conteo.values())})
            df_conteo = df_conteo.sort_values("Cantidad", ascending=True)
            fig_cat = px.bar(
                df_conteo, x="Cantidad", y="Categoría", orientation="h",
                color="Cantidad",
                color_continuous_scale=["#BFDBFE", "#1D4ED8"],
                labels={"Cantidad": "N° de usuarios", "Categoría": ""},
            )
            fig_cat.update_layout(
                height=max(200, len(conteo) * 35 + 80),
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="white", plot_bgcolor="white",
                coloraxis_showscale=False,
                yaxis=dict(tickfont=dict(size=10)),
                font=dict(family="Inter"),
            )
            fig_cat.update_traces(marker_line_width=0)
            st.plotly_chart(fig_cat, width='stretch')

        with col2:
            df_pct = pd.DataFrame({
                "Categoría": list(porcentaje.keys()),
                "Porcentaje (%)": list(porcentaje.values())
            }).sort_values("Porcentaje (%)", ascending=False)
            st.dataframe(df_pct, width='stretch', hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------
# 3. ENTRENAMIENTO
# --------------------------------------------------------------------------
elif pestana == "Entrenamiento del Modelo":
    section_header("Entrenamiento del Modelo", "Gaussian Mixture Model",
                   "Agrupa automáticamente a los estudiantes en perfiles vocacionales similares")

    explain_box(
        "¿Cómo funciona el entrenamiento?",
        "El modelo analiza los puntajes RIASEC de cada estudiante y los agrupa automáticamente en grupos de "
        "perfiles similares (sin necesidad de definirlos manualmente). Tú decides <b>cuántos grupos</b> buscar. "
        "El algoritmo aprende sólo a partir de los datos."
    )

    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown("#### Configuración del Entrenamiento")
    st.caption("El modelo agrupa los vectores [R, I, A, S, E, C] de forma no supervisada y etiqueta cada grupo según la dimensión dominante.")

    col1, col2, col3 = st.columns(3)
    with col1:
        n_componentes = st.number_input(
            "Número de grupos vocacionales",
            min_value=2, max_value=12, value=6,
            help="Cuántos grupos de perfiles vocacionales distintos buscará el modelo. Recomendado: 6 (uno por dimensión RIASEC)."
        )
    with col2:
        algoritmo = st.selectbox(
            "Algoritmo",
            ["GaussianMixture", "KMeans"],
            help="KMeans hace grupos rígidos (rápidos), GaussianMixture permite que un usuario pertenezca a varios (probabilístico)."
        )
    with col3:
        if algoritmo == "GaussianMixture":
            covariance_type = st.selectbox(
                "Tipo de covarianza",
                ["full", "tied", "diag", "spherical"],
                help="'full' (recomendado): cada grupo tiene su propia forma. 'spherical': todos los grupos son circulares (más simple)."
            )
        else:
            covariance_type = "N/A"
            st.info("No aplica para KMeans")

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        entrenar_btn = st.button("Iniciar Entrenamiento", width='stretch')
    with col_info:
        st.markdown("""
        <div style="background:#EFF6FF; border:1px solid #BFDBFE; border-radius:10px; padding:0.6rem 1rem; font-size:0.82rem; color:#1E40AF;">
            <b>Importante:</b> Al entrenar, el modelo anterior quedará inactivo.
            Necesitas al menos <b>{n}</b> estudiantes registrados para crear <b>{n}</b> grupos.
        </div>
        """.replace("{n}", str(int(n_componentes))), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if entrenar_btn:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        with st.spinner("Analizando y agrupando perfiles RIASEC..."):
            try:
                resultado = entrenar_modelo(n_componentes=n_componentes, covariance_type=covariance_type, algoritmo=algoritmo)
                st.success(f"Modelo **#{resultado['modelo_id']}** entrenado con **{resultado['n_registros']}** estudiantes.")

                m1, m2, m3 = st.columns(3)
                sil_val = resultado['silhouette']
                with m1:
                    sil_str = f"{sil_val:.4f}" if sil_val else "N/A"
                    st.metric(
                        "Calidad de Separación",
                        sil_str,
                        help="Silhouette Score: va de -1 a 1. Mayor a 0.5 = excelente, mayor a 0.3 = aceptable."
                    )
                    st.markdown(semaforo_html(sil_val), unsafe_allow_html=True)
                with m2:
                    if resultado['bic'] is not None:
                        st.metric("BIC", f"{resultado['bic']:.1f}", help="Bayesian Information Criterion: penaliza la complejidad del modelo. Menor = mejor.")
                    else:
                        st.metric("BIC", "N/A", help="No aplica para KMeans")
                with m3:
                    if resultado['aic'] is not None:
                        st.metric("AIC", f"{resultado['aic']:.1f}", help="Akaike Information Criterion: similar al BIC. Menor = mejor.")
                    else:
                        st.metric("AIC", "N/A", help="No aplica para KMeans")

                st.markdown("---")
                st.markdown("#### Perfiles Vocacionales Identificados")
                st.caption("Cada grupo recibe una etiqueta basada en la dimensión RIASEC con mayor puntaje promedio.")
                etiquetas_df = pd.DataFrame([
                    {"Grupo": f"Grupo {k+1}", "Perfil Vocacional": v}
                    for k, v in resultado["etiquetas"].items()
                ])
                st.dataframe(etiquetas_df, width='stretch', hide_index=True)
            except ValueError as e:
                st.error(f"{str(e)}")
        st.markdown('</div>', unsafe_allow_html=True)

    # Historial de Modelos
    st.markdown('<div class="panel-card" style="margin-top: 1rem;">', unsafe_allow_html=True)
    st.markdown("#### Historial de Modelos")
    historial = obtener_historial_modelos()
    if not historial:
        st.info("No hay modelos entrenados en el historial.")
    else:
        datos_historial = []
        for m in historial:
            datos_historial.append({
                "ID": m.id,
                "Fecha": m.fecha_entrenamiento.strftime("%Y-%m-%d %H:%M"),
                "Algoritmo": m.algoritmo,
                "Grupos": m.n_componentes,
                "Silueta": round(m.silhouette_score, 4) if m.silhouette_score else "N/A",
                "Estado": "ACTIVO 🟢" if m.activo else "Inactivo ⚪"
            })
        
        st.dataframe(pd.DataFrame(datos_historial), hide_index=True, width='stretch')
        
        st.markdown("##### Activar un modelo anterior")
        col_sel, col_btn2 = st.columns([2, 1])
        with col_sel:
            opciones_modelos = {f"ID: {m.id} | {m.algoritmo} | {m.fecha_entrenamiento.strftime('%Y-%m-%d')}": m.id for m in historial}
            modelo_seleccionado = st.selectbox("Selecciona el modelo que deseas activar:", options=list(opciones_modelos.keys()))
        with col_btn2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Activar Modelo", key="btn_activar"):
                if activar_modelo_historico(opciones_modelos[modelo_seleccionado]):
                    st.success("¡Modelo activado exitosamente!")
                    st.rerun()
                else:
                    st.error("Hubo un problema al activar el modelo.")
    st.markdown('</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------
# 4. RESULTADOS
# --------------------------------------------------------------------------
elif pestana == "Resultados":
    section_header("Resultados de la Clasificación", "PCA · Asignaciones",
                   "Aquí ves cómo quedó clasificado cada estudiante después del análisis")

    modelo, registro = cargar_modelo_activo()
    if modelo is None:
        st.info("Aún no hay ningún modelo entrenado. Ve a **Entrenamiento del Modelo** para generar uno.")
    else:
        explain_box(
            "¿Cómo leer este análisis?",
            "Cada <b>punto en la gráfica</b> representa un estudiante. El <b>color</b> indica a qué perfil vocacional pertenece. "
            "Puntos muy cercanos entre sí = estudiantes con perfiles muy similares. La tabla de abajo muestra el grupo asignado a cada estudiante."
        )
        session = get_session()
        usuario_ids, X = obtener_matriz_entrenamiento(session)
        session.close()

        proyeccion, varianza = proyeccion_pca_2d(X)
        df_resultados = resultados_a_dataframe(modelo_id=registro.id)
        df_plot = pd.DataFrame({
            "usuario_id": usuario_ids,
            "PCA_1": proyeccion[:, 0],
            "PCA_2": proyeccion[:, 1],
        }).merge(df_resultados[["usuario_id", "etiqueta_riasec"]], on="usuario_id", how="left")

        # KPIs rápidos
        r1, r2, r3 = st.columns(3)
        r1.metric("Estudiantes clasificados", len(df_resultados),
                  help="Total de estudiantes que fueron asignados a un grupo vocacional.")
        r2.metric("Perfiles distintos encontrados",
                  df_resultados["cluster_id"].nunique() if not df_resultados.empty else "—",
                  help="Cuántos grupos vocacionales diferentes identificó el modelo.")
        r3.metric("Precisión de la visualización",
                  f"{round(sum(varianza)*100, 1)}%",
                  help="Porcentaje de información conservada al reducir de 6 dimensiones a 2 para la gráfica.")

        st.markdown("<br>", unsafe_allow_html=True)

        # Scatter PCA
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown("#### Mapa de Perfiles Vocacionales")
        fig = px.scatter(
            df_plot, x="PCA_1", y="PCA_2", color="etiqueta_riasec",
            hover_data=["usuario_id"],
            color_discrete_sequence=px.colors.qualitative.Set2,
            labels={"PCA_1": "Eje 1", "PCA_2": "Eje 2", "etiqueta_riasec": "Perfil Vocacional"},
        )
        fig.update_traces(marker=dict(size=9, opacity=0.82, line=dict(width=0.5, color="white")))
        fig.update_layout(
            height=450,
            paper_bgcolor="white", plot_bgcolor="#F8FAFC",
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(font=dict(size=10, family="Plus Jakarta Sans"), orientation="v", x=1.01, y=1,
                        bgcolor="rgba(255,255,255,0.8)", bordercolor="#E5E7EB", borderwidth=1),
            font=dict(family="Plus Jakarta Sans"),
        )
        st.plotly_chart(fig, width='stretch')
        st.caption("Cada punto = un estudiante · El color = su perfil vocacional · Puntos del mismo color = perfiles similares")
        st.markdown('</div>', unsafe_allow_html=True)

        # Tabla de asignaciones
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown("#### Asignación de Perfil por Estudiante")
        st.caption("Cada fila muestra a qué grupo vocacional fue asignado cada estudiante según sus respuestas.")
        df_tabla = df_resultados[["usuario_id", "cluster_id", "etiqueta_riasec"]].copy()
        df_tabla = df_tabla.rename(columns={"usuario_id": "ID Estudiante", "cluster_id": "Grupo #", "etiqueta_riasec": "Perfil Vocacional"})
        st.dataframe(df_tabla, width='stretch', hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------
# 5. COMPARATIVA DE CLÚSTERES
# --------------------------------------------------------------------------
elif pestana == "Comparativa de Clústeres":
    section_header("Comparativa entre Grupos Vocacionales", "Gráfico Radar",
                   "Compara visualmente el perfil RIASEC promedio de cada grupo identificado")

    modelo, registro = cargar_modelo_activo()
    if modelo is None:
        st.info("Aún no hay ningún modelo entrenado.")
    else:
        from src.application.entrenamiento_service import etiquetar_clusters
        centroides = modelo.means_
        etiquetas = etiquetar_clusters(centroides)

        explain_box(
            "¿Cómo leer el radar?",
            "Cada <b>polígono de color</b> representa un grupo vocacional. Cada <b>eje</b> (R, I, A, S, E, C) va de 0 a 6. "
            "Mientras más grande el área en un eje, más fuerte es ese perfil en ese grupo. "
            "Grupos que se <b>superponen</b> son similares; grupos <b>separados</b> son muy distintos."
        )

        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown("#### Gráfico Radar — Perfil Promedio por Grupo")

        colores_radar = ["#3B82F6", "#22C55E", "#F59E0B", "#8B5CF6", "#EF4444", "#06B6D4",
                         "#EC4899", "#84CC16", "#F97316", "#14B8A6", "#6366F1", "#A78BFA"]

        def hex_to_rgba(hex_color, alpha=0.13):
            h = hex_color.lstrip("#")
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            return f"rgba({r},{g},{b},{alpha})"

        fig = go.Figure()
        for idx, centro in enumerate(centroides):
            color = colores_radar[idx % len(colores_radar)]
            fig.add_trace(go.Scatterpolar(
                r=list(centro) + [centro[0]],
                theta=DIMENSIONES + [DIMENSIONES[0]],
                fill="toself",
                name=f"C{idx}: {etiquetas[idx].replace('Predominantemente ', '')}",
                line=dict(color=color, width=2.5),
                fillcolor=hex_to_rgba(color),
                opacity=0.9,
            ))
        fig.update_layout(
            height=520,
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 6], tickfont=dict(size=10, family="Inter"),
                                gridcolor="#E5E7EB", linecolor="#D1D5DB"),
                angularaxis=dict(tickfont=dict(size=13, family="Inter", color="#374151"),
                                 linecolor="#D1D5DB"),
                bgcolor="#F8FAFC",
            ),
            paper_bgcolor="white",
            margin=dict(l=60, r=60, t=30, b=30),
            legend=dict(font=dict(size=10, family="Inter"), bgcolor="rgba(255,255,255,0.9)",
                        bordercolor="#E5E7EB", borderwidth=1),
            font=dict(family="Inter"),
        )
        st.plotly_chart(fig, width='stretch')
        st.caption("Cada eje va de 0 (nulo) a 6 (máximo) · Área más grande = perfil más marcado en esa dimensión")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown("#### Tabla de Perfiles Promedio por Grupo")
        st.caption("Valores del 0 al 6 en cada dimensión RIASEC · Mayor valor = mayor afinidad con ese perfil vocacional")
        tabla_centroides = pd.DataFrame(centroides, columns=DIMENSIONES)
        tabla_centroides.insert(0, "Clúster", [f"C{i}" for i in range(len(centroides))])
        tabla_centroides.insert(1, "Perfil Vocacional", [etiquetas[i] for i in range(len(centroides))])
        for col in DIMENSIONES:
            tabla_centroides[col] = tabla_centroides[col].round(3)
        st.dataframe(tabla_centroides, width='stretch', hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------
# 6. METADATOS DEL MODELO
# --------------------------------------------------------------------------
elif pestana == "Metadatos del Modelo":
    section_header("Historial de Modelos Entrenados", "Metadatos · Métricas",
                   "Compara diferentes versiones del modelo y consulta sus métricas de rendimiento")

    explain_box(
        "¿Qué son estas métricas?",
        "<b>Calidad de separación (Silhouette):</b> qué tan bien definidos están los grupos · más cercano a 1 = mejor. "
        "<b>BIC / AIC:</b> penalizan la complejidad del modelo · menor valor = modelo más eficiente."
    )

    session = get_session()
    registros = session.query(ModeloEntrenado).order_by(ModeloEntrenado.fecha_entrenamiento.desc()).all()
    session.close()

    if not registros:
        st.info("Aún no hay modelos entrenados. Ve a **Entrenamiento del Modelo** para generar uno.")
    else:
        # Modelo activo card
        activo = next((r for r in registros if r.activo), None)
        if activo:
            st.markdown(f"""
            <div class="panel-card" style="border-left: 4px solid #22C55E;">
                <div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:0.5rem;">
                    <span class="badge-active">MODELO ACTIVO</span>
                    <span style="font-size:0.85rem; color:#374151; font-weight:600;">Modelo #{activo.id} — {activo.algoritmo}</span>
                </div>
                <div style="display:grid; grid-template-columns: repeat(4,1fr); gap:1rem; margin-top:0.75rem;">
                    <div><div style="font-size:0.7rem;color:#6B7280;text-transform:uppercase;">Componentes</div><div style="font-size:1.3rem;font-weight:700;color:#1B2A4A;">{activo.n_componentes}</div></div>
                    <div><div style="font-size:0.7rem;color:#6B7280;text-transform:uppercase;">Covarianza</div><div style="font-size:1.3rem;font-weight:700;color:#1B2A4A;">{activo.covariance_type}</div></div>
                    <div><div style="font-size:0.7rem;color:#6B7280;text-transform:uppercase;">Silhouette</div><div style="font-size:1.3rem;font-weight:700;color:#22C55E;">{round(activo.silhouette_score,4) if activo.silhouette_score else "N/A"}</div></div>
                    <div><div style="font-size:0.7rem;color:#6B7280;text-transform:uppercase;">Registros</div><div style="font-size:1.3rem;font-weight:700;color:#1B2A4A;">{activo.n_registros_entrenamiento}</div></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Tabla histórica
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown("#### Historial Completo de Entrenamientos")
        tabla = pd.DataFrame([{
            "ID": r.id,
            "Algoritmo": r.algoritmo,
            "Grupos": r.n_componentes,
            "Covarianza": r.covariance_type,
            "Estudiantes": r.n_registros_entrenamiento,
            "Calidad (Silhouette)": round(r.silhouette_score, 4) if r.silhouette_score else None,
            "BIC ↓": round(r.bic, 1) if r.bic else None,
            "AIC ↓": round(r.aic, 1) if r.aic else None,
            "Fecha": r.fecha_entrenamiento.strftime("%Y-%m-%d %H:%M") if r.fecha_entrenamiento else None,
            "Estado": "Activo" if r.activo else "Inactivo",
        } for r in registros])
        st.dataframe(tabla, width='stretch', hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Gráfica comparativa de métricas
        if len(registros) > 1:
            st.markdown('<div class="panel-card">', unsafe_allow_html=True)
            st.markdown("#### Comparativa de Métricas entre Modelos")
            df_metricas = pd.DataFrame([{
                "Modelo": f"#{r.id}",
                "Silhouette": r.silhouette_score,
                "BIC": r.bic,
                "AIC": r.aic,
            } for r in registros if r.silhouette_score is not None])
            if not df_metricas.empty:
                tab1, tab2 = st.tabs(["Silhouette Score", "BIC / AIC"])
                with tab1:
                    fig_sil = px.bar(df_metricas, x="Modelo", y="Silhouette",
                                     color="Silhouette", color_continuous_scale=["#BFDBFE", "#1D4ED8"],
                                     labels={"Silhouette": "Silhouette Score"})
                    fig_sil.update_layout(height=280, paper_bgcolor="white", plot_bgcolor="white",
                                          coloraxis_showscale=False, font=dict(family="Inter"),
                                          margin=dict(l=10, r=10, t=10, b=10))
                    st.plotly_chart(fig_sil, width='stretch')
                with tab2:
                    fig_bic = px.line(df_metricas, x="Modelo", y=["BIC", "AIC"],
                                      markers=True, labels={"value": "Score", "variable": "Métrica"},
                                      color_discrete_map={"BIC": "#3B82F6", "AIC": "#22C55E"})
                    fig_bic.update_layout(height=280, paper_bgcolor="white", plot_bgcolor="#F8FAFC",
                                          font=dict(family="Inter"), margin=dict(l=10, r=10, t=10, b=10))
                    st.plotly_chart(fig_bic, width='stretch')
            st.markdown('</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------
# 7. DESCARGAS
# --------------------------------------------------------------------------
elif pestana == "Descargas":
    section_header("Descargas", "Exportación de Datos",
                   "Exporta los datos y resultados del análisis en distintos formatos")

    explain_box(
        "¿Qué puedes descargar?",
        "<b>CSV:</b> todos los datos de los estudiantes (para abrir en Excel). "
        "<b>Reporte TXT:</b> descripción cualitativa de cada grupo vocacional. "
        "<b>JSON:</b> métricas técnicas del modelo (para desarrolladores)."
    )

    #  CSV Datos filtrados 
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown("#### Datos de Estudiantes (CSV)")
    st.caption("Contiene los puntajes RIASEC de todos los estudiantes. Si aplicaste filtros en 'Carga y Visualización', solo se exportan los filtrados.")
    df_filtrado = st.session_state.get("df_filtrado", usuarios_a_dataframe())
    st.markdown(f"<p style='color:#6B7280; font-size:0.85rem;'><b>{len(df_filtrado)}</b> registros disponibles para exportar.</p>", unsafe_allow_html=True)
    csv_buffer = io.StringIO()
    df_filtrado.to_csv(csv_buffer, index=False)
    st.download_button(
        "Descargar CSV de datos filtrados",
        data=csv_buffer.getvalue(),
        file_name="riasec_datos_filtrados.csv",
        mime="text/csv",
        width='content',
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # PDF Datos filtrados (y Reporte Cualitativo)
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown("####  Reporte de Análisis Vocacional (PDF)")
    st.caption("Exporta los datos filtrados a un PDF profesional con el modelo activo y análisis cualitativo.")
    
    # Preparamos la información para el PDF
    modelo, registro = cargar_modelo_activo()
    etiquetas_dict = {}
    fig_radar = None
    
    if modelo is not None and registro is not None:
        etiquetas_dict = etiquetar_clusters(modelo.means_)

    if st.button("Generar Reporte PDF"):
        with st.spinner("Generando PDF (esto puede tardar unos segundos)..."):
            pdf_buffer = crear_reporte_pdf(df_filtrado, modelo, registro, etiquetas_dict, fig_radar)
            st.download_button(
                "📥 Descargar PDF Generado",
                data=pdf_buffer,
                file_name="riasec_reporte_vocacional.pdf",
                mime="application/pdf",
                width='content',
            )
    st.markdown('</div>', unsafe_allow_html=True)

    #  Reporte cualitativo 
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown("#### Reporte Cualitativo de Interpretación (.txt)")
    modelo, registro = cargar_modelo_activo()
    if modelo is not None:
        from src.application.entrenamiento_service import etiquetar_clusters
        etiquetas = etiquetar_clusters(modelo.means_)
        lineas = [
            "=" * 60,
            " REPORTE CUALITATIVO — PERFIL VOCACIONAL RIASEC",
            f" Modelo #{registro.id} | {registro.algoritmo} | {registro.n_componentes} componentes",
            f" Fecha de entrenamiento: {registro.fecha_entrenamiento}",
            f" Silhouette Score: {round(registro.silhouette_score, 4) if registro.silhouette_score else 'N/A'}",
            "=" * 60,
            "",
        ]
        for idx, etiqueta in etiquetas.items():
            centro = modelo.means_[idx]
            lineas.append(f"Clúster {idx}: {etiqueta}")
            lineas.append(f"Perfil promedio [R, I, A, S, E, C]: {[round(v, 3) for v in centro]}")
            dom = sorted(zip(DIMENSIONES, centro), key=lambda x: x[1], reverse=True)
            lineas.append(f"Dimensión dominante: {NOMBRES_DIMENSION[dom[0][0].upper()]} ({dom[0][1]:.2f})")
            lineas.append("")

        texto_reporte = "\n".join(lineas)
        st.text_area("Vista previa del reporte", texto_reporte, height=300)
        st.download_button(
            "Descargar reporte cualitativo (.txt)",
            data=texto_reporte,
            file_name="riasec_reporte_cualitativo.txt",
            mime="text/plain",
        )
    else:
        st.info("Entrena un modelo primero para generar el reporte cualitativo.")
    st.markdown('</div>', unsafe_allow_html=True)

    #  Resumen JSON del modelo activo 
    if modelo is not None:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown("#### Metadatos del Modelo Activo (JSON)")
        from src.application.entrenamiento_service import etiquetar_clusters
        etiquetas = etiquetar_clusters(modelo.means_)
        meta_json = {
            "modelo_id": registro.id,
            "algoritmo": registro.algoritmo,
            "n_componentes": registro.n_componentes,
            "covariance_type": registro.covariance_type,
            "silhouette_score": registro.silhouette_score,
            "bic": registro.bic,
            "aic": registro.aic,
            "n_registros_entrenamiento": registro.n_registros_entrenamiento,
            "fecha_entrenamiento": str(registro.fecha_entrenamiento),
            "etiquetas_cluster": etiquetas,
            "centroides": {f"cluster_{k}": dict(zip(DIMENSIONES, v.tolist())) for k, v in enumerate(modelo.means_)},
        }
        json_str = json.dumps(meta_json, indent=2, ensure_ascii=False)
        st.download_button(
            "Descargar metadatos JSON",
            data=json_str,
            file_name="riasec_modelo_metadatos.json",
            mime="application/json",
        )
        st.markdown('</div>', unsafe_allow_html=True)
