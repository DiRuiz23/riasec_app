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

from db import (
    init_db, get_session, Usuario, Pregunta, OpcionPregunta, Respuesta,
    VectorRiasec, ModeloEntrenado, ResultadoClustering
)
from cuestionario import PREGUNTAS, DIMENSIONES, NOMBRES_DIMENSION, agregar_vector
from estadistica import media, desviacion_estandar, moda, mediana, distribucion_por_categoria, resumen_dimensiones
from entrenamiento import entrenar_modelo, cargar_modelo_activo, obtener_matriz_entrenamiento, proyeccion_pca_2d
# PDF generation imports
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/*  Reset y base  */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/*  Fondo principal  */
.main .block-container {
    background-color: #F4F6F9;
    padding: 1.5rem 2rem 2rem 2rem;
    max-width: 1400px;
}
.stApp {
    background-color: #F4F6F9;
}

/*  Sidebar  */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1B2A4A 0%, #243452 100%) !important;
    border-right: none;
    box-shadow: 4px 0 15px rgba(0,0,0,0.12);
}
[data-testid="stSidebar"] * {
    color: #C8D3E8 !important;
}
[data-testid="stSidebar"] .stRadio label {
    color: #A8B8D8 !important;
    font-size: 0.875rem !important;
    font-weight: 400;
    padding: 0.35rem 0;
    cursor: pointer;
    transition: color 0.2s;
}
[data-testid="stSidebar"] .stRadio label:hover {
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.1) !important;
    margin: 0.75rem 0;
}

/*  Logo sidebar  */
.sidebar-logo {
    text-align: center;
    padding: 1.5rem 1rem 1rem 1rem;
    border-bottom: 1px solid rgba(255,255,255,0.1);
    margin-bottom: 1rem;
}
.sidebar-logo h1 {
    color: #FFFFFF !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    letter-spacing: 4px;
    margin: 0;
}
.sidebar-logo p {
    color: #7B97C7 !important;
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin: 0.2rem 0 0 0;
}

/*  KPI Cards  */
.kpi-card {
    background: #FFFFFF;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    box-shadow: 0 2px 12px rgba(27, 42, 74, 0.08);
    border-left: 4px solid #3B82F6;
    transition: transform 0.2s, box-shadow 0.2s;
    height: 100%;
}
.kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(27, 42, 74, 0.14);
}
.kpi-card.green  { border-left-color: #22C55E; }
.kpi-card.orange { border-left-color: #F59E0B; }
.kpi-card.violet { border-left-color: #8B5CF6; }
.kpi-card.blue   { border-left-color: #3B82F6; }
.kpi-icon {
    font-size: 1.8rem;
    margin-bottom: 0.5rem;
    line-height: 1;
}
.kpi-value {
    font-size: 2rem;
    font-weight: 700;
    color: #1B2A4A;
    line-height: 1.1;
}
.kpi-label {
    font-size: 0.78rem;
    font-weight: 500;
    color: #6B7280;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-top: 0.25rem;
}
.kpi-sub {
    font-size: 0.7rem;
    color: #9CA3AF;
    margin-top: 0.2rem;
}

/*  Títulos de sección  */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 1.5rem;
    padding-bottom: 0.75rem;
    border-bottom: 2px solid #E5E7EB;
}
.section-header h2 {
    font-size: 1.35rem !important;
    font-weight: 700 !important;
    color: #1B2A4A !important;
    margin: 0 !important;
}
.section-badge {
    background: #EEF2FF;
    color: #3B82F6;
    border-radius: 6px;
    padding: 0.15rem 0.6rem;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.5px;
}

/*  Panel/Card contenedor  */
.panel-card {
    background: #FFFFFF;
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 2px 12px rgba(27, 42, 74, 0.07);
    margin-bottom: 1.25rem;
    border: 1px solid #F0F2F6;
}
.panel-card h4 {
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    color: #374151 !important;
    margin-bottom: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/*  Métricas de Streamlit  */
[data-testid="metric-container"] {
    background: #FFFFFF;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    box-shadow: 0 2px 8px rgba(27, 42, 74, 0.07);
    border: 1px solid #F0F2F6;
}
[data-testid="metric-container"] label {
    color: #6B7280 !important;
    font-size: 0.78rem !important;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #1B2A4A !important;
    font-weight: 700 !important;
}

/*  Tablas  */
.stDataFrame {
    border-radius: 10px !important;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(27, 42, 74, 0.07);
    border: 1px solid #E5E7EB !important;
}

/*  Botones  */
.stButton > button {
    background: linear-gradient(135deg, #2563EB, #3B82F6) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    padding: 0.55rem 1.5rem !important;
    transition: all 0.2s !important;
    box-shadow: 0 2px 6px rgba(37, 99, 235, 0.3) !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1D4ED8, #2563EB) !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4) !important;
    transform: translateY(-1px) !important;
}

/*  Inputs y Selects  */
.stSelectbox > div > div,
.stMultiSelect > div > div,
.stNumberInput > div > div > input,
.stTextInput > div > div > input {
    border-radius: 8px !important;
    border-color: #D1D5DB !important;
    font-family: 'Inter', sans-serif !important;
}
.stSelectbox > div > div:focus-within,
.stTextInput > div > div:focus-within {
    border-color: #3B82F6 !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.15) !important;
}

/*  Alertas / Info boxes  */
.stInfo {
    background: #EFF6FF !important;
    border-color: #BFDBFE !important;
    color: #1E40AF !important;
    border-radius: 8px !important;
}
.stSuccess {
    border-radius: 8px !important;
}
.stError {
    border-radius: 8px !important;
}

/*  Spinner  */
.stSpinner > div {
    border-color: #3B82F6 !important;
}

/*  Badge de estado  */
.badge-active {
    display: inline-block;
    background: #D1FAE5;
    color: #065F46;
    border-radius: 20px;
    padding: 0.2rem 0.75rem;
    font-size: 0.72rem;
    font-weight: 600;
}
.badge-inactive {
    display: inline-block;
    background: #F3F4F6;
    color: #6B7280;
    border-radius: 20px;
    padding: 0.2rem 0.75rem;
    font-size: 0.72rem;
    font-weight: 600;
}

/*  Dividers  */
hr {
    border-color: #E5E7EB !important;
    margin: 1.5rem 0 !important;
}

/*  Títulos principales  */
h1 { color: #1B2A4A !important; font-weight: 700 !important; }
h2 { color: #1B2A4A !important; font-weight: 600 !important; }
h3 { color: #374151 !important; font-weight: 600 !important; }

/*  Download button  */
[data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg, #059669, #10B981) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 6px rgba(5, 150, 105, 0.3) !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: linear-gradient(135deg, #047857, #059669) !important;
    box-shadow: 0 4px 12px rgba(5, 150, 105, 0.4) !important;
    transform: translateY(-1px) !important;
}

/*  Plotly charts  */
.js-plotly-plot .plotly .main-svg {
    border-radius: 10px;
}

/*  Ocultar menú hamburguesa y footer de Streamlit  */
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
                "usuario_id": u.id, "sexo": u.sexo, "edad": u.edad,
                "carrera_interes": u.carrera_interes,
                "fecha_registro": u.fecha_registro,
            }
            if u.vector:
                fila.update({"R": u.vector.r, "I": u.vector.i, "A": u.vector.a,
                             "S": u.vector.s, "E": u.vector.e, "C": u.vector.c})
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


def kpi_card(value, label, sub="", color="blue"):
    return f"""
    <div class="kpi-card {color}">
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
        {"<div class='kpi-sub'>" + sub + "</div>" if sub else ""}
    </div>
    """


def section_header(title, badge=""):
    badge_html = f'<span class="section-badge">{badge}</span>' if badge else ""
    st.markdown(f"""
    <div class="section-header">
        <h2>{title}</h2>
        {badge_html}
    </div>
    """, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Barra lateral
# --------------------------------------------------------------------------
st.sidebar.markdown("""
<div class="sidebar-logo">
    <h1>RIASEC</h1>
    <p>Análisis Vocacional · Unidad IV</p>
</div>
""", unsafe_allow_html=True)

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
<div style="padding: 0.5rem 0; color: #7B97C7; font-size: 0.7rem; text-align: center; letter-spacing: 0.5px;">
    RIASEC · Gaussian Mixture Model<br>
    <span style="color: #4A6B9A;">Análisis No Supervisado</span>
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
    sexo, edad, carrera_interes, R, I, A, S, E, C.
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
            "edad": None,
            "carrera_interes": None,
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
    section_header("Dashboard", "General")

    df = usuarios_a_dataframe()
    modelo, registro = cargar_modelo_activo()

    #  KPI Cards 
    c1, c2, c3, c4 = st.columns(4)

    total_usuarios = len(df)
    with c1:
        st.markdown(kpi_card(str(total_usuarios), "Usuarios Registrados",
            sub="Total en base de datos", color="blue"
        ), unsafe_allow_html=True)

    if modelo is not None:
        n_componentes_activo = registro.n_componentes
        silhouette_val = registro.silhouette_score
        algo = registro.algoritmo
        n_registros_ent = registro.n_registros_entrenamiento

        with c2:
            st.markdown(kpi_card(f"{n_componentes_activo}", "Clústeres Activos",
                sub=f"{algo}", color="violet"
            ), unsafe_allow_html=True)

        with c3:
            sil_str = f"{silhouette_val:.4f}" if silhouette_val is not None else "N/A"
            sil_sub = "Bueno >0.5" if silhouette_val and silhouette_val > 0.5 else "Aceptable >0.3" if silhouette_val and silhouette_val > 0.3 else "—"
            st.markdown(kpi_card(sil_str, "Silhouette Score",
                sub=sil_sub, color="green"
            ), unsafe_allow_html=True)

        with c4:
            st.markdown(kpi_card(str(n_registros_ent), "Registros Entrenados",
                sub=f"Modelo #{registro.id}", color="orange"
            ), unsafe_allow_html=True)
    else:
        with c2:
            st.markdown(kpi_card("—", "Clústeres Activos", sub="Sin modelo entrenado", color="violet"), unsafe_allow_html=True)
        with c3:
            st.markdown(kpi_card("—", "Silhouette Score", sub="Entrenar modelo primero", color="green"), unsafe_allow_html=True)
        with c4:
            st.markdown(kpi_card("—", "Registros Entrenados", sub="—", color="orange"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    #  Gráficas principales 
    if not df.empty and modelo is not None:
        col_left, col_right = st.columns(2)

        # PCA Scatter
        with col_left:
            st.markdown('<div class="panel-card"><h4> Proyección PCA 2D — Distribución de Perfiles</h4>', unsafe_allow_html=True)
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
            st.caption(f"Varianza explicada: {round(sum(varianza)*100, 1)}%")
            st.markdown('</div>', unsafe_allow_html=True)

        # Radar de Centroides
        with col_right:
            st.markdown('<div class="panel-card"><h4> Radar de Clústeres — Perfil de Centroides</h4>', unsafe_allow_html=True)
            from entrenamiento import etiquetar_clusters
            centroides = modelo.means_
            etiquetas = etiquetar_clusters(centroides)
            fig_radar = go.Figure()
            colores_radar = ["#3B82F6", "#22C55E", "#F59E0B", "#8B5CF6", "#EF4444", "#06B6D4"]
            for idx, centro in enumerate(centroides):
                color = colores_radar[idx % len(colores_radar)]
                fig_radar.add_trace(go.Scatterpolar(
                    r=list(centro) + [centro[0]],
                    theta=DIMENSIONES + [DIMENSIONES[0]],
                    fill="toself",
                    name=f"C{idx}: {etiquetas[idx].replace('Predominantemente ', '')}",
                    line=dict(color=color, width=2),
                    fillcolor=color.replace("#", "rgba(").replace("F6", "F6,0.15)") if "F6" in color else color + "26",
                ))
            fig_radar.update_layout(
                height=360,
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 6], tickfont=dict(size=9)),
                    angularaxis=dict(tickfont=dict(size=11, family="Inter", color="#374151")),
                    bgcolor="#F8FAFC",
                ),
                margin=dict(l=40, r=40, t=20, b=20),
                paper_bgcolor="white",
                legend=dict(font=dict(size=9), x=1.02, y=1),
                font=dict(family="Inter"),
            )
            st.plotly_chart(fig_radar, width='stretch')
            st.markdown('</div>', unsafe_allow_html=True)

    elif df.empty:
        st.info(" No hay datos en la base de datos. Carga un CSV en la sección **Carga y Visualización** para comenzar.")
    else:
        st.info(" Hay datos disponibles. Ve a **Entrenamiento del Modelo** para generar el modelo de clustering.")

    st.markdown("<br>", unsafe_allow_html=True)

    #  Fila inferior: distribuciones + últimos registros 
    if not df.empty:
        col_a, col_b, col_c = st.columns([1, 1, 2])

        with col_a:
            st.markdown('<div class="panel-card"><h4> Distribución por Sexo</h4>', unsafe_allow_html=True)
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
            st.markdown('<div class="panel-card"><h4> Top Carreras de Interés</h4>', unsafe_allow_html=True)
            if "carrera_interes" in df.columns:
                top_carreras = (
                    df["carrera_interes"].dropna()
                    .value_counts()
                    .head(6)
                    .reset_index()
                )
                top_carreras.columns = ["carrera", "count"]
                if not top_carreras.empty:
                    fig_bar = px.bar(
                        top_carreras, x="count", y="carrera",
                        orientation="h",
                        color="count",
                        color_continuous_scale=["#BFDBFE", "#2563EB"],
                        labels={"count": "", "carrera": ""},
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
                    st.caption("Sin datos de carrera disponibles.")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_c:
            st.markdown('<div class="panel-card"><h4> Últimos Registros</h4>', unsafe_allow_html=True)
            cols_mostrar = [c for c in ["usuario_id", "sexo", "edad", "carrera_interes", "fecha_registro"] if c in df.columns]
            ultimos = df.sort_values("fecha_registro", ascending=False).head(8)[cols_mostrar]
            if "fecha_registro" in ultimos.columns:
                ultimos["fecha_registro"] = pd.to_datetime(ultimos["fecha_registro"]).dt.strftime("%Y-%m-%d")
            st.dataframe(ultimos, width='stretch', hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)



# --------------------------------------------------------------------------
# 1. CARGA Y VISUALIZACIÓN
# --------------------------------------------------------------------------
elif pestana == "Carga y Visualización":
    section_header("Carga de Datos y Visualización", "Importación · Filtros")

    #  Subir CSV 
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown("####  Subir Dataset Externo (CSV)")
    st.caption(
        "Formatos soportados: **CSV estándar** con columnas `sexo`, `edad`, `R`, `I`, `A`, `S`, `E`, `C` "
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
                "📋 **Formato Google Forms detectado** — Las respuestas de texto se convierten "
                "automáticamente a puntajes RIASEC (0–6). "
                "`edad` y `carrera_interes` no están disponibles en este formulario."
            )
            df_nuevo = parsear_csv_google_forms(df_raw)
            st.success(f"✅ {len(df_nuevo)} registros convertidos correctamente.")
            with st.expander("👁 Vista previa de los datos convertidos (primeros 5)", expanded=True):
                st.dataframe(df_nuevo.head(5), use_container_width=True, hide_index=True)

        elif columnas_estandar.issubset(set(df_raw.columns)):
            # ── Formato estándar ──────────────────────────────────────────
            df_nuevo = df_raw
            st.success(f"✅ Archivo válido — {len(df_nuevo)} registros detectados.")
            with st.expander("👁 Vista previa (primeros 5)", expanded=True):
                st.dataframe(df_nuevo.head(5), use_container_width=True, hide_index=True)

        else:
            # ── Error: formato desconocido ────────────────────────────────
            st.error(
                "❌ Formato de archivo no reconocido.\n\n"
                "**Opciones válidas:**\n"
                "- CSV estándar con columnas: `sexo`, `edad`, `R`, `I`, `A`, `S`, `E`, `C`\n"
                "- CSV exportado directamente de Google Forms (detección automática)"
            )
            with st.expander("🔍 Columnas encontradas en el archivo"):
                st.write(list(df_raw.columns))
            df_nuevo = None

        if 'df_nuevo' in dir() and df_nuevo is not None:
            if st.button("💾 Confirmar e insertar en la base de datos", type="primary"):
                session = get_session()
                try:
                    insertados = 0
                    for _, fila in df_nuevo.iterrows():
                        edad_val = fila.get("edad")
                        edad_int = int(edad_val) if pd.notna(edad_val) and edad_val is not None else None
                        usuario = Usuario(
                            sexo=fila.get("sexo"),
                            edad=edad_int,
                            carrera_interes=fila.get("carrera_interes"),
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
                    st.success(f"✅ Se insertaron **{insertados}** registros nuevos correctamente.")
                    st.balloons()
                except Exception as ex:
                    session.rollback()
                    st.error(f"❌ Error al insertar: {ex}")
                finally:
                    session.close()
    st.markdown('</div>', unsafe_allow_html=True)

    #  Tabla y filtros 
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown("####  Explorador de Datos Actuales")
    df = usuarios_a_dataframe()

    if df.empty:
        st.info(" Aún no hay registros. Carga un CSV o ejecuta `seed.py` desde consola.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            filtro_sexo = st.multiselect("Filtrar por sexo", options=df["sexo"].dropna().unique().tolist())

        # Slider de edad solo si hay registros con edad definida
        edad_valida = df["edad"].dropna()
        with col2:
            if not edad_valida.empty:
                rango_edad = st.slider(
                    "Rango de edad",
                    int(edad_valida.min()), int(edad_valida.max()),
                    (int(edad_valida.min()), int(edad_valida.max()))
                )
            else:
                rango_edad = None
                st.caption("⚠️ Sin datos de edad disponibles")

        with col3:
            fecha_min = df["fecha_registro"].min()
            fecha_max = df["fecha_registro"].max()
            rango_fecha = st.date_input("Rango de fecha de registro", (fecha_min.date(), fecha_max.date()))

        df_filtrado = df.copy()
        if filtro_sexo:
            df_filtrado = df_filtrado[df_filtrado["sexo"].isin(filtro_sexo)]

        # Filtro de edad: incluir siempre registros sin edad (NaN)
        if rango_edad is not None:
            mascara_edad = (
                df_filtrado["edad"].isna() |
                ((df_filtrado["edad"] >= rango_edad[0]) & (df_filtrado["edad"] <= rango_edad[1]))
            )
            df_filtrado = df_filtrado[mascara_edad]

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
            st.markdown("####  Distribución Media de Dimensiones RIASEC (datos filtrados)")
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
    section_header("Estadística Descriptiva", "Algoritmos Propios")

    df = usuarios_a_dataframe()
    if df.empty:
        st.info(" No hay datos suficientes. Carga información primero.")
    else:
        vectores = df[["R", "I", "A", "S", "E", "C"]].rename(columns=str.lower).to_dict("records")
        resumen = resumen_dimensiones(vectores)

        #  Tabla resumen 
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown("####  Resumen Estadístico por Dimensión RIASEC")
        st.caption("Calculado con algoritmos propios: media, mediana, moda, desviación estándar, mínimo y máximo.")
        df_resumen = pd.DataFrame(resumen).T
        df_resumen.index.name = "Dimensión"
        st.dataframe(df_resumen.style.format({"media": "{:.2f}", "mediana": "{:.2f}",
                                               "desviacion_estandar": "{:.2f}"}),
                     width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)

        #  Gráficas de cajas / distribución 
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown("####  Dispersión por Dimensión (Box Plot)")
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
        st.markdown("####  Distribución por Campo Categórico")
        campo = st.selectbox("Selecciona campo", ["sexo", "carrera_interes"])
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
    section_header("Entrenamiento del Modelo", "Gaussian Mixture Model")

    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown("####  Configuración de Hiperparámetros")
    st.caption("El modelo agrupa los vectores `[R,I,A,S,E,C]` de forma no supervisada y luego etiqueta cada clúster según su centroide dominante.")

    col1, col2 = st.columns(2)
    with col1:
        n_componentes = st.number_input(
            "Número de componentes (clústeres)",
            min_value=2, max_value=12, value=6,
            help="Cuántos grupos vocacionales distintos buscará el modelo."
        )
    with col2:
        covariance_type = st.selectbox(
            "Tipo de covarianza",
            ["full", "tied", "diag", "spherical"],
            help="`full`: cada clúster tiene su propia matriz de covarianza completa (más flexible)."
        )

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        entrenar_btn = st.button(" Entrenar Modelo", width='stretch')
    with col_info:
        st.markdown("""
        <div style="background:#EFF6FF; border:1px solid #BFDBFE; border-radius:8px; padding:0.6rem 1rem; font-size:0.82rem; color:#1E40AF;">
            <b>ℹ Nota:</b> Al entrenar, el modelo anterior quedará inactivo. Se necesitan al menos <b>n_componentes</b> registros en la base de datos.
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if entrenar_btn:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        with st.spinner(" Entrenando el modelo Gaussian Mixture..."):
            try:
                resultado = entrenar_modelo(n_componentes=n_componentes, covariance_type=covariance_type)

                st.success(f" Modelo **#{resultado['modelo_id']}** entrenado exitosamente con **{resultado['n_registros']}** registros.")

                m1, m2, m3 = st.columns(3)
                m1.metric("Silhouette Score", f"{resultado['silhouette']:.4f}" if resultado['silhouette'] else "N/A")
                m2.metric("BIC", f"{resultado['bic']:.2f}")
                m3.metric("AIC", f"{resultado['aic']:.2f}")

                st.markdown("---")
                st.markdown("####  Etiquetas Asignadas por Clúster")
                etiquetas_df = pd.DataFrame([
                    {"Clúster": f"Clúster {k}", "Etiqueta Vocacional": v}
                    for k, v in resultado["etiquetas"].items()
                ])
                st.dataframe(etiquetas_df, width='stretch', hide_index=True)
            except ValueError as e:
                st.error(f" {str(e)}")
        st.markdown('</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------
# 4. RESULTADOS
# --------------------------------------------------------------------------
elif pestana == "Resultados":
    section_header("Resultados del Clustering", "PCA · Asignaciones")

    modelo, registro = cargar_modelo_activo()
    if modelo is None:
        st.info(" Aún no hay ningún modelo entrenado. Ve a **Entrenamiento del Modelo** para generar uno.")
    else:
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
        r1.metric("Total usuarios asignados", len(df_resultados))
        r2.metric("Clústeres identificados", df_resultados["cluster_id"].nunique() if not df_resultados.empty else "—")
        r3.metric("Varianza PCA explicada", f"{round(sum(varianza)*100, 1)}%")

        st.markdown("<br>", unsafe_allow_html=True)

        # Scatter PCA
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown("####  Clústeres Proyectados (PCA 6D → 2D)")
        fig = px.scatter(
            df_plot, x="PCA_1", y="PCA_2", color="etiqueta_riasec",
            hover_data=["usuario_id"],
            color_discrete_sequence=px.colors.qualitative.Set2,
            labels={"PCA_1": "Componente Principal 1", "PCA_2": "Componente Principal 2", "etiqueta_riasec": "Perfil Vocacional"},
        )
        fig.update_traces(marker=dict(size=9, opacity=0.82, line=dict(width=0.5, color="white")))
        fig.update_layout(
            height=450,
            paper_bgcolor="white", plot_bgcolor="#F8FAFC",
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(font=dict(size=10, family="Inter"), orientation="v", x=1.01, y=1,
                        bgcolor="rgba(255,255,255,0.8)", bordercolor="#E5E7EB", borderwidth=1),
            font=dict(family="Inter"),
        )
        st.plotly_chart(fig, width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)

        # Tabla de asignaciones
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown("####  Tabla de Asignaciones por Usuario")
        df_tabla = df_resultados[["usuario_id", "cluster_id", "etiqueta_riasec"]].copy()
        st.dataframe(df_tabla, width='stretch', hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------
# 5. COMPARATIVA DE CLÚSTERES
# --------------------------------------------------------------------------
elif pestana == "Comparativa de Clústeres":
    section_header("Comparativa entre Clústeres", "Perfil Radar")

    modelo, registro = cargar_modelo_activo()
    if modelo is None:
        st.info(" Aún no hay ningún modelo entrenado.")
    else:
        from entrenamiento import etiquetar_clusters
        centroides = modelo.means_
        etiquetas = etiquetar_clusters(centroides)

        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown("####  Gráfico Radar — Medias por Dimensión por Clúster")

        colores_radar = ["#3B82F6", "#22C55E", "#F59E0B", "#8B5CF6", "#EF4444", "#06B6D4",
                         "#EC4899", "#84CC16", "#F97316", "#14B8A6", "#6366F1", "#A78BFA"]

        fig = go.Figure()
        for idx, centro in enumerate(centroides):
            color = colores_radar[idx % len(colores_radar)]
            fig.add_trace(go.Scatterpolar(
                r=list(centro) + [centro[0]],
                theta=DIMENSIONES + [DIMENSIONES[0]],
                fill="toself",
                name=f"C{idx}: {etiquetas[idx].replace('Predominantemente ', '')}",
                line=dict(color=color, width=2.5),
                fillcolor=color + "22",
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
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown("####  Tabla de Centroides por Clúster")
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
    section_header("Historial de Modelos Entrenados", "Metadatos · Métricas")

    session = get_session()
    registros = session.query(ModeloEntrenado).order_by(ModeloEntrenado.fecha_entrenamiento.desc()).all()
    session.close()

    if not registros:
        st.info(" Aún no hay modelos entrenados. Ve a **Entrenamiento del Modelo** para generar uno.")
    else:
        # Modelo activo card
        activo = next((r for r in registros if r.activo), None)
        if activo:
            st.markdown(f"""
            <div class="panel-card" style="border-left: 4px solid #22C55E;">
                <div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:0.5rem;">
                    <span class="badge-active"> MODELO ACTIVO</span>
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
        st.markdown("####  Historial Completo de Entrenamientos")
        tabla = pd.DataFrame([{
            "ID": r.id,
            "Algoritmo": r.algoritmo,
            "Componentes": r.n_componentes,
            "Covarianza": r.covariance_type,
            "Registros": r.n_registros_entrenamiento,
            "Silhouette": round(r.silhouette_score, 4) if r.silhouette_score else None,
            "BIC": round(r.bic, 2) if r.bic else None,
            "AIC": round(r.aic, 2) if r.aic else None,
            "Fecha": r.fecha_entrenamiento.strftime("%Y-%m-%d %H:%M") if r.fecha_entrenamiento else None,
            "Activo": " Activo" if r.activo else " Inactivo",
        } for r in registros])
        st.dataframe(tabla, width='stretch', hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Gráfica comparativa de métricas
        if len(registros) > 1:
            st.markdown('<div class="panel-card">', unsafe_allow_html=True)
            st.markdown("####  Comparativa de Métricas entre Modelos")
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
    section_header("Descargas", "Exportación de Datos")

    #  CSV Datos filtrados 
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown("####  Dataset de Usuarios (CSV)")
    st.caption("Se exportan los datos tal como están filtrados en la sección **Carga y Visualización**. Si no usaste filtros, se incluyen todos los registros.")
    df_filtrado = st.session_state.get("df_filtrado", usuarios_a_dataframe())
    st.markdown(f"<p style='color:#6B7280; font-size:0.85rem;'><b>{len(df_filtrado)}</b> registros disponibles para exportar.</p>", unsafe_allow_html=True)
    csv_buffer = io.StringIO()
    df_filtrado.to_csv(csv_buffer, index=False)
    st.download_button(
        " Descargar CSV de datos filtrados",
        data=csv_buffer.getvalue(),
        file_name="riasec_datos_filtrados.csv",
        mime="text/csv",
        width='content',
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # PDF Datos filtrados
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown("####  Dataset de Usuarios (PDF)")
    st.caption("Exporta los datos filtrados a PDF. Se incluye una tabla con los registros.")
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()
    title = Paragraph("Dataset de Usuarios", styles["Title"])
    elements.append(title)
    data = [df_filtrado.columns.tolist()] + df_filtrado.values.tolist()
    table = Table(data, repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold')
    ]))
    elements.append(table)
    doc.build(elements)
    pdf_buffer.seek(0)
    st.download_button(
        " Descargar PDF de datos filtrados",
        data=pdf_buffer,
        file_name="riasec_datos_filtrados.pdf",
        mime="application/pdf",
        width='content',
    )
    st.markdown('</div>', unsafe_allow_html=True)

    #  Reporte cualitativo 
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown("####  Reporte Cualitativo de Interpretación (.txt)")
    modelo, registro = cargar_modelo_activo()
    if modelo is not None:
        from entrenamiento import etiquetar_clusters
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
            " Descargar reporte cualitativo (.txt)",
            data=texto_reporte,
            file_name="riasec_reporte_cualitativo.txt",
            mime="text/plain",
        )
    else:
        st.info(" Entrena un modelo primero para generar el reporte cualitativo.")
    st.markdown('</div>', unsafe_allow_html=True)

    #  Resumen JSON del modelo activo 
    if modelo is not None:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown("####  Metadatos del Modelo Activo (JSON)")
        from entrenamiento import etiquetar_clusters
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
            " Descargar metadatos JSON",
            data=json_str,
            file_name="riasec_modelo_metadatos.json",
            mime="application/json",
        )
        st.markdown('</div>', unsafe_allow_html=True)
