"""
streamlit_ui.py
Interfaz Streamlit del Perfil de Personalidad Vocacional RIASEC.
- Gestión de datos en memoria (no almacena respuestas en BD).
- Algoritmo GMM con validación BIC/AIC paralela.
- Pestaña "Mi Perfil" con cuestionario interactivo.
- Visualizaciones avanzadas: radar, PCA 2D, curva BIC.
- Generación de reportes PDF completos.
- Exportación de datos a CSV.
"""

import json
import os
import datetime
import base64
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.infrastructure.database.db import init_db, get_session, ModeloEntrenado
from src.application.cuestionario_service import (
    NOMBRES_DIMENSION, PREGUNTAS, CARRERAS_RIASEC,
    agregar_vector, validar_mapeo_csv, obtener_carreras_para_perfil,
)
from src.application.clustering_service import (
    encontrar_k_optimo,
    ejecutar_clustering_en_memoria,
    cargar_y_aplicar_modelo,
)
from src.application.report_generator import generar_reporte_pdf
from src.domain.riasec_profile import (
    PerfilRIASEC, DESCRIPCIONES_DIMENSION, DIMENSIONES_ORDEN
)

# ──────────────────────────────────────────────────────────────────────────────
# Configuración de página
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RIASEC Analytics",
    page_icon=":material/analytics:",
    layout="wide",
    initial_sidebar_state="expanded",
)
init_db()

# ──────────────────────────────────────────────────────────────────────────────
# Estilos globales
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0F172A 0%, #1E3A8A 100%);
}
[data-testid="stSidebar"] * {
    color: #E2E8F0 !important;
}
[data-testid="stSidebar"] .stRadio label {
    color: #CBD5E1 !important;
    font-size: 0.92rem;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
    border: 1px solid #BFDBFE;
    border-left: 4px solid #3B82F6;
    padding: 1.2rem 1.4rem;
    border-radius: 12px;
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(59,130,246,0.15);
}
.metric-card .value {
    margin: 0;
    font-size: 2rem;
    font-weight: 700;
    color: #1E3A8A;
    letter-spacing: -0.5px;
}
.metric-card .label {
    margin: 0;
    font-size: 0.75rem;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 500;
    margin-top: 4px;
}

/* Stepper */
.step-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 12px;
    border-radius: 8px;
    margin-bottom: 4px;
}
.step-done   { background: rgba(16,185,129,0.15); }
.step-active { background: rgba(59,130,246,0.25); }
.step-locked { background: rgba(255,255,255,0.05); opacity: 0.6; }
.step-icon   { font-size: 1rem; width: 24px; text-align: center; }
.step-text   { font-size: 0.82rem; font-weight: 500; }

/* Section header */
.section-header {
    background: linear-gradient(90deg, #1E3A8A, #2563EB);
    color: white;
    padding: 1rem 1.5rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
}
.section-header h2 { margin: 0; font-size: 1.3rem; font-weight: 600; }
.section-header p  { margin: 0; font-size: 0.85rem; opacity: 0.8; margin-top: 4px; }

/* Alert quality */
.alert-warning {
    background: #FEF9C3;
    border-left: 4px solid #EAB308;
    padding: 10px 14px;
    border-radius: 6px;
    font-size: 0.88rem;
    margin-bottom: 8px;
}

/* Profile card */
.profile-card {
    background: linear-gradient(135deg, #1E3A8A, #2563EB);
    color: white;
    padding: 2rem;
    border-radius: 16px;
    text-align: center;
    margin-bottom: 1rem;
}
.profile-card .dim-name {
    font-size: 2.5rem;
    font-weight: 800;
    letter-spacing: -1px;
}
.profile-card .dim-label {
    font-size: 1.1rem;
    opacity: 0.9;
    margin-top: 4px;
}
.profile-card .dim-desc {
    font-size: 0.88rem;
    opacity: 0.8;
    margin-top: 12px;
    line-height: 1.5;
}

/* Career pills */
.career-pill {
    display: inline-block;
    background: #EFF6FF;
    color: #1E40AF;
    border: 1px solid #BFDBFE;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.82rem;
    margin: 3px;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Estado inicial
# ──────────────────────────────────────────────────────────────────────────────
if "df_datos" not in st.session_state:
    st.session_state["df_datos"] = pd.DataFrame(
        columns=["id", "sexo", "fecha", "r", "i", "a", "s", "e", "c", "dominante"]
    )
    st.session_state["id_counter"] = 1

if "modelo_info" not in st.session_state:
    st.session_state["modelo_info"] = None

if "curva_bic" not in st.session_state:
    st.session_state["curva_bic"] = None


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def calcular_dominante(row) -> str:
    dims = {d: row[d] for d in ["r", "i", "a", "s", "e", "c"]}
    max_val = max(dims.values())
    ganadoras = [d for d, v in dims.items() if v == max_val]
    if len(ganadoras) == 1:
        return ganadoras[0].upper()
    # Empate: combinación de las dos primeras en orden RIASEC
    orden = ["r", "i", "a", "s", "e", "c"]
    empatadas_ordenadas = [d for d in orden if d in ganadoras]
    return "-".join(d.upper() for d in empatadas_ordenadas[:2])


def verificar_calidad_datos(df: pd.DataFrame) -> list[str]:
    """Detecta problemas de calidad en el dataset."""
    alertas = []
    dims = ["r", "i", "a", "s", "e", "c"]
    # Registros con todo en 0
    todos_cero = (df[dims] == 0).all(axis=1).sum()
    if todos_cero > 0:
        alertas.append(f":material/warning: **{todos_cero}** registro(s) con todas las dimensiones en 0.")
    # Posibles duplicados
    dup = df[dims].duplicated().sum()
    if dup > 0:
        alertas.append(f":material/warning: **{dup}** registro(s) con puntajes RIASEC idénticos a otro(s) (posibles duplicados).")
    # Distribución muy sesgada
    for dim in dims:
        pct_max = (df[dim] == 6).mean()
        if pct_max > 0.6:
            alertas.append(f":material/warning: Dimensión **{dim.upper()}**: el {pct_max:.0%} de registros tiene puntaje máximo (6). Distribución muy sesgada.")
    return alertas


def crear_radar_plotly(df: pd.DataFrame, titulo: str = "Perfil RIASEC") -> go.Figure:
    """Crea un gráfico radar con Plotly, con comparativa por sexo si aplica."""
    dims_lower = ["r", "i", "a", "s", "e", "c"]
    labels_full = ["Realista", "Investigador", "Artístico", "Social", "Emprendedor", "Convencional"]

    fig = go.Figure()

    colores_sexo = {"M": "#1E3A8A", "F": "#EC4899", "Otro": "#10B981"}

    if "sexo" in df.columns and df["sexo"].nunique() > 1:
        for sexo in sorted(df["sexo"].unique()):
            sub = df[df["sexo"] == sexo]
            valores = sub[dims_lower].mean().tolist()
            valores_closed = valores + [valores[0]]
            labels_closed = labels_full + [labels_full[0]]
            color = colores_sexo.get(sexo, "#6366F1")
            nombre_sexo = {"M": "Masculino", "F": "Femenino"}.get(sexo, sexo)
            fig.add_trace(go.Scatterpolar(
                r=valores_closed,
                theta=labels_closed,
                fill="toself",
                fillcolor=f"rgba({int(color[1:3], 16)},{int(color[3:5], 16)},{int(color[5:7], 16)},0.1)" if color.startswith("#") else color,
                line=dict(color=color, width=2.5),
                name=nombre_sexo,
            ))
    else:
        valores = df[dims_lower].mean().tolist()
        valores_closed = valores + [valores[0]]
        labels_closed = labels_full + [labels_full[0]]
        fig.add_trace(go.Scatterpolar(
            r=valores_closed,
            theta=labels_closed,
            fill="toself",
            fillcolor="rgba(30,58,138,0.12)",
            line=dict(color="#1E3A8A", width=3),
            name="Promedio",
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 6], tickfont=dict(size=9)),
            angularaxis=dict(tickfont=dict(size=11)),
        ),
        showlegend=True,
        title=dict(text=titulo, font=dict(size=14, color="#1E3A8A"), x=0.5),
        margin=dict(t=60, b=20, l=30, r=30),
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR con stepper de flujo
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔷 RIASEC Analytics")
    st.markdown("*Análisis de Perfiles Vocacionales*")
    st.markdown("---")

    tiene_datos = not st.session_state["df_datos"].empty
    tiene_filtro = "df_filtrado" in st.session_state and not st.session_state.get("df_filtrado", pd.DataFrame()).empty
    tiene_modelo = st.session_state["modelo_info"] is not None
    tiene_resultados = "df_filtrado_clusterizado" in st.session_state

    def paso(icono_done, icono_lock, texto, done, active):
        if done:
            cls, icono = "step-done", "✅"
        elif active:
            cls, icono = "step-active", "🔄"
        else:
            cls, icono = "step-locked", icono_lock
        st.markdown(
            f'<div class="step-item {cls}">'
            f'<span class="step-icon">{icono}</span>'
            f'<span class="step-text">{texto}</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown("**Flujo de Trabajo**")
    paso("✓", "1", "Cargar Datos", tiene_datos, not tiene_datos)
    paso("✓", "2", "Filtrar en Dashboard", tiene_filtro, tiene_datos and not tiene_filtro)
    paso("✓", "3", "Ejecutar GMM", tiene_modelo, tiene_filtro and not tiene_modelo)
    paso("✓", "4", "Ver Resultados / PDF", tiene_resultados, tiene_modelo and not tiene_resultados)

    st.markdown("---")

    pestana = st.radio("Navegación", [
        ":material/home: Mi Perfil",
        ":material/dashboard: Dashboard",
        ":material/folder: Gestión de Datos",
        ":material/science: Algoritmo GMM",
        ":material/description: Resultados y Reportes",
        ":material/history: Historial de Modelos",
    ])

# ──────────────────────────────────────────────────────────────────────────────
# 0. MI PERFIL — Cuestionario interactivo individual
# ──────────────────────────────────────────────────────────────────────────────
if pestana == ":material/home: Mi Perfil":
    st.markdown("""
    <div class="section-header">
      <h2>Mi Perfil Vocacional RIASEC</h2>
      <p>Responde las 18 preguntas para descubrir tu perfil vocacional y las carreras más afines a ti.</p>
    </div>""", unsafe_allow_html=True)

    preguntas_dim = [(n, d, t, o) for n, d, t, o in PREGUNTAS if d != "DEMOGRAFICA"]
    dimensiones_agrupadas = {d: [(n, t, o) for n, dim, t, o in preguntas_dim if dim == d]
                              for d in ["R", "I", "A", "S", "E", "C"]}

    with st.form("form_mi_perfil"):
        col_sx, _ = st.columns([1, 3])
        sexo_perfil = col_sx.selectbox("Tu sexo", ["M", "F", "Otro"], key="sexo_mp")

        respuestas_usuario: dict[int, int] = {}

        for dim, nombre in NOMBRES_DIMENSION.items():
            color_dim = {
                "R": "#DC2626", "I": "#2563EB", "A": "#7C3AED",
                "S": "#059669", "E": "#D97706", "C": "#0891B2"
            }.get(dim, "#64748B")
            st.markdown(f"### <span style='color:{color_dim}'>● {nombre} ({dim})</span>", unsafe_allow_html=True)
            pregs = dimensiones_agrupadas.get(dim, [])
            for num, texto, opciones in pregs:
                textos_opciones = [o[0] for o in opciones]
                pesos_opciones = [o[1] for o in opciones]
                respuesta = st.radio(
                    texto,
                    options=textos_opciones,
                    index=1,
                    key=f"mp_{num}",
                    horizontal=True,
                )
                idx_resp = textos_opciones.index(respuesta)
                respuestas_usuario[num] = pesos_opciones[idx_resp]

        submitted = st.form_submit_button(":material/search: Calcular mi perfil", type="primary", use_container_width=True)

    if submitted:
        vector = agregar_vector(respuestas_usuario)
        try:
            perfil = PerfilRIASEC.desde_dict(vector, sexo=sexo_perfil)
        except ValueError as e:
            st.error(str(e))
            st.stop()

        st.markdown("---")
        dominante = perfil.dimension_dominante()
        nombre_dom = perfil.nombre_dominante()
        desc_dom = perfil.descripcion_dominante()

        # Tarjeta de perfil dominante
        st.markdown(f"""
        <div class="profile-card">
          <div class="dim-name">{dominante}</div>
          <div class="dim-label">Perfil Dominante: {nombre_dom}</div>
          <div class="dim-desc">{desc_dom}</div>
        </div>
        """, unsafe_allow_html=True)

        if perfil.es_perfil_empatado():
            st.info(":material/shuffle: Tu perfil muestra un empate entre varias dimensiones. Considera las carreras de todas las dimensiones empatadas.")

        col_radar, col_ranking = st.columns([1, 1])

        with col_radar:
            # Radar individual
            labels_full = ["Realista", "Investigador", "Artístico", "Social", "Emprendedor", "Convencional"]
            valores = [vector[d] for d in ["R", "I", "A", "S", "E", "C"]]
            fig_personal = go.Figure(go.Scatterpolar(
                r=valores + [valores[0]],
                theta=labels_full + [labels_full[0]],
                fill="toself",
                fillcolor="rgba(30,58,138,0.15)",
                line=dict(color="#1E3A8A", width=3),
                name="Mi perfil",
            ))
            fig_personal.update_layout(
                polar=dict(radialaxis=dict(range=[0, 6], tickfont=dict(size=9))),
                showlegend=False,
                title=dict(text="Mi Radar RIASEC", font=dict(size=13, color="#1E3A8A"), x=0.5),
                height=360,
                margin=dict(t=50, b=10, l=20, r=20),
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_personal, use_container_width=True)

        with col_ranking:
            st.subheader(":material/bar_chart: Tus puntajes")
            ranking = perfil.ranking_dimensiones()
            for i, (dim, score) in enumerate(ranking):
                nombre = NOMBRES_DIMENSION[dim]
                pct = score / 6
                emoji = ["1.", "2.", "3.", "4.", "5.", "6."][i]
                st.markdown(f"**{emoji} {nombre} ({dim}):** {score}/6")
                st.progress(pct)

        # Carreras sugeridas
        st.subheader(":material/school: Carreras Sugeridas para tu Perfil")
        carreras = obtener_carreras_para_perfil(dominante)
        pills_html = " ".join(f'<span class="career-pill">{c}</span>' for c in carreras)
        st.markdown(pills_html, unsafe_allow_html=True)

        # Descripción de todas las dimensiones
        with st.expander(":material/menu_book: Descripción completa de todas las dimensiones"):
            for dim, nombre in NOMBRES_DIMENSION.items():
                puntaje = vector[dim]
                st.markdown(f"**{nombre} ({dim}) — {puntaje}/6**")
                st.write(DESCRIPCIONES_DIMENSION[dim])
                st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# 1. GESTIÓN DE DATOS
# ──────────────────────────────────────────────────────────────────────────────
elif pestana == ":material/folder: Gestión de Datos":
    st.markdown("""
    <div class="section-header">
      <h2>Gestión de Conjuntos de Datos</h2>
      <p>Los datos se mantienen en memoria. Carga CSV, añade registros manualmente o genera datos sintéticos.</p>
    </div>""", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([":material/upload_file: Cargar CSV", ":material/science: Generar Sintéticos", ":material/movie: Datos de Demo"])

    # ── TAB 1: CSV ────────────────────────────────────────────────────────────
    with tab1:
        archivo = st.file_uploader(
            "Sube un CSV de Google Forms o con columnas (sexo, r, i, a, s, e, c)",
            type=["csv"],
        )
        if archivo:
            df_csv = pd.read_csv(archivo)

            MAPEO_RESPUESTAS = {
                "No me identifica": 0, "A veces me identifica": 1, "Sí, me identifica mucho": 2,
                "No, prefiero actividades de oficina": 0, "Depende de la situación": 1, "Sí, lo prefiero": 2,
                "No me interesa": 0, "Me interesa un poco": 1, "Me interesa mucho": 2,
                "Casi nunca": 0, "Algunas veces": 1, "Siempre": 2,
                "Poco o nada": 0, "De vez en cuando": 1, "Mucho": 2,
                "No suele gustarme": 0, "Solo cuando el tema me interesa": 1, "Sí, frecuentemente": 2,
                "No es algo que disfrute": 0, "En algunas ocasiones": 1, "Sí, es una de mis formas favoritas": 2,
                "Prefiero soluciones tradicionales": 0, "Casi siempre": 2,
                "Prefiero no involucrarme": 0, "Solo en algunas ocasiones": 1, "Siempre que puedo": 2,
                "Me cuesta trabajo": 0, "Depende de la persona": 1, "Sí, con facilidad": 2,
                "Prefiero trabajar solo": 0, "Depende de la actividad": 1, "Sí, definitivamente": 2,
                "Prefiero no liderar": 0, "Solo cuando es necesario": 1, "Sí, disfruto liderar": 2,
                "Prefiero evitar riesgos": 0, "Solo si el riesgo es razonable": 1, "Sí, sin problema": 2,
                "No está en mis planes": 0, "Tal vez": 1,
                "Prefiero trabajar de manera flexible": 0, "Sí, siempre": 2,
                "Prefiero tener libertad para decidir": 0, "Sí, me siento más cómodo(a)": 2,
                "Sí, de manera constante": 2,
            }

            if "Marca temporal" in df_csv.columns:
                # Validación de mapeo antes de procesar
                advertencias_mapeo = validar_mapeo_csv(df_csv, MAPEO_RESPUESTAS)
                if advertencias_mapeo:
                    st.warning("**Valores no mapeados detectados** (serán convertidos a 0):")
                    for adv in advertencias_mapeo:
                        st.markdown(f'<div class="alert-warning">{adv}</div>', unsafe_allow_html=True)

                cols = df_csv.columns
                df_procesado = pd.DataFrame()
                df_procesado["sexo"] = df_csv[cols[1]].map({"Hombre": "M", "Mujer": "F"}).fillna("Otro")
                df_procesado["r"] = df_csv[cols[2]].map(MAPEO_RESPUESTAS).fillna(0) + df_csv[cols[3]].map(MAPEO_RESPUESTAS).fillna(0) + df_csv[cols[4]].map(MAPEO_RESPUESTAS).fillna(0)
                df_procesado["i"] = df_csv[cols[5]].map(MAPEO_RESPUESTAS).fillna(0) + df_csv[cols[6]].map(MAPEO_RESPUESTAS).fillna(0) + df_csv[cols[7]].map(MAPEO_RESPUESTAS).fillna(0)
                df_procesado["a"] = df_csv[cols[8]].map(MAPEO_RESPUESTAS).fillna(0) + df_csv[cols[9]].map(MAPEO_RESPUESTAS).fillna(0) + df_csv[cols[10]].map(MAPEO_RESPUESTAS).fillna(0)
                df_procesado["s"] = df_csv[cols[11]].map(MAPEO_RESPUESTAS).fillna(0) + df_csv[cols[12]].map(MAPEO_RESPUESTAS).fillna(0) + df_csv[cols[13]].map(MAPEO_RESPUESTAS).fillna(0)
                df_procesado["e"] = df_csv[cols[14]].map(MAPEO_RESPUESTAS).fillna(0) + df_csv[cols[15]].map(MAPEO_RESPUESTAS).fillna(0) + df_csv[cols[16]].map(MAPEO_RESPUESTAS).fillna(0)
                df_procesado["c"] = df_csv[cols[17]].map(MAPEO_RESPUESTAS).fillna(0) + df_csv[cols[18]].map(MAPEO_RESPUESTAS).fillna(0) + df_csv[cols[19]].map(MAPEO_RESPUESTAS).fillna(0)
                df_csv = df_procesado
            else:
                df_csv.columns = [str(c).lower() for c in df_csv.columns]

            cols_req = {"sexo", "r", "i", "a", "s", "e", "c"}
            if cols_req.issubset(set(df_csv.columns)):
                st.success(f"CSV válido — {len(df_csv)} registros detectados.")
                st.dataframe(df_csv.head(10), use_container_width=True)
                if st.button(":material/add: Integrar datos al conjunto actual", type="primary"):
                    df_csv["id"] = range(st.session_state["id_counter"], st.session_state["id_counter"] + len(df_csv))
                    st.session_state["id_counter"] += len(df_csv)
                    df_csv["fecha"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                    df_csv["dominante"] = df_csv.apply(calcular_dominante, axis=1)
                    st.session_state["df_datos"] = pd.concat(
                        [st.session_state["df_datos"], df_csv[list(st.session_state["df_datos"].columns)]],
                        ignore_index=True,
                    )
                    if "df_filtrado" in st.session_state:
                        del st.session_state["df_filtrado"]
                    st.success(f"{len(df_csv)} registros integrados. Total: {len(st.session_state['df_datos'])}")
            else:
                st.error("El CSV no contiene las columnas requeridas (sexo, r, i, a, s, e, c) o no se pudo mapear.")


    # ── TAB 2: SINTÉTICOS ─────────────────────────────────────────────────────
    with tab2:
        st.subheader(":material/science: Simulación de Carga")
        st.write("Genera un conjunto basado en la tendencia de los datos actuales (si existen) o de forma aleatoria.")
        num_regs = st.number_input("Cantidad de registros", 100, 20000, 1000)
        if st.button(":material/settings: Generar Conjunto Sintético", type="primary"):
            sexos = ["M", "F"]
            df_actual = st.session_state["df_datos"]
            nuevos_datos = {"sexo": np.random.choice(sexos, num_regs)}
            for dim in ["r", "i", "a", "s", "e", "c"]:
                if not df_actual.empty:
                    media = df_actual[dim].mean()
                    desv = df_actual[dim].std()
                    if pd.isna(desv) or desv == 0:
                        desv = 1.0
                    valores = np.random.normal(loc=media, scale=desv, size=num_regs)
                else:
                    valores = np.random.uniform(0, 6, size=num_regs)
                valores = np.clip(np.round(valores), 0, 6).astype(int)
                nuevos_datos[dim] = valores

            syn_df = pd.DataFrame(nuevos_datos)
            syn_df["id"] = range(st.session_state["id_counter"], st.session_state["id_counter"] + num_regs)
            st.session_state["id_counter"] += num_regs
            syn_df["fecha"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            syn_df["dominante"] = syn_df.apply(calcular_dominante, axis=1)
            st.session_state["df_datos"] = pd.concat(
                [st.session_state["df_datos"], syn_df], ignore_index=True
            )
            if "df_filtrado" in st.session_state:
                del st.session_state["df_filtrado"]
            st.success(f"{num_regs} registros sintéticos generados. Total: {len(st.session_state['df_datos'])}")

    # ── TAB 3: DEMO ───────────────────────────────────────────────────────────
    with tab3:
        st.subheader(":material/movie: Datos de Demo")
        st.write(
            "Carga un conjunto de datos sintéticos preconfigurado para explorar todas "
            "las funcionalidades de RIASEC Analytics sin necesidad de tener datos propios."
        )

        c1, c2 = st.columns(2)
        n_demo = c1.selectbox("Tamaño del conjunto demo", [200, 500, 1000, 2000], index=1)
        seed_demo = c2.number_input("Semilla aleatoria", 1, 9999, 42)

        if st.button(":material/rocket_launch: Cargar Datos de Demo", type="primary"):
            np.random.seed(seed_demo)
            # Generar perfiles distribuidos por dimensión dominante para garantizar variedad
            registros_demo = []
            dims_list = ["R", "I", "A", "S", "E", "C"]
            por_dim = n_demo // 6
            resto = n_demo % 6

            for i_dim, dim in enumerate(dims_list):
                cantidad = por_dim + (1 if i_dim < resto else 0)
                for _ in range(cantidad):
                    datos = {}
                    for d in ["r", "i", "a", "s", "e", "c"]:
                        if d == dim.lower():
                            datos[d] = int(np.clip(np.round(np.random.normal(4.5, 0.8)), 3, 6))
                        else:
                            datos[d] = int(np.clip(np.round(np.random.normal(2.0, 1.2)), 0, 4))
                    datos["sexo"] = np.random.choice(["M", "F"])
                    registros_demo.append(datos)

            np.random.shuffle(registros_demo)
            df_demo = pd.DataFrame(registros_demo)
            df_demo["id"] = range(st.session_state["id_counter"], st.session_state["id_counter"] + len(df_demo))
            st.session_state["id_counter"] += len(df_demo)
            df_demo["fecha"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            df_demo["dominante"] = df_demo.apply(calcular_dominante, axis=1)

            st.session_state["df_datos"] = pd.concat(
                [st.session_state["df_datos"], df_demo], ignore_index=True
            )
            if "df_filtrado" in st.session_state:
                del st.session_state["df_filtrado"]
            st.success(f":material/check_circle: {len(df_demo)} registros de demo cargados. ¡Explora el Dashboard!")

    # ── Panel inferior: resumen y limpieza ────────────────────────────────────
    st.markdown("---")
    total_registros = len(st.session_state["df_datos"])
    col_total, col_clean = st.columns([3, 1])
    col_total.metric("Total de registros en memoria", total_registros)

    if col_clean.button(":material/delete: Limpiar Todos los Datos", type="secondary"):
        st.session_state["df_datos"] = st.session_state["df_datos"].iloc[0:0]
        st.session_state["modelo_info"] = None
        st.session_state["curva_bic"] = None
        for key in ["df_filtrado", "df_filtrado_clusterizado"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    if total_registros > 0:
        # Alertas de calidad
        alertas = verificar_calidad_datos(st.session_state["df_datos"])
        if alertas:
            with st.expander(":material/warning: Alertas de Calidad de Datos"):
                for a in alertas:
                    st.markdown(f'<div class="alert-warning">{a}</div>', unsafe_allow_html=True)

        st.dataframe(st.session_state["df_datos"].head(100), use_container_width=True)

        # Descarga CSV
        csv_bytes = st.session_state["df_datos"].to_csv(index=False).encode("utf-8")
        st.download_button(
            ":material/download: Exportar todos los datos a CSV",
            data=csv_bytes,
            file_name="datos_riasec.csv",
            mime="text/csv",
        )

# ──────────────────────────────────────────────────────────────────────────────
# 2. DASHBOARD
# ──────────────────────────────────────────────────────────────────────────────
elif pestana == ":material/dashboard: Dashboard":
    st.markdown("""
    <div class="section-header">
      <h2>Dashboard de Análisis</h2>
      <p>Explora la distribución de perfiles RIASEC en el conjunto de datos cargado.</p>
    </div>""", unsafe_allow_html=True)

    df = st.session_state["df_datos"].copy()

    if df.empty:
        st.info(":material/info: El Dashboard está en blanco. Ve a **📁 Gestión de Datos** para cargar registros o usa los **Datos de Demo**.")
    else:
        df["dominante_nombre"] = df["dominante"].apply(
            lambda x: NOMBRES_DIMENSION.get(x.split("-")[0], x) if isinstance(x, str) else x
        )

        # ── Filtros (ahora opcionales) ────────────────────────────────────────
        with st.expander(":material/filter_alt: Filtros (opcionales)", expanded=True):
            col1, col2 = st.columns(2)
            filtro_sexo = col1.multiselect("Sexo", df["sexo"].unique())
            filtro_dom = col2.multiselect("Dimensión Dominante", sorted(df["dominante_nombre"].unique()))

        df_filtrado = df.copy()
        if filtro_sexo:
            df_filtrado = df_filtrado[df_filtrado["sexo"].isin(filtro_sexo)]
        if filtro_dom:
            df_filtrado = df_filtrado[df_filtrado["dominante_nombre"].isin(filtro_dom)]

        st.session_state["df_filtrado"] = df_filtrado

        # ── Métricas ──────────────────────────────────────────────────────────
        dom_max = df_filtrado["dominante_nombre"].mode()[0] if not df_filtrado.empty else "N/A"
        pct_dom = (df_filtrado["dominante_nombre"] == dom_max).mean() * 100 if not df_filtrado.empty else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="metric-card"><p class="label">Registros</p><h3 class="value">{len(df_filtrado):,}</h3></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><p class="label">Perfil Mayoritario</p><h3 class="value" style="font-size:1.2rem">{dom_max}</h3></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card"><p class="label">% Perfil Dom.</p><h3 class="value">{pct_dom:.1f}%</h3></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="metric-card"><p class="label">Motor Analítico</p><h3 class="value" style="font-size:1rem">GMM</h3></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if df_filtrado.empty:
            st.warning("No hay datos con los filtros aplicados.")
        else:
            # Alertas de calidad del filtrado
            alertas = verificar_calidad_datos(df_filtrado)
            if alertas:
                with st.expander(":material/warning: Alertas de Calidad"):
                    for a in alertas:
                        st.markdown(f'<div class="alert-warning">{a}</div>', unsafe_allow_html=True)

            # ── Fila 1: Radar + Distribución dominante ────────────────────────
            col_r, col_p = st.columns(2)
            with col_r:
                fig_radar = crear_radar_plotly(df_filtrado, "Perfil Promedio RIASEC")
                st.plotly_chart(fig_radar, use_container_width=True)

            with col_p:
                st.subheader("Distribución por Dimensión Dominante")
                fig_dom = px.pie(
                    df_filtrado, names="dominante_nombre", hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Bold,
                )
                fig_dom.update_layout(height=380, margin=dict(t=30))
                st.plotly_chart(fig_dom, use_container_width=True)

            # ── Fila 2: Barras + Distribución por sexo ─────────────────────────
            col_b, col_s = st.columns(2)
            with col_b:
                st.subheader("Puntajes Promedio por Dimensión")
                promedios = df_filtrado[["r", "i", "a", "s", "e", "c"]].mean().reset_index()
                promedios.columns = ["Dimensión", "Promedio"]
                promedios["Dimensión"] = promedios["Dimensión"].str.upper()
                fig_bar = px.bar(
                    promedios, x="Dimensión", y="Promedio",
                    text="Promedio", color="Dimensión",
                    color_discrete_sequence=px.colors.qualitative.Bold,
                )
                fig_bar.update_traces(texttemplate="%{text:.2f}", textposition="outside")
                fig_bar.update_layout(yaxis=dict(range=[0, 6.5]), showlegend=False, height=350)
                st.plotly_chart(fig_bar, use_container_width=True)

            with col_s:
                if "sexo" in df_filtrado.columns and df_filtrado["sexo"].nunique() > 1:
                    st.subheader("Distribución por Sexo")
                    fig_sex = px.histogram(
                        df_filtrado, x="dominante_nombre", color="sexo",
                        barmode="group",
                        color_discrete_map={"M": "#1E3A8A", "F": "#EC4899", "Otro": "#10B981"},
                        labels={"dominante_nombre": "Dimensión Dominante", "sexo": "Sexo"},
                    )
                    fig_sex.update_layout(height=350, showlegend=True)
                    st.plotly_chart(fig_sex, use_container_width=True)
                else:
                    st.subheader("Box Plot por Dimensión")
                    df_melt = df_filtrado[["r", "i", "a", "s", "e", "c"]].melt(var_name="Dimensión", value_name="Puntaje")
                    df_melt["Dimensión"] = df_melt["Dimensión"].str.upper()
                    fig_box = px.box(df_melt, x="Dimensión", y="Puntaje", color="Dimensión",
                                     color_discrete_sequence=px.colors.qualitative.Bold)
                    fig_box.update_layout(showlegend=False, height=350)
                    st.plotly_chart(fig_box, use_container_width=True)

            # ── Info RIASEC ───────────────────────────────────────────────────
            with st.expander(":material/info: ¿Qué significa cada dimensión RIASEC?"):
                cols_info = st.columns(3)
                for i, (dim, nombre) in enumerate(NOMBRES_DIMENSION.items()):
                    with cols_info[i % 3]:
                        st.markdown(f"**{nombre} ({dim})**")
                        st.caption(DESCRIPCIONES_DIMENSION[dim])

# ──────────────────────────────────────────────────────────────────────────────
# 3. ALGORITMO GMM (CLUSTERING)
# ──────────────────────────────────────────────────────────────────────────────
elif pestana == ":material/science: Algoritmo GMM":
    st.markdown("""
    <div class="section-header">
      <h2>Entrenamiento Gaussian Mixture Model (GMM)</h2>
      <p>Clustering probabilístico de perfiles vocacionales con validación BIC/AIC.</p>
    </div>""", unsafe_allow_html=True)

    df = st.session_state.get("df_filtrado", st.session_state["df_datos"]).copy()

    if len(df) < 5:
        st.error("Requiere al menos 5 registros (filtrados) para ejecutar el clustering. Ve al Dashboard para filtrar.")
    else:
        st.info(f"Utilizando **{len(df):,}** registros del filtro actual del Dashboard.")

        # ── Parámetros avanzados ──────────────────────────────────────────────
        with st.expander(":material/settings: Configuración Avanzada", expanded=False):
            col_adv1, col_adv2, col_adv3 = st.columns(3)
            seed_val = col_adv1.number_input("Semilla aleatoria (random seed)", 0, 99999, 42, step=1)
            umbral_val = col_adv2.slider(
                "Umbral de perfil combinado",
                0.0, 2.0, 0.75, step=0.05,
                help="Si la diferencia entre la dimensión principal y la secundaria es ≤ este valor, se etiqueta como perfil combinado.",
            )
            max_k_val = col_adv3.number_input("k máximo para búsqueda BIC", 4, 20, 10)

        # ── Búsqueda de k óptimo ──────────────────────────────────────────────
        if st.button(":material/lightbulb: Calcular Clústeres Óptimos (Método BIC paralelo)"):
            with st.spinner("Analizando componentes en paralelo..."):
                k_sug, curva = encontrar_k_optimo(df, max_k=max_k_val, random_state=seed_val)
                st.session_state["k_sugerido"] = k_sug
                st.session_state["curva_bic"] = curva
            st.success(f"Cantidad óptima sugerida: **{k_sug} clústeres** (menor BIC).")

        # ── Gráfica curva BIC/AIC ─────────────────────────────────────────────
        if st.session_state["curva_bic"]:
            curva = st.session_state["curva_bic"]
            fig_bic = go.Figure()
            fig_bic.add_trace(go.Scatter(
                x=curva["k"], y=curva["bic"],
                mode="lines+markers", name="BIC",
                line=dict(color="#1E3A8A", width=2.5),
                marker=dict(size=8),
            ))
            fig_bic.add_trace(go.Scatter(
                x=curva["k"], y=curva["aic"],
                mode="lines+markers", name="AIC",
                line=dict(color="#EC4899", width=2, dash="dash"),
                marker=dict(size=7),
            ))
            k_opt = st.session_state.get("k_sugerido", curva["k"][0])
            k_idx = curva["k"].index(k_opt) if k_opt in curva["k"] else 0
            fig_bic.add_vline(x=k_opt, line_dash="dot", line_color="#10B981",
                               annotation_text=f"k óptimo = {k_opt}", annotation_position="top right")
            fig_bic.update_layout(
                title="Criterios de Información BIC/AIC por número de clústeres",
                xaxis_title="Número de clústeres (k)",
                yaxis_title="Valor del criterio",
                height=350,
                legend=dict(orientation="h", y=1.1),
            )
            st.plotly_chart(fig_bic, use_container_width=True)

        # ── Controles de entrenamiento ────────────────────────────────────────
        k_val = st.session_state.get("k_sugerido", 6)
        col1, col2 = st.columns(2)
        min_k = max(2, k_val - 3)
        max_k = min(20, k_val + 3)
        k = col1.number_input(
            "Número de clústeres (variación ±3 sobre el sugerido)",
            min_value=min_k, max_value=max_k, value=k_val,
        )

        opciones_cov = {
            "Completa (full)": "full",
            "Atada (tied)": "tied",
            "Diagonal (diag)": "diag",
            "Esférica (spherical)": "spherical",
        }
        cov_type_label = col2.selectbox("Tipo de Covarianza", list(opciones_cov.keys()))
        cov_type = opciones_cov[cov_type_label]

        if st.button(":material/rocket_launch: Ejecutar Clustering GMM", type="primary"):
            with st.spinner("Entrenando GMM..."):
                try:
                    res = ejecutar_clustering_en_memoria(
                        df, k, cov_type,
                        random_state=seed_val,
                        umbral_diferencia=umbral_val,
                    )
                    df["cluster"] = res["etiquetas_asignadas"]
                    df["perfil_cluster"] = df["cluster"].map(res["etiquetas_texto"])
                    st.session_state["df_filtrado_clusterizado"] = df
                    st.session_state["modelo_info"] = res
                    st.success("Clustering completado y modelo guardado en historial.")
                except Exception as ex:
                    st.error(f"Error durante el clustering: {ex}")

        # ── Resultados del modelo ─────────────────────────────────────────────
        if st.session_state["modelo_info"]:
            res = st.session_state["modelo_info"]

            st.markdown("### 📈 Métricas del Modelo")
            m1, m2, m3 = st.columns(3)
            m1.metric("Silhouette Score", round(res["silhouette"], 3) if res["silhouette"] else "N/A",
                      help="Cercano a 1 = clusters bien separados")
            m2.metric("BIC", round(res["bic"], 1),
                      help="Menor BIC = mejor modelo")
            m3.metric("AIC", round(res["aic"], 1),
                      help="Menor AIC = mejor modelo")

            with st.expander(":material/info: ¿Cómo interpretar estas métricas?"):
                st.write("- **Silhouette Score:** Mide cuán bien separados están los clusters. Rango [-1, 1]; valores > 0.5 son buenos.")
                st.write("- **BIC / AIC:** Criterios de información; penalizan la complejidad. **Valores más bajos = mejor modelo.**")

            st.markdown("### 🏷️ Etiquetas de Clústeres")
            for c_idx, label in res["etiquetas_texto"].items():
                st.markdown(f"- **Clúster {c_idx}:** {label}")

            # PCA visualization moved to Resultados y Reportes tab

# ──────────────────────────────────────────────────────────────────────────────
# 4. RESULTADOS Y REPORTES
# ──────────────────────────────────────────────────────────────────────────────
elif pestana == ":material/description: Resultados y Reportes":
    st.markdown("""
    <div class="section-header">
      <h2>Resultados y Generación de Reportes PDF</h2>
      <p>Visualiza la clasificación de estudiantes y descarga el reporte completo.</p>
    </div>""", unsafe_allow_html=True)

    if "df_filtrado_clusterizado" not in st.session_state or st.session_state["modelo_info"] is None:
        st.info(":material/info: Ejecuta primero el Clustering GMM para ver esta sección.")
    else:
        df_completo = st.session_state["df_filtrado_clusterizado"]
        modelo_info = st.session_state["modelo_info"]

        # Filtros en sidebar
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Filtros del Reporte PDF")
        f_sexo = st.sidebar.multiselect("Filtrar por Sexo (PDF)", df_completo["sexo"].unique())
        f_cluster = st.sidebar.multiselect("Filtrar por Clúster (PDF)", sorted(df_completo["cluster"].unique()))

        df_pdf = df_completo.copy()
        if f_sexo:
            df_pdf = df_pdf[df_pdf["sexo"].isin(f_sexo)]
        if f_cluster:
            df_pdf = df_pdf[df_pdf["cluster"].isin(f_cluster)]

        tab1, tab2, tab3, tab4 = st.tabs([":material/table_chart: Tabla de Clasificación", ":material/school: Carreras Sugeridas", ":material/picture_as_pdf: PDF", ":material/map: PCA 2D"])

        with tab1:
            st.write(f"Clasificación de **{len(df_pdf):,}** registros (con filtros actuales).")
            st.dataframe(
                df_pdf[["id", "sexo", "dominante", "cluster", "perfil_cluster"]],
                use_container_width=True,
            )
            # Exportar CSV filtrado
            csv_bytes = df_pdf.to_csv(index=False).encode("utf-8")
            st.download_button(
                ":material/download: Descargar clasificación filtrada (CSV)",
                data=csv_bytes,
                file_name="clasificacion_riasec.csv",
                mime="text/csv",
            )

        with tab2:
            st.subheader(":material/school: Carreras Sugeridas por Perfil de Cluster")
            for c_idx, label in modelo_info["etiquetas_texto"].items():
                with st.expander(f"Clúster {c_idx} — {label}"):
                    # Extraer la dimensión principal del label
                    dim_principal = None
                    for dim, nombre in NOMBRES_DIMENSION.items():
                        if nombre in label:
                            dim_principal = dim
                            break

                    if dim_principal:
                        carreras = CARRERAS_RIASEC.get(dim_principal, [])
                        pills_html = " ".join(f'<span class="career-pill">{c}</span>' for c in carreras)
                        st.markdown(pills_html, unsafe_allow_html=True)
                        st.markdown(f"*{DESCRIPCIONES_DIMENSION.get(dim_principal, '')}*")

                    n_cluster = (df_pdf["cluster"] == c_idx).sum() if "cluster" in df_pdf.columns else 0
                    st.metric("Registros en este cluster", n_cluster)

        with tab3:
            st.write("Genera y descarga el reporte PDF completo con radar chart, estadísticas y carreras sugeridas.")
            with st.spinner("Generando PDF..."):
                buffer_pdf = generar_reporte_pdf(df_pdf, modelo_info)

            col_dl, col_prev = st.columns([1, 3])
            col_dl.download_button(
                label=":material/download: Descargar PDF",
                data=buffer_pdf.getvalue(),
                file_name="Reporte_Clustering_RIASEC.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True,
            )

            st.markdown("### 👁️ Previsualización")
            base64_pdf = base64.b64encode(buffer_pdf.getvalue()).decode("utf-8")
            pdf_display = (
                f'<iframe src="data:application/pdf;base64,{base64_pdf}" '
                f'width="100%" height="800" type="application/pdf"></iframe>'
            )
            st.markdown(pdf_display, unsafe_allow_html=True)
            st.caption(":material/warning: Si el visor no muestra el PDF, usa el botón de descarga arriba.")

        with tab4:
            st.subheader(":material/map: Visualización PCA 2D")
            if "pca_data" in modelo_info and modelo_info["pca_data"]:
                pca = modelo_info["pca_data"]
                var1 = pca["varianza_explicada"][0] * 100
                var2 = pca["varianza_explicada"][1] * 100

                df_pca = pd.DataFrame({
                    "PC1": pca["x"],
                    "PC2": pca["y"],
                    "Cluster": [str(c) for c in pca["cluster"]],
                })
                # Añadir etiqueta de texto
                df_pca["Perfil"] = df_pca["Cluster"].map(
                    {str(k): v for k, v in modelo_info["etiquetas_texto"].items()}
                )

                fig_pca = px.scatter(
                    df_pca, x="PC1", y="PC2", color="Cluster",
                    hover_data=["Perfil"],
                    labels={"PC1": f"PC1 ({var1:.1f}% var.)", "PC2": f"PC2 ({var2:.1f}% var.)"},
                    title=f"Clusters en 2D — Varianza explicada: {var1+var2:.1f}%",
                    color_discrete_sequence=px.colors.qualitative.Bold,
                    opacity=0.7,
                )
                fig_pca.update_traces(marker=dict(size=6))
                fig_pca.update_layout(height=450)
                st.plotly_chart(fig_pca, use_container_width=True)
            else:
                st.info("La visualización PCA no está disponible para este modelo.")

# ──────────────────────────────────────────────────────────────────────────────
# 5. HISTORIAL DE MODELOS
# ──────────────────────────────────────────────────────────────────────────────
elif pestana == ":material/history: Historial de Modelos":
    st.markdown("""
    <div class="section-header">
      <h2>Historial de Modelos Entrenados</h2>
      <p>Compara y reutiliza modelos GMM previamente entrenados.</p>
    </div>""", unsafe_allow_html=True)

    session = get_session()
    modelos = session.query(ModeloEntrenado).order_by(ModeloEntrenado.id.desc()).all()
    session.close()

    if not modelos:
        st.info("No hay modelos entrenados en el historial. Ejecuta el clustering GMM para guardar uno.")
    else:
        data_modelos = []
        for m in modelos:
            data_modelos.append({
                "ID": m.id,
                "Fecha": m.fecha_creacion.strftime("%Y-%m-%d %H:%M") if m.fecha_creacion else "N/A",
                "Clústeres (k)": m.n_componentes,
                "Covarianza": m.covariance_type,
                "BIC": round(m.bic, 1) if m.bic else None,
                "AIC": round(m.aic, 1) if m.aic else None,
                "Registros": m.n_registros_entrenamiento,
                "Activo": "✅" if m.activo else "—",
            })

        df_hist = pd.DataFrame(data_modelos)
        st.dataframe(df_hist, use_container_width=True)

        # ── Gráfico comparativo de métricas ───────────────────────────────────
        if len(df_hist) > 1:
            st.subheader(":material/bar_chart: Comparativa de Métricas entre Modelos")
            df_metricas = df_hist.dropna(subset=["BIC", "AIC"]).copy()
            if not df_metricas.empty:
                fig_comp = go.Figure()
                fig_comp.add_trace(go.Bar(
                    x=[f"Modelo #{i}" for i in df_metricas["ID"]],
                    y=df_metricas["BIC"],
                    name="BIC",
                    marker_color="#1E3A8A",
                ))
                fig_comp.add_trace(go.Bar(
                    x=[f"Modelo #{i}" for i in df_metricas["ID"]],
                    y=df_metricas["AIC"],
                    name="AIC",
                    marker_color="#60A5FA",
                ))
                fig_comp.update_layout(
                    barmode="group",
                    title="BIC y AIC por modelo (menor = mejor)",
                    xaxis_title="Modelo",
                    yaxis_title="Valor",
                    height=350,
                    legend=dict(orientation="h"),
                )
                st.plotly_chart(fig_comp, use_container_width=True)

        # ── Aplicar o Eliminar modelo histórico ───────────────────────────────────────────
        st.markdown("### :material/history: Gestionar Modelo Histórico")
        modelo_seleccionado = st.selectbox("Selecciona el ID del modelo", df_hist["ID"].tolist())

        col_btn1, col_btn2 = st.columns(2)

        with col_btn1:
            if st.button(":material/bolt: Aplicar Modelo", type="primary", use_container_width=True):
                df_actual = st.session_state.get("df_filtrado", st.session_state["df_datos"]).copy()
                if df_actual.empty:                    st.error("No hay datos cargados. Ve a 📁 Gestión de Datos primero.")
                else:
                    with st.spinner("Cargando y aplicando modelo..."):
                        try:
                            res = cargar_y_aplicar_modelo(df_actual, modelo_seleccionado)
                            df_actual["cluster"] = res["etiquetas_asignadas"]
                            df_actual["perfil_cluster"] = df_actual["cluster"].map(res["etiquetas_texto"])
                            st.session_state["df_filtrado_clusterizado"] = df_actual
                            st.session_state["modelo_info"] = {
                                "modelo_id": modelo_seleccionado,
                                "etiquetas_asignadas": res["etiquetas_asignadas"],
                                "etiquetas_texto": res["etiquetas_texto"],
                                "n_componentes": res.get("n_componentes", len(res["etiquetas_texto"])),
                                "silhouette": None,
                                "bic": res.get("bic"),
                                "aic": res.get("aic"),
                                "pca_data": res.get("pca_data"),
                            }
                            # Actualizar estado activo en BD
                            db_sess = get_session()
                            try:
                                db_sess.query(ModeloEntrenado).update({ModeloEntrenado.activo: False})
                                db_sess.query(ModeloEntrenado).filter_by(id=modelo_seleccionado).update({ModeloEntrenado.activo: True})
                                db_sess.commit()
                            except Exception as e_db:
                                st.warning(f"No se pudo actualizar el estado activo en la base de datos: {e_db}")
                            finally:
                                db_sess.close()
    
                            st.success(f"Modelo #{modelo_seleccionado} aplicado correctamente.")
                            st.info("Ve a :material/description: Resultados y Reportes para visualizar.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error al cargar el modelo: {e}")
        with col_btn2:
            if st.button(":material/delete: Eliminar Modelo", type="secondary", use_container_width=True):
                db_sess = get_session()
                try:
                    modelo_a_eliminar = db_sess.query(ModeloEntrenado).filter_by(id=modelo_seleccionado).first()
                    if modelo_a_eliminar:
                        if modelo_a_eliminar.ruta_archivo and os.path.exists(modelo_a_eliminar.ruta_archivo):
                            try:
                                os.remove(modelo_a_eliminar.ruta_archivo)
                            except Exception as e_os:
                                st.warning(f"No se pudo eliminar el archivo físico: {e_os}")
                        db_sess.delete(modelo_a_eliminar)
                        db_sess.commit()
                        st.success(f"Modelo #{modelo_seleccionado} eliminado correctamente.")
                        if st.session_state.get("modelo_info") and st.session_state["modelo_info"].get("modelo_id") == modelo_seleccionado:
                            st.session_state["modelo_info"] = None
                            if "df_filtrado_clusterizado" in st.session_state:
                                del st.session_state["df_filtrado_clusterizado"]
                        st.rerun()
                except Exception as e_del:
                    st.error(f"Error al eliminar el modelo: {e_del}")
                finally:
                    db_sess.close()
