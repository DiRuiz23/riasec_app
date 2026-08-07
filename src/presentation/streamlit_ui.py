"""
app.py
Interfaz Streamlit del Perfil de Personalidad Vocacional RIASEC.
- Gestión de datos en memoria (no almacena respuestas en BD).
- Algoritmo GMM con validación BIC/AIC.
- Generación de reportes PDF.
"""

import json
import datetime
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.infrastructure.database.db import init_db
from src.application.cuestionario_service import NOMBRES_DIMENSION
from src.application.clustering_service import (
    encontrar_k_optimo, 
    ejecutar_clustering_en_memoria
)
from src.application.report_generator import generar_reporte_pdf

# --------------------------------------------------------------------------
# Configuración
# --------------------------------------------------------------------------
st.set_page_config(page_title="RIASEC · Clustering", layout="wide")
init_db()

# Inicialización de estado en memoria
if "df_datos" not in st.session_state:
    st.session_state["df_datos"] = pd.DataFrame(columns=["id", "sexo", "fecha", "r", "i", "a", "s", "e", "c", "dominante"])
    st.session_state["id_counter"] = 1
    
if "modelo_info" not in st.session_state:
    st.session_state["modelo_info"] = None

st.markdown("""
<style>
.metric-box { background-color: #f8f9fa; padding: 1rem; border-radius: 8px; text-align: center; border-left: 4px solid #3B82F6;}
.metric-box h3 { margin:0; font-size: 1.8rem; color: #1E3A8A; }
.metric-box p { margin:0; font-size: 0.8rem; color: #6B7280; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Barra lateral
# --------------------------------------------------------------------------
st.sidebar.title("RIASEC Analytics")
pestana = st.sidebar.radio("Navegación", [
    "Dashboard", 
    "Gestión de Datos", 
    "Algoritmo GMM (Clustering)", 
    "Resultados y Reportes"
])

def calcular_dominante(row):
    dims = {'r': row['r'], 'i': row['i'], 'a': row['a'], 's': row['s'], 'e': row['e'], 'c': row['c']}
    return max(dims, key=dims.get).upper()

# --------------------------------------------------------------------------
# 1. GESTIÓN DE DATOS
# --------------------------------------------------------------------------
if pestana == "Gestión de Datos":
    st.header("Gestión de Conjuntos de Datos")
    st.info("Los datos se mantienen temporalmente en memoria (hasta 10,000 registros recomendados) y no se guardan en la base de datos.")
    
    tab1, tab2, tab3 = st.tabs(["Cargar CSV", "Formulario Manual", "Generar Sintéticos"])
    
    with tab1:
        archivo = st.file_uploader("Sube un CSV (columnas requeridas: sexo, r, i, a, s, e, c)", type=["csv"])
        if archivo:
            df_csv = pd.read_csv(archivo)
            cols_req = {"sexo", "r", "i", "a", "s", "e", "c"}
            df_csv.columns = [str(c).lower() for c in df_csv.columns]
            if cols_req.issubset(set(df_csv.columns)):
                if st.button("Integrar datos al conjunto actual"):
                    df_csv["id"] = range(st.session_state["id_counter"], st.session_state["id_counter"] + len(df_csv))
                    st.session_state["id_counter"] += len(df_csv)
                    df_csv["fecha"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                    df_csv["dominante"] = df_csv.apply(calcular_dominante, axis=1)
                    st.session_state["df_datos"] = pd.concat([st.session_state["df_datos"], df_csv[list(st.session_state["df_datos"].columns)]], ignore_index=True)
                    st.success(f"{len(df_csv)} registros integrados.")
            else:
                st.error("El CSV no contiene las columnas requeridas.")
                
    with tab2:
        st.subheader("Ingreso de Nuevo Registro")
        with st.form("form_registro"):
            col1, col2 = st.columns(2)
            sexo = col1.selectbox("Sexo", ["M", "F", "Otro"])
            r = col1.slider("Realista (R)", 0, 6, 3)
            i = col1.slider("Investigativo (I)", 0, 6, 3)
            a = col1.slider("Artístico (A)", 0, 6, 3)
            s = col2.slider("Social (S)", 0, 6, 3)
            e = col2.slider("Emprendedor (E)", 0, 6, 3)
            c = col2.slider("Convencional (C)", 0, 6, 3)
            if st.form_submit_button("Añadir al conjunto"):
                nuevo = {
                    "id": st.session_state["id_counter"], "sexo": sexo,
                    "fecha": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "r": r, "i": i, "a": a, "s": s, "e": e, "c": c
                }
                st.session_state["id_counter"] += 1
                nuevo["dominante"] = calcular_dominante(nuevo)
                st.session_state["df_datos"] = pd.concat([st.session_state["df_datos"], pd.DataFrame([nuevo])], ignore_index=True)
                st.success("Registro añadido.")
                
    with tab3:
        st.subheader("Simulación de Carga")
        st.write("Genera un conjunto aleatorio (ideal 10,000 registros para pruebas de GMM).")
        num_regs = st.number_input("Cantidad de registros", 100, 20000, 10000)
        if st.button("Generar Conjunto Sintético"):
            np.random.seed(42)
            syn_df = pd.DataFrame({
                "sexo": np.random.choice(["M", "F"], num_regs),
                "r": np.random.randint(0, 7, num_regs),
                "i": np.random.randint(0, 7, num_regs),
                "a": np.random.randint(0, 7, num_regs),
                "s": np.random.randint(0, 7, num_regs),
                "e": np.random.randint(0, 7, num_regs),
                "c": np.random.randint(0, 7, num_regs),
            })
            syn_df["id"] = range(st.session_state["id_counter"], st.session_state["id_counter"] + num_regs)
            st.session_state["id_counter"] += num_regs
            syn_df["fecha"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            syn_df["dominante"] = syn_df.apply(calcular_dominante, axis=1)
            st.session_state["df_datos"] = syn_df
            st.success(f"{num_regs} registros generados y cargados en memoria.")

    st.markdown("---")
    st.write(f"**Total de registros actuales:** {len(st.session_state['df_datos'])}")
    st.dataframe(st.session_state["df_datos"].head(100), use_container_width=True)
    if st.button("Limpiar Datos"):
        st.session_state["df_datos"] = st.session_state["df_datos"].iloc[0:0]
        st.session_state["modelo_info"] = None
        st.success("Conjunto de datos limpiado.")

# --------------------------------------------------------------------------
# 2. DASHBOARD
# --------------------------------------------------------------------------
elif pestana == "Dashboard":
    st.header("Dashboard de Análisis")
    
    df = st.session_state["df_datos"].copy()
    
    if df.empty:
        st.info("El Dashboard está en blanco. Dirígete a 'Gestión de Datos' para cargar registros.")
    else:
        st.write("Aplica filtros para generar las estadísticas:")
        col1, col2 = st.columns(2)
        filtro_sexo = col1.multiselect("Sexo", df["sexo"].unique())
        filtro_dom = col2.multiselect("Dimensión Dominante", df["dominante"].unique())
        
        if filtro_sexo:
            df = df[df["sexo"].isin(filtro_sexo)]
        if filtro_dom:
            df = df[df["dominante"].isin(filtro_dom)]
            
        st.session_state["df_filtrado"] = df
        
        if not filtro_sexo and not filtro_dom:
            st.warning("⚠️ **Debe aplicar al menos un filtro** para habilitar la generación de estadísticas del modelo.")
            if st.toggle("Habilitar Estadística Básica (Descriptiva sin filtros)", False):
                st.write(f"Total registros en memoria: {len(df)}")
                st.bar_chart(df["dominante"].value_counts())
        else:
            st.success(f"Estadísticas activadas para {len(df)} registros filtrados.")
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"<div class='metric-box'><p>Registros Filtrados</p><h3>{len(df)}</h3></div>", unsafe_allow_html=True)
            with c2:
                dom_max = df["dominante"].mode()[0] if not df.empty else "N/A"
                st.markdown(f"<div class='metric-box'><p>Perfil Mayoritario</p><h3>{dom_max}</h3></div>", unsafe_allow_html=True)
            with c3:
                st.markdown(f"<div class='metric-box'><p>Motor Analítico</p><h3>GMM</h3></div>", unsafe_allow_html=True)
            
            st.markdown("---")
            cc1, cc2 = st.columns(2)
            with cc1:
                st.subheader("Distribución por Dimensión Dominante")
                fig_dom = px.pie(df, names="dominante", hole=0.4)
                st.plotly_chart(fig_dom, use_container_width=True)
            with cc2:
                st.subheader("Puntajes Promedio RIASEC")
                promedios = df[["r", "i", "a", "s", "e", "c"]].mean().reset_index()
                promedios.columns = ["Dimensión", "Promedio"]
                fig_bar = px.bar(promedios, x="Dimensión", y="Promedio", text="Promedio", color="Dimensión")
                fig_bar.update_traces(texttemplate='%{text:.2f}')
                fig_bar.update_layout(yaxis=dict(range=[0,6]))
                st.plotly_chart(fig_bar, use_container_width=True)

# --------------------------------------------------------------------------
# 3. ALGORITMO GMM (CLUSTERING)
# --------------------------------------------------------------------------
elif pestana == "Algoritmo GMM (Clustering)":
    st.header("Entrenamiento Gaussian Mixture Model (GMM)")
    df = st.session_state.get("df_filtrado", st.session_state["df_datos"]).copy()
    
    if len(df) < 5:
        st.error("Requiere al menos 5 registros (filtrados) para ejecutar el clustering. Verifique el Dashboard.")
    else:
        st.write(f"Utilizando **{len(df)}** registros del filtro actual.")
        
        if st.button("💡 Calcular Clústeres Óptimos (Método BIC)"):
            with st.spinner("Analizando componentes..."):
                k_sug = encontrar_k_optimo(df, max_k=10, covariance_type="full")
                st.session_state["k_sugerido"] = k_sug
                st.success(f"Cantidad óptima sugerida: **{k_sug} clústeres**.")
                
        k_val = st.session_state.get("k_sugerido", 6)
        col1, col2 = st.columns(2)
        min_k = max(2, k_val - 3)
        max_k = min(20, k_val + 3)
        
        k = col1.number_input("Número de clústeres (variación ±3 permitida)", min_value=min_k, max_value=max_k, value=k_val)
        cov_type = col2.selectbox("Tipo de Covarianza", ["full", "tied", "diag", "spherical"])
        
        if st.button("Ejecutar Clustering GMM", type="primary"):
            with st.spinner("Ejecutando GMM en memoria..."):
                res = ejecutar_clustering_en_memoria(df, k, cov_type)
                
                df["cluster"] = res["etiquetas_asignadas"]
                df["perfil_cluster"] = df["cluster"].map(res["etiquetas_texto"])
                
                st.session_state["df_filtrado_clusterizado"] = df
                st.session_state["modelo_info"] = res
                
                st.success("Clustering completado. Las estadísticas se han guardado en la Base de Datos. (No se guardaron registros crudos).")
                
        if st.session_state["modelo_info"] is not None:
            res = st.session_state["modelo_info"]
            st.markdown("### Métricas del Modelo")
            m1, m2, m3 = st.columns(3)
            m1.metric("Silhouette Score", round(res['silhouette'], 3) if res['silhouette'] else 'N/A')
            m2.metric("BIC", round(res['bic'], 1))
            m3.metric("AIC", round(res['aic'], 1))
            
            st.markdown("### Patrones y Sesgos Identificados")
            for c_idx, label in res["etiquetas_texto"].items():
                st.write(f"- **Clúster {c_idx}**: {label}")

# --------------------------------------------------------------------------
# 4. RESULTADOS Y REPORTES
# --------------------------------------------------------------------------
elif pestana == "Resultados y Reportes":
    st.header("Resultados y Generación de Reportes PDF")
    
    if "df_filtrado_clusterizado" not in st.session_state or st.session_state["modelo_info"] is None:
        st.info("Ejecuta primero el Clustering GMM para ver esta sección.")
    else:
        df = st.session_state["df_filtrado_clusterizado"]
        modelo_info = st.session_state["modelo_info"]
        
        tab1, tab2 = st.tabs(["Tabla de Clasificación", "Descarga de Reporte PDF"])
        
        with tab1:
            st.write(f"Mostrando clasificación para {len(df)} estudiantes (solo registros filtrados).")
            st.dataframe(df[["id", "sexo", "dominante", "cluster", "perfil_cluster"]], use_container_width=True)
            
        with tab2:
            st.write("El reporte en PDF incluirá únicamente la información generada y filtrada en esta vista, omitiendo datos no filtrados.")
            if st.button("Generar Reporte PDF"):
                with st.spinner("Creando PDF..."):
                    buffer_pdf = generar_reporte_pdf(df, modelo_info)
                    st.download_button(
                        label="📥 Descargar Reporte GMM PDF",
                        data=buffer_pdf.getvalue(),
                        file_name="Reporte_Clustering_RIASEC.pdf",
                        mime="application/pdf"
                    )
