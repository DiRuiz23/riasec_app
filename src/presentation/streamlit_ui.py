"""
app.py
Interfaz Streamlit del Perfil de Personalidad Vocacional RIASEC.
- Gestión de datos en memoria (no almacena respuestas en BD).
- Algoritmo GMM con validación BIC/AIC.
- Generación de reportes PDF.
"""

import json
import datetime
import base64
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.infrastructure.database.db import init_db, get_session, ModeloEntrenado
from src.application.cuestionario_service import NOMBRES_DIMENSION
from src.application.clustering_service import (
    encontrar_k_optimo, 
    ejecutar_clustering_en_memoria,
    cargar_y_aplicar_modelo
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
    "Resultados y Reportes",
    "Historial de Modelos"
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
        archivo = st.file_uploader("Sube un CSV de Google Forms o con columnas requeridas (sexo, r, i, a, s, e, c)", type=["csv"])
        if archivo:
            df_csv = pd.read_csv(archivo)
            
            # --- NUEVA LÓGICA DE MAPEO DE FORMULARIO DE GOOGLE ---
            if 'Marca temporal' in df_csv.columns:
                mapeo_respuestas = {
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
                    "Sí, de manera constante": 2
                }

                cols = df_csv.columns
                df_procesado = pd.DataFrame()
                
                # Asignar sexo
                df_procesado['sexo'] = df_csv[cols[1]].map({"Hombre": "M", "Mujer": "F"}).fillna("Otro")
                
                # Calcular puntajes por dimensión (sumando las 3 preguntas respectivas)
                df_procesado['r'] = df_csv[cols[2]].map(mapeo_respuestas).fillna(0) + df_csv[cols[3]].map(mapeo_respuestas).fillna(0) + df_csv[cols[4]].map(mapeo_respuestas).fillna(0)
                df_procesado['i'] = df_csv[cols[5]].map(mapeo_respuestas).fillna(0) + df_csv[cols[6]].map(mapeo_respuestas).fillna(0) + df_csv[cols[7]].map(mapeo_respuestas).fillna(0)
                df_procesado['a'] = df_csv[cols[8]].map(mapeo_respuestas).fillna(0) + df_csv[cols[9]].map(mapeo_respuestas).fillna(0) + df_csv[cols[10]].map(mapeo_respuestas).fillna(0)
                df_procesado['s'] = df_csv[cols[11]].map(mapeo_respuestas).fillna(0) + df_csv[cols[12]].map(mapeo_respuestas).fillna(0) + df_csv[cols[13]].map(mapeo_respuestas).fillna(0)
                df_procesado['e'] = df_csv[cols[14]].map(mapeo_respuestas).fillna(0) + df_csv[cols[15]].map(mapeo_respuestas).fillna(0) + df_csv[cols[16]].map(mapeo_respuestas).fillna(0)
                df_procesado['c'] = df_csv[cols[17]].map(mapeo_respuestas).fillna(0) + df_csv[cols[18]].map(mapeo_respuestas).fillna(0) + df_csv[cols[19]].map(mapeo_respuestas).fillna(0)

                df_csv = df_procesado
            else:
                df_csv.columns = [str(c).lower() for c in df_csv.columns]
                
            cols_req = {"sexo", "r", "i", "a", "s", "e", "c"}
            if cols_req.issubset(set(df_csv.columns)):
                if st.button("Integrar datos al conjunto actual"):
                    df_csv["id"] = range(st.session_state["id_counter"], st.session_state["id_counter"] + len(df_csv))
                    st.session_state["id_counter"] += len(df_csv)
                    df_csv["fecha"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                    df_csv["dominante"] = df_csv.apply(calcular_dominante, axis=1)
                    st.session_state["df_datos"] = pd.concat([st.session_state["df_datos"], df_csv[list(st.session_state["df_datos"].columns)]], ignore_index=True)
                    if "df_filtrado" in st.session_state: del st.session_state["df_filtrado"]
                    st.success(f"{len(df_csv)} registros integrados.")
            else:
                st.error("El CSV no contiene las columnas requeridas o no se pudo mapear.")
                
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
                if "df_filtrado" in st.session_state: del st.session_state["df_filtrado"]
                st.success("Registro añadido.")
                
    with tab3:
        st.subheader("Simulación de Carga")
        st.write("Genera un conjunto basado en la tendencia de los datos actuales (si existen) o de forma aleatoria.")
        num_regs = st.number_input("Cantidad de registros", 100, 20000, 10000)
        if st.button("Generar Conjunto Sintético"):
            sexos = ["M", "F"]
            df_actual = st.session_state["df_datos"]
            
            # Crear diccionario para los nuevos datos
            nuevos_datos = {"sexo": np.random.choice(sexos, num_regs)}
            
            for dim in ["r", "i", "a", "s", "e", "c"]:
                if not df_actual.empty:
                    media = df_actual[dim].mean()
                    desv = df_actual[dim].std()
                    # Si la desviación es 0 o nula, añadir un poco de ruido
                    if pd.isna(desv) or desv == 0:
                        desv = 1.0
                    valores = np.random.normal(loc=media, scale=desv, size=num_regs)
                else:
                    # Fallback aleatorio
                    valores = np.random.uniform(0, 6, size=num_regs)
                
                # Restringir los valores entre 0 y 6 y redondear
                valores = np.clip(np.round(valores), 0, 6).astype(int)
                nuevos_datos[dim] = valores

            syn_df = pd.DataFrame(nuevos_datos)
            syn_df["id"] = range(st.session_state["id_counter"], st.session_state["id_counter"] + num_regs)
            st.session_state["id_counter"] += num_regs
            syn_df["fecha"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            syn_df["dominante"] = syn_df.apply(calcular_dominante, axis=1)
            
            # Acumular los datos en lugar de sobreescribir
            st.session_state["df_datos"] = pd.concat([st.session_state["df_datos"], syn_df], ignore_index=True)
            if "df_filtrado" in st.session_state: del st.session_state["df_filtrado"]
            st.success(f"{num_regs} registros sintéticos generados y añadidos exitosamente. Total: {len(st.session_state['df_datos'])}")

    st.markdown("---")
    st.write(f"**Total de registros actuales:** {len(st.session_state['df_datos'])}")
    st.dataframe(st.session_state["df_datos"].head(100), use_container_width=True)
    if st.button("Limpiar Datos"):
        st.session_state["df_datos"] = st.session_state["df_datos"].iloc[0:0]
        st.session_state["modelo_info"] = None
        st.rerun()

# --------------------------------------------------------------------------
# 2. DASHBOARD
# --------------------------------------------------------------------------
elif pestana == "Dashboard":
    st.header("Dashboard de Análisis")
    
    df = st.session_state["df_datos"].copy()
    
    if df.empty:
        st.info("El Dashboard está en blanco. Dirígete a 'Gestión de Datos' para cargar registros.")
    else:
        # Mapear a nombres completos
        df["dominante_nombre"] = df["dominante"].map(NOMBRES_DIMENSION)
        
        st.write("Aplica filtros para generar las estadísticas:")
        col1, col2 = st.columns(2)
        filtro_sexo = col1.multiselect("Sexo", df["sexo"].unique())
        filtro_dom = col2.multiselect("Dimensión Dominante", df["dominante_nombre"].unique())
        
        if filtro_sexo:
            df = df[df["sexo"].isin(filtro_sexo)]
        if filtro_dom:
            df = df[df["dominante_nombre"].isin(filtro_dom)]
            
        st.session_state["df_filtrado"] = df
        
        if not filtro_sexo and not filtro_dom:
            st.warning("⚠️ **Debe aplicar al menos un filtro** para habilitar la generación de estadísticas del modelo.")
            if st.toggle("Habilitar Estadística Básica (Descriptiva sin filtros)", False):
                st.write(f"Total registros en memoria: {len(df)}")
                st.bar_chart(df["dominante_nombre"].value_counts())
        else:
            st.success(f"Estadísticas activadas para {len(df)} registros filtrados.")
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"<div class='metric-box'><p>Registros Filtrados</p><h3>{len(df)}</h3></div>", unsafe_allow_html=True)
            with c2:
                dom_max = df["dominante_nombre"].mode()[0] if not df.empty else "N/A"
                st.markdown(f"<div class='metric-box'><p>Perfil Mayoritario</p><h3>{dom_max}</h3></div>", unsafe_allow_html=True)
            with c3:
                st.markdown(f"<div class='metric-box'><p>Motor Analítico</p><h3>GMM</h3></div>", unsafe_allow_html=True)
            
            with st.expander("ℹ️ ¿Qué es el modelo RIASEC?"):
                st.write("**R (Realista):** Prefieren actividades prácticas, trabajar con herramientas, máquinas o al aire libre.")
                st.write("**I (Investigativo):** Prefieren actividades analíticas, observar, investigar y resolver problemas complejos.")
                st.write("**A (Artístico):** Valoran la creatividad, la autoexpresión, el arte, la música y entornos no estructurados.")
                st.write("**S (Social):** Les gusta ayudar, enseñar, curar o servir a los demás. Tienen fuertes habilidades interpersonales.")
                st.write("**E (Emprendedor):** Prefieren persuadir, liderar o dirigir a otros para alcanzar metas organizacionales o económicas.")
                st.write("**C (Convencional):** Prefieren actividades organizadas, estructuradas, manejo de datos y rutinas claras.")

            st.markdown("---")
            cc1, cc2 = st.columns(2)
            with cc1:
                st.subheader("Distribución por Dimensión Dominante")
                fig_dom = px.pie(df, names="dominante_nombre", hole=0.4)
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
        
        opciones_cov = {
            "Completa (full)": "full", 
            "Atada (tied)": "tied", 
            "Diagonal (diag)": "diag", 
            "Esférica (spherical)": "spherical"
        }
        cov_type_label = col2.selectbox("Tipo de Covarianza", list(opciones_cov.keys()))
        cov_type = opciones_cov[cov_type_label]
        
        if st.button("Ejecutar Clustering GMM", type="primary"):
            with st.spinner("Ejecutando GMM en memoria..."):
                res = ejecutar_clustering_en_memoria(df, k, cov_type)
                
                df["cluster"] = res["etiquetas_asignadas"]
                df["perfil_cluster"] = df["cluster"].map(res["etiquetas_texto"])
                
                st.session_state["df_filtrado_clusterizado"] = df
                st.session_state["modelo_info"] = res
                
                st.success("Clustering completado. Las estadísticas y el modelo físico se han guardado.")
                
        if st.session_state["modelo_info"] is not None:
            res = st.session_state["modelo_info"]
            st.markdown("### Métricas del Modelo")
            
            with st.expander("ℹ️ ¿Cómo interpretar estas métricas?"):
                st.write("- **Silhouette Score:** Mide cuán similar es un objeto a su propio clúster frente a otros. Varía de -1 a 1 (valores cercanos a 1 son mejores).")
                st.write("- **BIC (Criterio de Información Bayesiano) y AIC:** Criterios para medir la calidad del modelo. **Valores más bajos indican un mejor modelo**, ya que penalizan modelos demasiado complejos.")
                
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
        df_completo = st.session_state["df_filtrado_clusterizado"]
        modelo_info = st.session_state["modelo_info"]
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Filtros del Reporte PDF")
        f_sexo = st.sidebar.multiselect("Filtrar por Sexo (PDF)", df_completo["sexo"].unique())
        f_cluster = st.sidebar.multiselect("Filtrar por Clúster (PDF)", df_completo["cluster"].unique())
        
        df_pdf = df_completo.copy()
        if f_sexo:
            df_pdf = df_pdf[df_pdf["sexo"].isin(f_sexo)]
        if f_cluster:
            df_pdf = df_pdf[df_pdf["cluster"].isin(f_cluster)]
            
        tab1, tab2 = st.tabs(["Tabla de Clasificación", "Previsualización y Descarga de PDF"])
        
        with tab1:
            st.write(f"Mostrando clasificación para {len(df_pdf)} estudiantes (con los filtros actuales).")
            st.dataframe(df_pdf[["id", "sexo", "dominante", "cluster", "perfil_cluster"]], use_container_width=True)
            
        with tab2:
            st.write("A continuación puedes visualizar y descargar el reporte (los filtros seleccionados aplican automáticamente al PDF):")
            
            # Generar el PDF en memoria para ambas funciones
            buffer_pdf = generar_reporte_pdf(df_pdf, modelo_info)
            
            st.download_button(
                label="📥 Descargar Reporte GMM PDF",
                data=buffer_pdf.getvalue(),
                file_name="Reporte_Clustering_RIASEC.pdf",
                mime="application/pdf",
                type="primary"
            )
            
            st.markdown("### Previsualización del Documento")
            base64_pdf = base64.b64encode(buffer_pdf.getvalue()).decode('utf-8')
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)

# --------------------------------------------------------------------------
# 5. HISTORIAL DE MODELOS
# --------------------------------------------------------------------------
elif pestana == "Historial de Modelos":
    st.header("Historial de Modelos Entrenados")
    st.write("Selecciona un modelo previamente guardado para aplicarlo sobre los datos que tienes actualmente cargados en memoria.")
    
    session = get_session()
    modelos = session.query(ModeloEntrenado).order_by(ModeloEntrenado.id.desc()).all()
    session.close()
    
    if not modelos:
        st.info("No hay modelos entrenados en el historial.")
    else:
        # Preparar data para mostrar
        data_modelos = []
        for m in modelos:
            data_modelos.append({
                "ID": m.id,
                "Fecha": m.fecha_creacion.strftime("%Y-%m-%d %H:%M") if m.fecha_creacion else "N/A",
                "Clústeres (k)": m.n_componentes,
                "BIC": round(m.bic, 1) if m.bic else "N/A",
                "Datos Originales": m.n_registros_entrenamiento
            })
            
        df_hist = pd.DataFrame(data_modelos)
        st.dataframe(df_hist, use_container_width=True)
        
        st.markdown("### Aplicar un Modelo Histórico")
        modelo_seleccionado = st.selectbox("Selecciona el ID del modelo a cargar", df_hist["ID"].tolist())
        
        if st.button("Aplicar Modelo Seleccionado"):
            df_actual = st.session_state.get("df_filtrado", st.session_state["df_datos"]).copy()
            if df_actual.empty:
                st.error("No hay datos cargados en memoria. Ve a la pestaña 'Gestión de Datos' para cargar datos primero.")
            else:
                with st.spinner("Cargando y aplicando modelo..."):
                    try:
                        res = cargar_y_aplicar_modelo(df_actual, modelo_seleccionado)
                        
                        df_actual["cluster"] = res["etiquetas_asignadas"]
                        df_actual["perfil_cluster"] = df_actual["cluster"].map(res["etiquetas_texto"])
                        
                        st.session_state["df_filtrado_clusterizado"] = df_actual
                        # Recreamos un diccionario parecido al de entrenamiento
                        st.session_state["modelo_info"] = {
                            "modelo_id": modelo_seleccionado,
                            "etiquetas_asignadas": res["etiquetas_asignadas"],
                            "etiquetas_texto": res["etiquetas_texto"],
                            "n_componentes": len(res["etiquetas_texto"]),
                            "silhouette": None, "bic": res.get("bic"), "aic": res.get("aic")
                        }
                        st.success(f"Modelo {modelo_seleccionado} cargado exitosamente.")
                        st.info("Ve a la pestaña 'Resultados y Reportes' para visualizar los resultados.")
                    except Exception as e:
                        st.error(f"Error al cargar el modelo: {str(e)}")
