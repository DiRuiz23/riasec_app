"""
app.py
Interfaz Streamlit del Perfil de Personalidad Vocacional RIASEC.
Pestañas: Cuestionario | Visualización y filtros | Estadística descriptiva |
Entrenamiento | Resultados | Comparativa entre clústeres | Metadatos del modelo | Descargas.

Ejecutar con: streamlit run app.py
"""

import json
import io
import datetime
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

st.set_page_config(page_title="RIASEC - Análisis No Supervisado", layout="wide")
init_db()

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


# --------------------------------------------------------------------------
# Barra lateral: navegacion
# --------------------------------------------------------------------------
st.sidebar.title("RIASEC · Unidad IV")
pestana = st.sidebar.radio(
    "Ir a:",
    ["1. Responder cuestionario", "2. Carga y visualización", "3. Estadística descriptiva",
     "4. Entrenamiento del modelo", "5. Resultados", "6. Comparativa de clústeres",
     "7. Metadatos del modelo", "8. Descargas"]
)

# --------------------------------------------------------------------------
# 1. CUESTIONARIO
# --------------------------------------------------------------------------
if pestana == "1. Responder cuestionario":
    st.title("Cuestionario de Perfil Vocacional (RIASEC)")
    st.caption("18 preguntas ponderadas + 1 pregunta demográfica (no ponderada).")

    with st.form("form_cuestionario"):
        sexo = st.radio("Selecciona tu sexo", ["Hombre", "Mujer"])
        edad = st.number_input("Edad", min_value=10, max_value=99, value=20)
        carrera = st.text_input("¿Qué carrera o área te interesa? (opcional)")

        respuestas_peso = {}
        for numero, dimension, texto, opciones in PREGUNTAS:
            if dimension == "DEMOGRAFICA":
                continue
            etiquetas_opcion = [o[0] for o in opciones]
            seleccion = st.radio(f"{numero}. {texto}", etiquetas_opcion, key=f"p{numero}")
            peso = next(o[1] for o in opciones if o[0] == seleccion)
            respuestas_peso[numero] = peso

        enviado = st.form_submit_button("Enviar respuestas")

    if enviado:
        session = get_session()
        try:
            usuario = Usuario(sexo=sexo, edad=edad, carrera_interes=carrera or None,
                               fecha_registro=datetime.datetime.utcnow())
            session.add(usuario)
            session.flush()

            preguntas_bd = {p.numero: p for p in session.query(Pregunta).all()}
            for numero, dimension, texto, opciones in PREGUNTAS:
                pregunta_bd = preguntas_bd[numero]
                if dimension == "DEMOGRAFICA":
                    opcion_bd = next(o for o in pregunta_bd.opciones
                                     if o.texto_opcion == (sexo))
                    session.add(Respuesta(usuario_id=usuario.id, pregunta_id=pregunta_bd.id,
                                           opcion_id=opcion_bd.id, peso_obtenido=None))
                    continue
                peso = respuestas_peso[numero]
                opcion_bd = next(o for o in pregunta_bd.opciones if o.peso == peso)
                session.add(Respuesta(usuario_id=usuario.id, pregunta_id=pregunta_bd.id,
                                       opcion_id=opcion_bd.id, peso_obtenido=peso))

            vector = agregar_vector(respuestas_peso)
            session.add(VectorRiasec(usuario_id=usuario.id, **{k.lower(): v for k, v in vector.items()}))
            session.commit()
            st.success(f"Respuestas guardadas (usuario #{usuario.id}). Vector RIASEC: {vector}")
        finally:
            session.close()

# --------------------------------------------------------------------------
# 2. CARGA Y VISUALIZACION
# --------------------------------------------------------------------------
elif pestana == "2. Carga y visualización":
    st.title("Carga de datos y visualización")

    st.subheader("Subir dataset externo (CSV)")
    st.caption("Columnas esperadas: sexo, edad, carrera_interes, R, I, A, S, E, C "
               "(0-6 cada dimensión). El algoritmo no cambia, solo los datos de entrada.")
    archivo = st.file_uploader("Selecciona un archivo CSV", type=["csv"])

    if archivo is not None:
        df_nuevo = pd.read_csv(archivo)
        columnas_esperadas = {"sexo", "edad", "R", "I", "A", "S", "E", "C"}
        if not columnas_esperadas.issubset(set(df_nuevo.columns)):
            st.error(f"El archivo debe contener al menos las columnas: {columnas_esperadas}")
        else:
            st.dataframe(df_nuevo.head())
            if st.button("Confirmar e insertar en la base de datos"):
                session = get_session()
                try:
                    for _, fila in df_nuevo.iterrows():
                        usuario = Usuario(
                            sexo=fila.get("sexo"), edad=fila.get("edad"),
                            carrera_interes=fila.get("carrera_interes"),
                            fecha_registro=datetime.datetime.utcnow(),
                        )
                        session.add(usuario)
                        session.flush()
                        session.add(VectorRiasec(
                            usuario_id=usuario.id, r=int(fila["R"]), i=int(fila["I"]),
                            a=int(fila["A"]), s=int(fila["S"]), e=int(fila["E"]), c=int(fila["C"]),
                        ))
                    session.commit()
                    st.success(f"Se insertaron {len(df_nuevo)} registros nuevos.")
                finally:
                    session.close()

    st.subheader("Datos actuales en la base de datos")
    df = usuarios_a_dataframe()
    if df.empty:
        st.info("Aún no hay registros. Ejecuta seed.py o responde el cuestionario.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            filtro_sexo = st.multiselect("Filtrar por sexo", options=df["sexo"].dropna().unique().tolist())
        with col2:
            rango_edad = st.slider("Rango de edad", int(df["edad"].min()), int(df["edad"].max()),
                                    (int(df["edad"].min()), int(df["edad"].max())))
        with col3:
            fecha_min = df["fecha_registro"].min()
            fecha_max = df["fecha_registro"].max()
            rango_fecha = st.date_input("Rango de fecha de registro",
                                         (fecha_min.date(), fecha_max.date()))

        df_filtrado = df.copy()
        if filtro_sexo:
            df_filtrado = df_filtrado[df_filtrado["sexo"].isin(filtro_sexo)]
        df_filtrado = df_filtrado[
            (df_filtrado["edad"] >= rango_edad[0]) & (df_filtrado["edad"] <= rango_edad[1])
        ]
        if isinstance(rango_fecha, tuple) and len(rango_fecha) == 2:
            df_filtrado = df_filtrado[
                (df_filtrado["fecha_registro"].dt.date >= rango_fecha[0]) &
                (df_filtrado["fecha_registro"].dt.date <= rango_fecha[1])
            ]

        st.dataframe(df_filtrado, use_container_width=True)
        st.session_state["df_filtrado"] = df_filtrado

# --------------------------------------------------------------------------
# 3. ESTADISTICA DESCRIPTIVA
# --------------------------------------------------------------------------
elif pestana == "3. Estadística descriptiva":
    st.title("Estadística descriptiva (algoritmos propios)")
    df = usuarios_a_dataframe()
    if df.empty:
        st.info("No hay datos suficientes.")
    else:
        vectores = df[["R", "I", "A", "S", "E", "C"]].rename(columns=str.lower).to_dict("records")
        resumen = resumen_dimensiones(vectores)
        st.dataframe(pd.DataFrame(resumen).T, use_container_width=True)

        st.subheader("Distribución por categoría")
        campo = st.selectbox("Selecciona campo categórico", ["sexo", "carrera_interes"])
        conteo, porcentaje = distribucion_por_categoria(df.to_dict("records"), campo)
        col1, col2 = st.columns(2)
        with col1:
            st.write("Conteo absoluto")
            st.bar_chart(pd.Series(conteo))
        with col2:
            st.write("Porcentaje")
            st.dataframe(pd.Series(porcentaje).rename("porcentaje (%)"))

# --------------------------------------------------------------------------
# 4. ENTRENAMIENTO
# --------------------------------------------------------------------------
elif pestana == "4. Entrenamiento del modelo":
    st.title("Entrenamiento del modelo (Gaussian Mixture Model)")
    st.write("El modelo agrupa los vectores [R,I,A,S,E,C] de forma no supervisada "
             "y despues etiqueta cada clúster según su centroide dominante.")

    col1, col2 = st.columns(2)
    with col1:
        n_componentes = st.number_input("Número de componentes (clústeres)", min_value=2, max_value=12, value=6)
    with col2:
        covariance_type = st.selectbox("Tipo de covarianza", ["full", "tied", "diag", "spherical"])

    if st.button("Entrenar modelo"):
        with st.spinner("Entrenando..."):
            try:
                resultado = entrenar_modelo(n_componentes=n_componentes, covariance_type=covariance_type)
                st.success(f"Modelo #{resultado['modelo_id']} entrenado con "
                           f"{resultado['n_registros']} registros.")
                st.json({
                    "silhouette_score": resultado["silhouette"],
                    "BIC": resultado["bic"],
                    "AIC": resultado["aic"],
                    "etiquetas_por_cluster": resultado["etiquetas"],
                })
            except ValueError as e:
                st.error(str(e))

# --------------------------------------------------------------------------
# 5. RESULTADOS
# --------------------------------------------------------------------------
elif pestana == "5. Resultados":
    st.title("Resultados del clustering")
    modelo, registro = cargar_modelo_activo()
    if modelo is None:
        st.info("Aún no hay ningún modelo entrenado.")
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

        st.caption(f"Varianza explicada por las 2 componentes: "
                   f"{round(sum(varianza) * 100, 1)}%")
        fig = px.scatter(df_plot, x="PCA_1", y="PCA_2", color="etiqueta_riasec",
                          hover_data=["usuario_id"], title="Clústeres proyectados (PCA 6D → 2D)")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_resultados, use_container_width=True)

# --------------------------------------------------------------------------
# 6. COMPARATIVA DE CLUSTERES
# --------------------------------------------------------------------------
elif pestana == "6. Comparativa de clústeres":
    st.title("Comparativa entre clústeres (perfil radar)")
    modelo, registro = cargar_modelo_activo()
    if modelo is None:
        st.info("Aún no hay ningún modelo entrenado.")
    else:
        etiquetas = etiquetar_clusters_local = None
        centroides = modelo.means_
        from entrenamiento import etiquetar_clusters
        etiquetas = etiquetar_clusters(centroides)

        fig = go.Figure()
        for idx, centro in enumerate(centroides):
            fig.add_trace(go.Scatterpolar(
                r=list(centro) + [centro[0]],
                theta=DIMENSIONES + [DIMENSIONES[0]],
                fill="toself",
                name=f"Clúster {idx}: {etiquetas[idx]}",
            ))
        fig.update_layout(title="Medias por dimensión (R,I,A,S,E,C) por clúster")
        st.plotly_chart(fig, use_container_width=True)

        tabla_centroides = pd.DataFrame(centroides, columns=DIMENSIONES)
        tabla_centroides.insert(0, "etiqueta", [etiquetas[i] for i in range(len(centroides))])
        st.dataframe(tabla_centroides, use_container_width=True)

# --------------------------------------------------------------------------
# 7. METADATOS DEL MODELO
# --------------------------------------------------------------------------
elif pestana == "7. Metadatos del modelo":
    st.title("Ficha de metadatos del algoritmo")
    session = get_session()
    registros = session.query(ModeloEntrenado).order_by(ModeloEntrenado.fecha_entrenamiento.desc()).all()
    session.close()
    if not registros:
        st.info("Aún no hay modelos entrenados.")
    else:
        tabla = pd.DataFrame([{
            "id": r.id, "algoritmo": r.algoritmo, "n_componentes": r.n_componentes,
            "covariance_type": r.covariance_type, "n_registros": r.n_registros_entrenamiento,
            "silhouette": r.silhouette_score, "BIC": r.bic, "AIC": r.aic,
            "fecha": r.fecha_entrenamiento, "activo": bool(r.activo),
        } for r in registros])
        st.dataframe(tabla, use_container_width=True)

# --------------------------------------------------------------------------
# 8. DESCARGAS
# --------------------------------------------------------------------------
elif pestana == "8. Descargas":
    st.title("Descargas")

    st.subheader("Datos cuantitativos (filtrados)")
    df_filtrado = st.session_state.get("df_filtrado", usuarios_a_dataframe())
    csv_buffer = io.StringIO()
    df_filtrado.to_csv(csv_buffer, index=False)
    st.download_button("Descargar CSV de datos filtrados", data=csv_buffer.getvalue(),
                        file_name="riasec_datos_filtrados.csv", mime="text/csv")

    st.subheader("Reporte cualitativo (interpretación)")
    modelo, registro = cargar_modelo_activo()
    if modelo is not None:
        from entrenamiento import etiquetar_clusters
        etiquetas = etiquetar_clusters(modelo.means_)
        lineas = ["REPORTE CUALITATIVO - PERFIL VOCACIONAL RIASEC", "=" * 50, ""]
        for idx, etiqueta in etiquetas.items():
            centro = modelo.means_[idx]
            lineas.append(f"Clúster {idx}: {etiqueta}")
            lineas.append(f"  Perfil promedio (R,I,A,S,E,C): {[round(v, 2) for v in centro]}")
            lineas.append("")
        texto_reporte = "\n".join(lineas)
        st.text_area("Vista previa", texto_reporte, height=250)
        st.download_button("Descargar reporte cualitativo (.txt)", data=texto_reporte,
                            file_name="riasec_reporte_cualitativo.txt", mime="text/plain")
    else:
        st.info("Entrena un modelo primero para generar el reporte cualitativo.")
