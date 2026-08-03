import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator

from db import get_session, ModeloEntrenado, ResultadoClustering, Dataset, Usuario, VectorRiasec
from preprocesamiento import construir_pipeline, detectar_tipos_columnas
from cuestionario import DIMENSIONES, NOMBRES_DIMENSION

CARPETA_MODELOS = "modelos"
os.makedirs(CARPETA_MODELOS, exist_ok=True)

def sugerir_parametros_dbscan(X_prep):
    """
    Estima eps (con KneeLocator) y min_samples para DBSCAN dado un dataset preprocesado.
    """
    n_samples, n_features = X_prep.shape
    min_samples = max(3, 2 * n_features)
    
    # K-vecinos para distancia a la (min_samples-1)-esima distancia
    k = min_samples
    if n_samples < k:
        return {"eps": 0.5, "min_samples": n_samples}
        
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X_prep)
    distances, _ = nn.kneighbors(X_prep)
    
    k_distances = np.sort(distances[:, -1])
    
    kneedle = KneeLocator(range(len(k_distances)), k_distances, S=1.0, curve="convex", direction="increasing")
    knee_point = kneedle.knee
    eps = k_distances[knee_point] if knee_point else 0.5
    
    return {"eps": float(eps), "min_samples": min_samples}

def etiquetar_clusters_dinamico(modelo_o_centroides, n_clusters, algoritmo, X_columns):
    """Genera etiquetas numericas para modelos genericos, o legibles para RIASEC."""
    etiquetas = {}
    is_riasec = (len(X_columns) == 6 and set(X_columns) == {"r", "i", "a", "s", "e", "c"})
    
    if algoritmo == "DBSCAN":
        # DBSCAN no tiene centroides y puede tener cluster -1 (ruido)
        # Los clusters van de -1 a n_clusters-2 (aprox)
        pass # se etiquetaran en base a los ids reales despues
        
    for idx in range(-1, n_clusters):
        if idx == -1:
            etiquetas[idx] = "Ruido (Outliers)"
        else:
            if is_riasec and hasattr(modelo_o_centroides, "cluster_centers_"):
                # Si K-Means
                centro = modelo_o_centroides.cluster_centers_[idx]
            elif is_riasec and hasattr(modelo_o_centroides, "means_"):
                # Si GMM
                centro = modelo_o_centroides.means_[idx]
            else:
                centro = None
                
            if centro is not None:
                orden = sorted(zip(X_columns, centro), key=lambda x: x[1], reverse=True)
                principal, valor_principal = orden[0]
                secundaria, valor_secundaria = orden[1]
                if (valor_principal - valor_secundaria) <= 0.75:
                    etiqueta = f"Predominantemente {NOMBRES_DIMENSION[principal.upper()]}-{NOMBRES_DIMENSION[secundaria.upper()]}"
                else:
                    etiqueta = f"Predominantemente {NOMBRES_DIMENSION[principal.upper()]}"
                etiquetas[idx] = etiqueta
            else:
                etiquetas[idx] = f"Cluster {idx}"
                
    return etiquetas

def entrenar_modelo(n_componentes: int = 6, covariance_type: str = "full", dataset_id: int = 1):
    """
    Compatibilidad con la interfaz anterior de la app Streamlit.
    Entrena un modelo GMM sobre los vectores existentes en la base de datos.
    """
    session = get_session()
    try:
        registros = session.query(VectorRiasec).all()
        if not registros:
            raise ValueError("No hay registros de vectores RIASEC en la base de datos para entrenar el modelo.")

        df = pd.DataFrame([
            {
                "R": r.r, "I": r.i, "A": r.a, "S": r.s, "E": r.e, "C": r.c,
                "usuario_id": r.usuario_id,
            }
            for r in registros
        ])

        config = {
            "algoritmo": "GaussianMixture",
            "columnas_usadas": ["R", "I", "A", "S", "E", "C"],
            "param_n_components": n_componentes,
            "param_covariance_type": covariance_type,
        }
        return entrenar_modelo_generico(df, dataset_id, config)
    finally:
        session.close()


def obtener_matriz_entrenamiento(session):
    """
    Devuelve los IDs de usuario y la matriz de entrenamiento usada por la app.
    """
    registros = session.query(VectorRiasec).order_by(VectorRiasec.usuario_id).all()
    if not registros:
        return [], np.empty((0, len(DIMENSIONES)), dtype=float)

    usuario_ids = [r.usuario_id for r in registros]
    X = np.array([[r.r, r.i, r.a, r.s, r.e, r.c] for r in registros], dtype=float)
    return usuario_ids, X


def etiquetar_clusters(centroides, X_columns=None):
    """Etiqueta los centroides usando los nombres reales de columna del modelo."""
    if isinstance(centroides, pd.DataFrame):
        if X_columns is None:
            X_columns = list(centroides.columns)
        centroides_array = centroides.to_numpy()
    else:
        centroides_array = np.asarray(centroides)
        if X_columns is None:
            if centroides_array.ndim > 1 and centroides_array.shape[1] == len(DIMENSIONES):
                X_columns = DIMENSIONES
            else:
                X_columns = [f"dim_{idx}" for idx in range(centroides_array.shape[1])] if centroides_array.ndim > 1 else ["valor"]

    etiquetas = {}
    for idx, centro in enumerate(centroides_array):
        orden = sorted(zip(X_columns, centro), key=lambda x: x[1], reverse=True)
        principal, valor_principal = orden[0]
        secundaria, valor_secundaria = orden[1]

        def _etiqueta_columna(col):
            if isinstance(col, str):
                col_upper = col.upper()
                return NOMBRES_DIMENSION.get(col_upper, col)
            return str(col)

        principal_label = _etiqueta_columna(principal)
        secundaria_label = _etiqueta_columna(secundaria)

        if (valor_principal - valor_secundaria) <= 0.75:
            etiqueta = f"Predominantemente {principal_label}-{secundaria_label}"
        else:
            etiqueta = f"Predominantemente {principal_label}"
        etiquetas[idx] = etiqueta
    return etiquetas


def resolver_columnas_usadas(df: pd.DataFrame, config: dict):
    """Resuelve qué columnas usar sin rechazar el dataset si algunas no existen."""
    solicitadas = config.get("columnas_usadas", list(df.columns))
    if not solicitadas:
        return list(df.columns)

    existentes = [col for col in solicitadas if col in df.columns]
    if existentes:
        return existentes

    return list(df.columns)


def entrenar_modelo_generico(df: pd.DataFrame, dataset_id: int, config: dict):
    """
    Entrena un modelo de clustering sobre un DataFrame usando un pipeline genérico.
    config debe contener:
    - algoritmo: "GaussianMixture", "KMeans", "Agglomerative", "DBSCAN"
    - columnas_usadas: list
    - param_n_components (para GMM, KMeans, Agglomerative)
    - param_eps, param_min_samples (para DBSCAN)
    - param_covariance_type (para GMM)
    - param_linkage (para Agglomerative)
    """
    session = get_session()
    try:
        columnas_usadas = resolver_columnas_usadas(df, config)
        if df[columnas_usadas].isnull().values.any():
            raise ValueError("El dataset contiene valores nulos en las columnas seleccionadas y el pipeline actual requiere imputación o que no haya nulos.")

        tipos = detectar_tipos_columnas(df[columnas_usadas])
        preprocessor = construir_pipeline(tipos["numericas"], tipos["categoricas"])
        
        # Ajustar pipeline y transformar
        X_prep = preprocessor.fit_transform(df[columnas_usadas])
        
        n_registros = len(X_prep)
        algoritmo = config.get("algoritmo", "GaussianMixture")
        
        # Validar n_components
        n_comp = config.get("param_n_components", 2)
        if algoritmo != "DBSCAN" and n_registros < n_comp:
             raise ValueError(f"Se necesitan al menos {n_comp} registros (hay {n_registros}).")
        if algoritmo != "DBSCAN" and n_comp < 2:
             raise ValueError("El número de componentes debe ser >= 2.")
             
        modelo = None
        if algoritmo == "GaussianMixture":
            modelo = GaussianMixture(n_components=n_comp, covariance_type=config.get("param_covariance_type", "full"), reg_covar=1e-6, random_state=42)
        elif algoritmo == "KMeans":
            modelo = KMeans(n_clusters=n_comp, random_state=42, n_init="auto")
        elif algoritmo == "Agglomerative":
            modelo = AgglomerativeClustering(n_clusters=n_comp, linkage=config.get("param_linkage", "ward"))
        elif algoritmo == "DBSCAN":
            modelo = DBSCAN(eps=config.get("param_eps", 0.5), min_samples=config.get("param_min_samples", 5))
        else:
            raise ValueError(f"Algoritmo {algoritmo} no soportado.")
            
        # Entrenar
        etiquetas_cluster = modelo.fit_predict(X_prep)
        
        # Metricas
        n_clusters_reales = len(set(etiquetas_cluster)) - (1 if -1 in etiquetas_cluster else 0)
        try:
            score_silueta = float(silhouette_score(X_prep, etiquetas_cluster)) if n_clusters_reales >= 2 else None
        except ValueError:
            score_silueta = None
            
        bic, aic = None, None
        if algoritmo == "GaussianMixture":
            bic = float(modelo.bic(X_prep))
            aic = float(modelo.aic(X_prep))
            
        etiquetas_texto = etiquetar_clusters_dinamico(modelo, max(etiquetas_cluster)+1, algoritmo, columnas_usadas)
        
        fecha = datetime.utcnow()
        nombre_archivo = f"{algoritmo}_{fecha.strftime('%Y%m%d_%H%M%S')}.pkl"
        ruta = os.path.join(CARPETA_MODELOS, nombre_archivo)
        
        artefacto = {
            "pipeline": preprocessor,
            "modelo": modelo,
            "columnas": columnas_usadas,
            "dataset_id": dataset_id,
            "metadata": {
                "algoritmo": algoritmo,
                "fecha": fecha.isoformat(),
                "metricas": {"silhouette": score_silueta, "bic": bic, "aic": aic},
                "hiperparametros": config
            }
        }
        joblib.dump(artefacto, ruta)
        
        # Gestion activa: desactivar todos los del mismo dataset_id
        session.query(ModeloEntrenado).filter_by(dataset_id=dataset_id).update({"activo": 0})
        
        registro_modelo = ModeloEntrenado(
            fecha_entrenamiento=fecha,
            algoritmo=algoritmo,
            n_componentes=n_comp if algoritmo != "DBSCAN" else n_clusters_reales,
            covariance_type=config.get("param_covariance_type", "N/A"),
            n_registros_entrenamiento=n_registros,
            silhouette_score=score_silueta,
            bic=bic,
            aic=aic,
            ruta_archivo=ruta,
            activo=1,
            dataset_id=dataset_id
        )
        session.add(registro_modelo)
        session.flush()
        
        # En una app real aqui se guardarian los resultados, pero como el dataframe no necesariamente
        # tiene id_usuario mapeado a la bd "usuarios", retornaremos los labels.
        # Para mantener compatibilidad si es el legacy dataset:
        if dataset_id == 1 and 'id' in df.columns:
            for i, uid in enumerate(df['id']):
                cid = int(etiquetas_cluster[i])
                prob_json = "{}"
                if algoritmo == "GaussianMixture":
                    prob_json = json.dumps(modelo.predict_proba(X_prep[i:i+1])[0].tolist())
                session.add(ResultadoClustering(
                    usuario_id=int(uid),
                    modelo_id=registro_modelo.id,
                    cluster_id=cid,
                    etiqueta_riasec=etiquetas_texto.get(cid, str(cid)),
                    probabilidades_json=prob_json
                ))
                
        session.commit()
        
        return {
            "modelo_id": registro_modelo.id,
            "ruta_archivo": ruta,
            "silhouette": score_silueta,
            "bic": bic,
            "aic": aic,
            "n_registros": n_registros,
            "etiquetas": etiquetas_texto,
            "labels": etiquetas_cluster,
            "X_prep": X_prep
        }
    finally:
        session.close()

def reentrenar_modelo(modelo_id: int, nuevos_datos: pd.DataFrame):
    """
    Recupera configuración de un modelo previo y reentrena.
    """
    session = get_session()
    try:
        registro = session.query(ModeloEntrenado).filter_by(id=modelo_id).first()
        if not registro:
            raise ValueError(f"No se encontró el modelo ID {modelo_id}")
            
        artefacto = joblib.load(registro.ruta_archivo)
        columnas_esperadas = artefacto["columnas"]
        config = artefacto["metadata"]["hiperparametros"]
        # Inyectar el mismo dataset_id o el actual
        # Suponiendo que nuevos_datos viene con un dataset_id
        # Pero dejaremos que la UI decida el dataset_id al llamar a entrenar_modelo_generico
        return config, columnas_esperadas
    finally:
        session.close()

def cargar_modelo_activo(dataset_id=1):
    session = get_session()
    try:
        registro = session.query(ModeloEntrenado).filter_by(activo=1, dataset_id=dataset_id).order_by(
            ModeloEntrenado.fecha_entrenamiento.desc()
        ).first()
        if registro is None:
            return None, None
        artefacto = joblib.load(registro.ruta_archivo)
        return artefacto, registro
    finally:
        session.close()

def proyeccion_pca_2d(X):
    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(X), pca.explained_variance_ratio_
