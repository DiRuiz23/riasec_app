"""
clustering_service.py
Implementa la lógica del algoritmo de clustering utilizando Gaussian Mixture Model (GMM).
Maneja datos en memoria (Pandas DataFrame) y guarda solo los modelos y estadísticas
agregadas en la base de datos, cumpliendo con el requerimiento de no guardar registros crudos.

Mejoras aplicadas:
- Paralelización de encontrar_k_optimo con joblib.Parallel
- Retorno de la curva BIC/AIC completa para graficar
- Umbral de etiquetado configurable
- Random seed expuesto como parámetro
- Uso de managed_session para evitar fugas de BD
"""

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import json
import os
import joblib
from joblib import Parallel, delayed

from src.infrastructure.database.db import managed_session, get_session, ModeloEntrenado, EstadisticaClustering
from src.application.cuestionario_service import NOMBRES_DIMENSION

DIMENSIONES = ["r", "i", "a", "s", "e", "c"]


def _evaluar_gmm_para_k(X: np.ndarray, k: int, covariance_type: str, random_state: int) -> tuple[int, float, float]:
    """
    Función auxiliar para evaluar un único valor de k.
    Diseñada para ser ejecutada en paralelo.
    Retorna (k, bic, aic).
    """
    try:
        gmm = GaussianMixture(n_components=k, covariance_type=covariance_type, random_state=random_state)
        gmm.fit(X)
        return k, float(gmm.bic(X)), float(gmm.aic(X))
    except Exception:
        return k, np.inf, np.inf


def encontrar_k_optimo(
    df_vectores: pd.DataFrame,
    max_k: int = 12,
    covariance_type: str = "full",
    random_state: int = 42,
) -> tuple[int, dict]:
    """
    Evalúa múltiples valores de k utilizando GMM y el BIC en paralelo.
    
    Retorna:
        k_optimo (int): el k con el BIC más bajo.
        curva (dict): {'k': [...], 'bic': [...], 'aic': [...]} para graficar.
    """
    if len(df_vectores) < 2:
        return 2, {"k": [2], "bic": [0.0], "aic": [0.0]}

    X = df_vectores[DIMENSIONES].values
    limite = min(len(X), max_k + 1)
    ks = list(range(2, limite))

    # Evaluación paralela usando todos los núcleos disponibles
    resultados = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_evaluar_gmm_para_k)(X, k, covariance_type, random_state) for k in ks
    )

    curva = {"k": [], "bic": [], "aic": []}
    mejor_bic = np.inf
    k_optimo = 2

    for k, bic, aic in sorted(resultados, key=lambda x: x[0]):
        curva["k"].append(k)
        curva["bic"].append(bic)
        curva["aic"].append(aic)
        if bic < mejor_bic:
            mejor_bic = bic
            k_optimo = k

    return k_optimo, curva


def etiquetar_clusters(centroides: np.ndarray, umbral_diferencia: float = 0.75) -> dict:
    """
    Interpreta cada centroide (media de las 6 dimensiones dentro del cluster)
    y le asigna una etiqueta legible.
    
    Args:
        centroides: array con las medias por cluster.
        umbral_diferencia: si la diferencia entre la dimensión principal y la
            secundaria es <= este valor, se considera un perfil combinado.
            Por defecto 0.75 (calibrado según escala 0-6).
    """
    etiquetas = {}
    for idx, centro in enumerate(centroides):
        orden = sorted(zip(DIMENSIONES, centro), key=lambda x: x[1], reverse=True)
        principal, valor_principal = orden[0]
        secundaria, valor_secundaria = orden[1]

        if (valor_principal - valor_secundaria) <= umbral_diferencia:
            etiqueta = (
                f"Predominantemente {NOMBRES_DIMENSION[principal.upper()]}-"
                f"{NOMBRES_DIMENSION[secundaria.upper()]}"
            )
        else:
            etiqueta = f"Predominantemente {NOMBRES_DIMENSION[principal.upper()]}"
        etiquetas[idx] = etiqueta
    return etiquetas


def calcular_pca_2d(X: np.ndarray, etiquetas_cluster: np.ndarray) -> dict:
    """
    Reduce los datos a 2 dimensiones con PCA para visualización de clusters.
    Retorna {'x': [...], 'y': [...], 'cluster': [...], 'varianza_explicada': [...]}.
    """
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)
    return {
        "x": X_2d[:, 0].tolist(),
        "y": X_2d[:, 1].tolist(),
        "cluster": etiquetas_cluster.tolist(),
        "varianza_explicada": pca.explained_variance_ratio_.tolist(),
    }


def ejecutar_clustering_en_memoria(
    df_vectores: pd.DataFrame,
    k: int,
    covariance_type: str = "full",
    random_state: int = 42,
    umbral_diferencia: float = 0.75,
):
    """
    Ejecuta GMM sobre el dataframe en memoria y genera estadísticas.
    Guarda los resultados estadísticos (NO LOS DATOS CRUDOS) en la base de datos.
    Retorna el modelo entrenado y las etiquetas asignadas al dataframe.
    """
    if len(df_vectores) < k:
        raise ValueError(f"Se necesitan al menos {k} registros para {k} clusters.")

    X = df_vectores[DIMENSIONES].values

    modelo = GaussianMixture(
        n_components=k,
        covariance_type=covariance_type,
        random_state=random_state,
    )
    modelo.fit(X)

    etiquetas_cluster = modelo.predict(X)
    probabilidades = modelo.predict_proba(X)

    bic = float(modelo.bic(X))
    aic = float(modelo.aic(X))

    try:
        score_silueta = float(silhouette_score(X, etiquetas_cluster))
    except ValueError:
        score_silueta = None

    # Etiquetas cualitativas con umbral configurable
    etiquetas_texto = etiquetar_clusters(modelo.means_, umbral_diferencia=umbral_diferencia)

    # PCA 2D para visualización
    pca_data = calcular_pca_2d(X, etiquetas_cluster)

    # Estadísticas por clúster para guardar en BD
    estadisticas = []
    total_registros = len(X)
    for c_idx in range(k):
        indices_cluster = np.where(etiquetas_cluster == c_idx)[0]
        cantidad = len(indices_cluster)
        porcentaje = (cantidad / total_registros) * 100 if total_registros > 0 else 0

        promedios = X[indices_cluster].mean(axis=0) if cantidad > 0 else np.zeros(len(DIMENSIONES))

        patrones = {
            "etiqueta": etiquetas_texto[c_idx],
            "promedios_riasec": dict(zip(DIMENSIONES, promedios.tolist())),
        }
        estadisticas.append({
            "cluster_index": c_idx,
            "cantidad_elementos": cantidad,
            "porcentaje": float(porcentaje),
            "patrones_json": json.dumps(patrones),
        })

    # Guardar métricas y estadísticas en BD con managed_session
    with managed_session() as session:
        # Desactivar modelos anteriores
        session.query(ModeloEntrenado).update({ModeloEntrenado.activo: False})

        # Generar nombre y ruta de archivo
        timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        nombre_archivo = f"gmm_{k}c_{timestamp_str}.pkl"
        ruta_archivo = os.path.join("models", nombre_archivo)

        # Guardar el binario con joblib
        os.makedirs("models", exist_ok=True)
        joblib.dump(modelo, ruta_archivo)

        # Registrar nuevo modelo
        registro_modelo = ModeloEntrenado(
            algoritmo="GaussianMixture",
            n_componentes=k,
            covariance_type=covariance_type,
            n_registros_entrenamiento=total_registros,
            bic=bic,
            aic=aic,
            activo=True,
            ruta_archivo=ruta_archivo,
            fecha_creacion=datetime.utcnow(),
            fecha_modificacion=datetime.utcnow(),
        )
        session.add(registro_modelo)
        session.flush()

        # Guardar estadísticas agregadas por cluster
        for est in estadisticas:
            session.add(EstadisticaClustering(
                modelo_id=registro_modelo.id,
                cluster_index=est["cluster_index"],
                cantidad_elementos=est["cantidad_elementos"],
                porcentaje=est["porcentaje"],
                patrones_json=est["patrones_json"],
            ))

        modelo_id = registro_modelo.id

    return {
        "modelo": modelo,
        "modelo_id": modelo_id,
        "etiquetas_asignadas": etiquetas_cluster.tolist(),
        "probabilidades": probabilidades.tolist(),
        "etiquetas_texto": etiquetas_texto,
        "silhouette": score_silueta,
        "bic": bic,
        "aic": aic,
        "n_registros": total_registros,
        "n_componentes": k,
        "pca_data": pca_data,
    }


def cargar_y_aplicar_modelo(df_vectores: pd.DataFrame, modelo_id: int) -> dict:
    """
    Carga un modelo entrenado desde el binario en disco y lo aplica a los datos proporcionados.
    """
    with managed_session() as session:
        registro_modelo = session.query(ModeloEntrenado).filter_by(id=modelo_id).first()
        if not registro_modelo or not registro_modelo.ruta_archivo or not os.path.exists(registro_modelo.ruta_archivo):
            raise FileNotFoundError("No se encontró el archivo físico del modelo.")

        modelo = joblib.load(registro_modelo.ruta_archivo)
        bic_historico = registro_modelo.bic
        aic_historico = registro_modelo.aic
        n_componentes = registro_modelo.n_componentes

        estadisticas = session.query(EstadisticaClustering).filter_by(modelo_id=modelo_id).all()
        etiquetas_texto = {}
        for est in estadisticas:
            datos = json.loads(est.patrones_json)
            etiquetas_texto[est.cluster_index] = datos.get("etiqueta", f"Clúster {est.cluster_index}")

    X = df_vectores[DIMENSIONES].values
    etiquetas_cluster = modelo.predict(X)
    probabilidades = modelo.predict_proba(X)
    pca_data = calcular_pca_2d(X, etiquetas_cluster)

    return {
        "etiquetas_asignadas": etiquetas_cluster.tolist(),
        "probabilidades": probabilidades.tolist(),
        "etiquetas_texto": etiquetas_texto,
        "bic": bic_historico,
        "aic": aic_historico,
        "n_componentes": n_componentes,
        "pca_data": pca_data,
    }
