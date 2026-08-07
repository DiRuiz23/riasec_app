"""
clustering_service.py
Implementa la lógica del algoritmo de clustering utilizando Gaussian Mixture Model (GMM).
Maneja datos en memoria (Pandas DataFrame) y guarda solo los modelos y estadísticas
agregadas en la base de datos, cumpliendo con el requerimiento de no guardar registros crudos.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import json

from src.infrastructure.database.db import get_session, ModeloEntrenado, EstadisticaClustering
from src.application.cuestionario_service import NOMBRES_DIMENSION

DIMENSIONES = ["r", "i", "a", "s", "e", "c"]

def encontrar_k_optimo(df_vectores: pd.DataFrame, max_k: int = 12, covariance_type: str = "full"):
    """
    Evalúa múltiples valores de k utilizando GMM y el Criterio de Información Bayesiano (BIC).
    Retorna el k con el BIC más bajo (mejor ajuste penalizando complejidad).
    """
    if len(df_vectores) < 2:
        return 2

    X = df_vectores[DIMENSIONES].values
    mejor_bic = np.inf
    k_optimo = 2
    
    limite = min(len(X), max_k + 1)
    for k in range(2, limite):
        gmm = GaussianMixture(n_components=k, covariance_type=covariance_type, random_state=42)
        gmm.fit(X)
        bic = gmm.bic(X)
        if bic < mejor_bic:
            mejor_bic = bic
            k_optimo = k
            
    return k_optimo


def etiquetar_clusters(centroides):
    """
    Interpreta cada centroide (media de las 6 dimensiones dentro del cluster)
    y le asigna una etiqueta legible.
    """
    etiquetas = {}
    for idx, centro in enumerate(centroides):
        orden = sorted(zip(DIMENSIONES, centro), key=lambda x: x[1], reverse=True)
        principal, valor_principal = orden[0]
        secundaria, valor_secundaria = orden[1]
        
        if (valor_principal - valor_secundaria) <= 0.75:
            etiqueta = (f"Predominantemente {NOMBRES_DIMENSION[principal.upper()]}-"
                        f"{NOMBRES_DIMENSION[secundaria.upper()]}")
        else:
            etiqueta = f"Predominantemente {NOMBRES_DIMENSION[principal.upper()]}"
        etiquetas[idx] = etiqueta
    return etiquetas


def ejecutar_clustering_en_memoria(df_vectores: pd.DataFrame, k: int, covariance_type: str = "full"):
    """
    Ejecuta GMM sobre el dataframe en memoria y genera estadísticas.
    Guarda los resultados estadísticos (NO LOS DATOS CRUDOS) en la base de datos.
    Retorna el modelo entrenado y las etiquetas asignadas al dataframe.
    """
    if len(df_vectores) < k:
        raise ValueError(f"Se necesitan al menos {k} registros para {k} clusters.")

    X = df_vectores[DIMENSIONES].values
    
    modelo = GaussianMixture(n_components=k, covariance_type=covariance_type, random_state=42)
    modelo.fit(X)
    
    etiquetas_cluster = modelo.predict(X)
    probabilidades = modelo.predict_proba(X)
    
    bic = float(modelo.bic(X))
    aic = float(modelo.aic(X))
    
    try:
        score_silueta = float(silhouette_score(X, etiquetas_cluster))
    except ValueError:
        score_silueta = None
        
    # Obtener etiquetas cualitativas
    etiquetas_texto = etiquetar_clusters(modelo.means_)
    
    # Calcular estadísticas por clúster para guardarlas
    estadisticas = []
    total_registros = len(X)
    for c_idx in range(k):
        indices_cluster = np.where(etiquetas_cluster == c_idx)[0]
        cantidad = len(indices_cluster)
        porcentaje = (cantidad / total_registros) * 100 if total_registros > 0 else 0
        
        if cantidad > 0:
            promedios = X[indices_cluster].mean(axis=0)
        else:
            promedios = np.zeros(len(DIMENSIONES))
            
        patrones = {
            "etiqueta": etiquetas_texto[c_idx],
            "promedios_riasec": dict(zip(DIMENSIONES, promedios.tolist()))
        }
        
        estadisticas.append({
            "cluster_index": c_idx,
            "cantidad_elementos": cantidad,
            "porcentaje": float(porcentaje),
            "patrones_json": json.dumps(patrones)
        })

    # Guardar métricas y estadísticas agregadas en Base de Datos
    session = get_session()
    try:
        # Desactivar anteriores
        session.query(ModeloEntrenado).update({ModeloEntrenado.activo: False})
        
        # Registrar nuevo modelo
        registro_modelo = ModeloEntrenado(
            algoritmo="GaussianMixture",
            n_componentes=k,
            covariance_type=covariance_type,
            n_registros_entrenamiento=total_registros,
            bic=bic,
            aic=aic,
            activo=True,
            fecha_creacion=datetime.utcnow(),
            fecha_modificacion=datetime.utcnow()
        )
        session.add(registro_modelo)
        session.flush()
        
        # Guardar solo las estadísticas agregadas por cluster
        for est in estadisticas:
            ses_est = EstadisticaClustering(
                modelo_id=registro_modelo.id,
                cluster_index=est["cluster_index"],
                cantidad_elementos=est["cantidad_elementos"],
                porcentaje=est["porcentaje"],
                patrones_json=est["patrones_json"]
            )
            session.add(ses_est)
            
        session.commit()
        modelo_id = registro_modelo.id
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

    # Devolver resultados para ser usados en memoria por la UI
    return {
        "modelo": modelo,
        "modelo_id": modelo_id,
        "etiquetas_asignadas": etiquetas_cluster.tolist(),
        "probabilidades": probabilidades.tolist(),
        "etiquetas_texto": etiquetas_texto,
        "silhouette": score_silueta,
        "bic": bic,
        "aic": aic,
        "n_registros": total_registros
    }
