"""
entrenamiento.py
Entrena el modelo no supervisado (Gaussian Mixture Model) sobre los vectores
[R,I,A,S,E,C] almacenados en la base de datos, guarda el modelo en disco
(joblib) y registra hiperparametros + metricas en la tabla modelos_entrenados.
"""

import os
import json
import joblib
import numpy as np
from datetime import datetime
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from db import get_session, VectorRiasec, Usuario, ModeloEntrenado, ResultadoClustering
from cuestionario import NOMBRES_DIMENSION

CARPETA_MODELOS = "modelos"
os.makedirs(CARPETA_MODELOS, exist_ok=True)

DIMENSIONES = ["r", "i", "a", "s", "e", "c"]


def obtener_matriz_entrenamiento(session):
    """Regresa (usuario_ids, X) con X de forma (n_muestras, 6)."""
    vectores = session.query(VectorRiasec).all()
    usuario_ids = [v.usuario_id for v in vectores]
    X = np.array([[v.r, v.i, v.a, v.s, v.e, v.c] for v in vectores], dtype=float)
    return usuario_ids, X


def etiquetar_clusters(centroides):
    """
    Interpreta cada centroide (media de las 6 dimensiones dentro del cluster)
    y le asigna una etiqueta legible con la(s) dimension(es) dominante(s).
    Si dos dimensiones estan muy cerca del maximo (diferencia <= 0.75),
    se etiqueta como perfil mixto ('Predominantemente X-Y').
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


def entrenar_modelo(n_componentes=6, covariance_type="full", desactivar_anteriores=True):
    """
    Entrena un GMM, guarda el .pkl, calcula metricas (silhouette, BIC, AIC),
    guarda el registro en modelos_entrenados y asigna resultados por usuario.
    """
    session = get_session()
    try:
        usuario_ids, X = obtener_matriz_entrenamiento(session)
        if len(X) < n_componentes:
            raise ValueError(
                f"Se necesitan al menos {n_componentes} registros para entrenar "
                f"con {n_componentes} componentes (hay {len(X)})."
            )

        modelo = GaussianMixture(
            n_components=n_componentes,
            covariance_type=covariance_type,
            random_state=42,
            n_init=10,
        )
        modelo.fit(X)

        etiquetas_cluster = modelo.predict(X)
        probabilidades = modelo.predict_proba(X)

        # Silhouette requiere al menos 2 clusters distintos presentes
        try:
            score_silueta = float(silhouette_score(X, etiquetas_cluster))
        except ValueError:
            score_silueta = None

        bic = float(modelo.bic(X))
        aic = float(modelo.aic(X))

        etiquetas_texto = etiquetar_clusters(modelo.means_)

        fecha = datetime.utcnow()
        nombre_archivo = f"gmm_riasec_{fecha.strftime('%Y%m%d_%H%M%S')}.pkl"
        ruta = os.path.join(CARPETA_MODELOS, nombre_archivo)
        joblib.dump(modelo, ruta)

        if desactivar_anteriores:
            session.query(ModeloEntrenado).update({ModeloEntrenado.activo: 0})

        registro_modelo = ModeloEntrenado(
            fecha_entrenamiento=fecha,
            algoritmo="GaussianMixture",
            n_componentes=n_componentes,
            covariance_type=covariance_type,
            n_registros_entrenamiento=len(X),
            silhouette_score=score_silueta,
            bic=bic,
            aic=aic,
            ruta_archivo=ruta,
            activo=1,
        )
        session.add(registro_modelo)
        session.flush()

        for i, usuario_id in enumerate(usuario_ids):
            cluster_id = int(etiquetas_cluster[i])
            session.add(ResultadoClustering(
                usuario_id=usuario_id,
                modelo_id=registro_modelo.id,
                cluster_id=cluster_id,
                etiqueta_riasec=etiquetas_texto[cluster_id],
                probabilidades_json=json.dumps(probabilidades[i].tolist()),
            ))

        session.commit()
        return {
            "modelo_id": registro_modelo.id,
            "ruta_archivo": ruta,
            "silhouette": score_silueta,
            "bic": bic,
            "aic": aic,
            "n_registros": len(X),
            "centroides": modelo.means_.tolist(),
            "etiquetas": etiquetas_texto,
        }
    finally:
        session.close()


def cargar_modelo_activo():
    session = get_session()
    try:
        registro = session.query(ModeloEntrenado).filter_by(activo=1).order_by(
            ModeloEntrenado.fecha_entrenamiento.desc()
        ).first()
        if registro is None:
            return None, None
        modelo = joblib.load(registro.ruta_archivo)
        return modelo, registro
    finally:
        session.close()


def proyeccion_pca_2d(X):
    """Reduce el vector de 6 dimensiones a 2 componentes principales para graficar."""
    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(X), pca.explained_variance_ratio_
