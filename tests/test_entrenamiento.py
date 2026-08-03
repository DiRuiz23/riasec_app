import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

import app
import entrenamiento
import streamlit as st
import utils
from db import Dataset


def test_wrapper_functions_exist():
    assert hasattr(entrenamiento, "entrenar_modelo")
    assert hasattr(entrenamiento, "obtener_matriz_entrenamiento")
    assert hasattr(entrenamiento, "etiquetar_clusters")


def test_etiquetar_clusters_labels_centroids():
    centroides = np.array([
        [5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 4.0, 0.0, 0.0, 0.0, 0.0],
    ])

    etiquetas = entrenamiento.etiquetar_clusters(centroides)

    assert etiquetas[0].startswith("Predominantemente")
    assert "Realista" in etiquetas[0] or "Investigador" in etiquetas[0]


def test_obtener_centroides_uses_real_column_names_for_generic_models():
    class FakeModel:
        def __init__(self):
            self.means_ = np.array([[1.0, 2.0], [3.0, 4.0]])

    artefacto = {"modelo": FakeModel(), "metadata": {"algoritmo": "GaussianMixture"}}
    X = pd.DataFrame([[0.0, 0.0], [1.0, 1.0]], columns=["edad", "score"])

    centroides = utils.obtener_centroides(artefacto, X, pd.Series([0, 1]))

    assert list(centroides.columns) == ["edad", "score"]
    assert centroides.iloc[0, 0] == 1.0
    assert centroides.iloc[1, 1] == 4.0


def test_etiquetar_clusters_uses_generic_column_names_when_not_riasec():
    centroides = np.array([[4.0, 0.5], [0.5, 3.0]])

    etiquetas = entrenamiento.etiquetar_clusters(centroides, X_columns=["edad", "score"])

    assert etiquetas[0].startswith("Predominantemente")
    assert "edad" in etiquetas[0] or "score" in etiquetas[0]


def test_resolver_columnas_usadas_accepts_any_dataset_columns():
    df = pd.DataFrame({"edad": [20, 21], "score": [1.2, 3.4]})
    config = {"columnas_usadas": ["edad", "score", "faltante"]}

    columnas = entrenamiento.resolver_columnas_usadas(df, config)

    assert columnas == ["edad", "score"]


def test_seleccionar_columnas_para_merge_handles_empty_results():
    df_resultados = pd.DataFrame(columns=[])

    columnas = utils.seleccionar_columnas_para_merge(df_resultados)

    assert columnas == []


def test_usuarios_a_dataframe_prefers_latest_uploaded_dataset():
    session = app.get_session()
    try:
        session.query(Dataset).delete()
        session.commit()

        ds = Dataset(
            source_name="demo.csv",
            columns=json.dumps(["edad", "score"]),
            records=json.dumps([{"edad": 20, "score": 3.5}, {"edad": 25, "score": 4.2}]),
        )
        session.add(ds)
        session.commit()
        st.session_state["dataset_actual_id"] = ds.id

        df = app.usuarios_a_dataframe()

        assert list(df.columns) == ["edad", "score"]
        assert len(df) == 2
        assert df.iloc[0]["edad"] == 20
    finally:
        session.close()
