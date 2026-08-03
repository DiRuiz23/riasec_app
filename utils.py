import re
import pandas as pd


def hex_to_rgba(hex_color: str, alpha: float = 0.15) -> str:
    """Convierte un color hexadecimal (#RRGGBB) a una cadena rgba() válida para Plotly."""
    if not isinstance(hex_color, str):
        raise ValueError(f"El color debe ser una cadena, se recibió: {type(hex_color)}")

    hex_color_clean = hex_color.lstrip("#")

    if not re.match(r"^[0-9A-Fa-f]{6}$", hex_color_clean):
        raise ValueError(f"Color hex inválido: {hex_color!r}")

    r = int(hex_color_clean[0:2], 16)
    g = int(hex_color_clean[2:4], 16)
    b = int(hex_color_clean[4:6], 16)

    return f"rgba({r},{g},{b},{alpha})"


def comparar_clusters_con_columna(df: pd.DataFrame, columna_referencia: str, clusters: pd.Series) -> pd.DataFrame:
    """
    Devuelve una tabla cruzada (crosstab) porcentual de `columna_referencia`
    dentro de cada cluster asignado.
    """
    if columna_referencia not in df.columns:
        raise ValueError(f"La columna '{columna_referencia}' no existe en el dataset.")

    crosstab_abs = pd.crosstab(clusters, df[columna_referencia], rownames=['Cluster'])
    crosstab_pct = crosstab_abs.div(crosstab_abs.sum(axis=1), axis=0) * 100

    return crosstab_pct.round(2)

def seleccionar_columnas_para_merge(df_resultados: pd.DataFrame):
    """Devuelve las columnas seguras para hacer merge cuando el dataframe de resultados está vacío."""
    if df_resultados is None or df_resultados.empty and len(df_resultados.columns) == 0:
        return []

    id_col_res = "usuario_id" if "usuario_id" in df_resultados.columns else None
    label_col = "etiqueta_riasec" if "etiqueta_riasec" in df_resultados.columns else None

    columnas = []
    if id_col_res is not None:
        columnas.append(id_col_res)
    if label_col is not None:
        columnas.append(label_col)

    if not columnas and len(df_resultados.columns) > 0:
        columnas = [df_resultados.columns[0], df_resultados.columns[-1]]

    return columnas

def obtener_centroides(artefacto, X_procesado, clusters):
    """Devuelve los centroides del modelo usando las columnas reales del entrenamiento."""
    if isinstance(artefacto, dict):
        modelo = artefacto.get("modelo", artefacto)
        columnas = artefacto.get("columnas") or list(X_procesado.columns)
        algoritmo = str(artefacto.get("metadata", {}).get("algoritmo", "")).lower()
    else:
        modelo = artefacto
        columnas = list(X_procesado.columns)
        algoritmo = ""

    if hasattr(modelo, "means_"):
        centroides = modelo.means_
    elif hasattr(modelo, "cluster_centers_"):
        centroides = modelo.cluster_centers_
    elif isinstance(artefacto, dict) and isinstance(artefacto.get("centroides"), list):
        return pd.DataFrame(artefacto["centroides"], columns=columnas)
    else:
        if isinstance(X_procesado, pd.DataFrame):
            return X_procesado.assign(cluster=clusters).groupby("cluster").mean()
        return pd.DataFrame()

    return pd.DataFrame(centroides, columns=columnas)
