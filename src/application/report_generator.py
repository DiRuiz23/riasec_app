"""
report_generator.py
Genera un reporte PDF completo en memoria con:
  - Radar chart de promedios RIASEC (matplotlib)
  - Métricas del modelo GMM
  - Patrones de clusters
  - Tabla de carreras sugeridas por cluster
  - Estadísticas descriptivas (usando estadistica_service)
  - Detalle de registros filtrados
"""

import io
from io import BytesIO
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend sin GUI para entornos de servidor
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib import colors

from src.application.cuestionario_service import NOMBRES_DIMENSION, CARRERAS_RIASEC
from src.application.estadistica_service import resumen_dimensiones


# ── Paleta de colores corporativa ─────────────────────────────────────────────
COLOR_PRIMARIO = colors.HexColor("#1E3A8A")
COLOR_SECUNDARIO = colors.HexColor("#3B82F6")
COLOR_ACENTO = colors.HexColor("#60A5FA")
COLOR_FONDO_HEADER = colors.HexColor("#1E3A8A")
COLOR_FONDO_FILA_ALT = colors.HexColor("#EFF6FF")
COLOR_TEXTO_CLARO = colors.whitesmoke
COLOR_GRID = colors.HexColor("#BFDBFE")


def _generar_radar_chart(df: pd.DataFrame) -> io.BytesIO | None:
    """
    Genera un gráfico radar (spider chart) de los promedios RIASEC.
    Si el DataFrame tiene columna 'sexo', genera dos radars superpuestos (M vs F).
    Retorna un buffer PNG.
    """
    dims_lower = ["r", "i", "a", "s", "e", "c"]
    labels_full = ["Realista", "Investigador", "Artístico", "Social", "Emprendedor", "Convencional"]

    if df.empty or not all(c in df.columns for c in dims_lower):
        return None

    angles = np.linspace(0, 2 * np.pi, len(dims_lower), endpoint=False).tolist()
    angles_closed = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_facecolor("#F8FAFC")

    # Colores por sexo o general
    series = []
    if "sexo" in df.columns and df["sexo"].nunique() > 1:
        for sexo, color, label in [("M", "#1E3A8A", "Masculino"), ("F", "#EC4899", "Femenino")]:
            sub = df[df["sexo"] == sexo]
            if not sub.empty:
                valores = sub[dims_lower].mean().values.tolist()
                series.append((valores, color, label))
    else:
        valores = df[dims_lower].mean().values.tolist()
        series.append((valores, "#1E3A8A", "Promedio"))

    for valores, color, label in series:
        vals_closed = valores + [valores[0]]
        ax.fill(angles_closed, vals_closed, color=color, alpha=0.15)
        ax.plot(angles_closed, vals_closed, color=color, linewidth=2.5, label=label)

    ax.set_xticks(angles)
    ax.set_xticklabels(labels_full, fontsize=9, color="#1E3A8A", fontweight="bold")
    ax.set_yticks([1, 2, 3, 4, 5, 6])
    ax.set_yticklabels(["1", "2", "3", "4", "5", "6"], fontsize=7, color="#94A3B8")
    ax.set_ylim(0, 6)
    ax.spines["polar"].set_color("#CBD5E1")
    ax.grid(color="#CBD5E1", linestyle="--", linewidth=0.5)

    if len(series) > 1:
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

    plt.title("Perfil RIASEC Promedio", fontsize=12, fontweight="bold",
               color="#1E3A8A", pad=20)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    buf.seek(0)
    return buf


def generar_reporte_pdf(df_filtrado: pd.DataFrame, modelo_info: dict) -> BytesIO:
    """
    Genera un archivo PDF completo en memoria.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "Title",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=22,
        textColor=COLOR_PRIMARIO,
        spaceAfter=4,
        alignment=TA_CENTER,
    )
    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        textColor=colors.HexColor("#64748B"),
        spaceAfter=16,
        alignment=TA_CENTER,
    )
    h2_style = ParagraphStyle(
        "H2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        textColor=COLOR_PRIMARIO,
        spaceBefore=14,
        spaceAfter=8,
    )
    normal_style = ParagraphStyle(
        "Normal",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        textColor=colors.HexColor("#1E293B"),
        spaceAfter=4,
    )
    bullet_style = ParagraphStyle(
        "Bullet",
        parent=normal_style,
        leftIndent=16,
        spaceAfter=3,
    )

    story = []

    # ── PORTADA ───────────────────────────────────────────────────────────────
    story.append(Spacer(1, 20))
    story.append(Paragraph("Reporte de Análisis Vocacional RIASEC", title_style))
    fecha_str = datetime.now().strftime("%d de %B de %Y, %H:%M")
    story.append(Paragraph(f"Generado el {fecha_str}", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=COLOR_SECUNDARIO, spaceAfter=16))

    # ── RADAR CHART ───────────────────────────────────────────────────────────
    radar_buf = _generar_radar_chart(df_filtrado)
    if radar_buf:
        story.append(Paragraph("Perfil RIASEC del Grupo Analizado", h2_style))
        story.append(Image(radar_buf, width=320, height=320))
        story.append(Spacer(1, 8))

    # ── ESTADÍSTICAS DESCRIPTIVAS ─────────────────────────────────────────────
    story.append(Paragraph("Estadísticas Descriptivas por Dimensión", h2_style))
    try:
        vectores = df_filtrado[["r", "i", "a", "s", "e", "c"]].to_dict("records")
        resumen = resumen_dimensiones(vectores)

        header = ["Dimensión", "Media", "Mediana", "Moda", "Desv. Est.", "Mín", "Máx"]
        data_stats = [header]
        for dim, stats in resumen.items():
            data_stats.append([
                NOMBRES_DIMENSION.get(dim, dim),
                str(stats["media"]),
                str(stats["mediana"]),
                str(stats["moda"]),
                str(stats["desviacion_estandar"]),
                str(stats["minimo"]),
                str(stats["maximo"]),
            ])

        t_stats = Table(data_stats, repeatRows=1)
        t_stats.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), COLOR_FONDO_HEADER),
            ("TEXTCOLOR", (0, 0), (-1, 0), COLOR_TEXTO_CLARO),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTSIZE", (0, 1), (-1, -1), 8),
            ("BACKGROUND", (0, 1), (-1, -1), colors.white),
            *[("BACKGROUND", (0, i), (-1, i), COLOR_FONDO_FILA_ALT)
              for i in range(1, len(data_stats), 2)],
            ("GRID", (0, 0), (-1, -1), 0.5, COLOR_GRID),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(t_stats)
    except Exception as e:
        story.append(Paragraph(f"(No se pudieron calcular estadísticas: {e})", normal_style))

    story.append(Spacer(1, 12))

    # ── INFORMACIÓN DEL MODELO GMM ─────────────────────────────────────────────
    if modelo_info:
        story.append(Paragraph("Información del Modelo GMM", h2_style))

        def fmt(val, d=1):
            return str(round(val, d)) if val is not None else "N/A"

        n_comp = modelo_info.get("n_componentes", len(modelo_info.get("etiquetas_texto", {})))
        info_texto = (
            f"<b>Algoritmo:</b> Gaussian Mixture Model &nbsp;|&nbsp; "
            f"<b>Clústeres:</b> {n_comp} &nbsp;|&nbsp; "
            f"<b>BIC:</b> {fmt(modelo_info.get('bic'))} &nbsp;|&nbsp; "
            f"<b>AIC:</b> {fmt(modelo_info.get('aic'))} &nbsp;|&nbsp; "
            f"<b>Silhouette:</b> {fmt(modelo_info.get('silhouette'), 3)}"
        )
        story.append(Paragraph(info_texto, normal_style))
        story.append(Spacer(1, 8))

        # Patrones de clústeres
        story.append(Paragraph("Patrones y Clústeres Identificados", h2_style))
        if "etiquetas_texto" in modelo_info:
            for c_idx, label in modelo_info["etiquetas_texto"].items():
                story.append(Paragraph(f"• <b>Clúster {c_idx}:</b> {label}", bullet_style))
        story.append(Spacer(1, 8))

    # ── CARRERAS SUGERIDAS POR DIMENSIÓN ──────────────────────────────────────
    story.append(Paragraph("Carreras Sugeridas por Dimensión RIASEC", h2_style))
    story.append(Paragraph(
        "Basado en la teoría vocacional de Holland (1997). "
        "Las carreras se presentan como orientación general, no como prescripción.",
        normal_style,
    ))
    story.append(Spacer(1, 6))

    career_header = ["Dimensión", "Descripción", "Carreras Sugeridas"]
    career_data = [career_header]
    DESCRIPCIONES_CORTAS = {
        "R": "Práctico / Técnico",
        "I": "Analítico / Científico",
        "A": "Creativo / Artístico",
        "S": "Social / Humanista",
        "E": "Emprendedor / Líder",
        "C": "Organizado / Administrativo",
    }
    for dim, nombre in NOMBRES_DIMENSION.items():
        carreras = ", ".join(CARRERAS_RIASEC.get(dim, []))
        p_carreras = Paragraph(carreras, normal_style)
        career_data.append([nombre, DESCRIPCIONES_CORTAS.get(dim, ""), p_carreras])

    t_careers = Table(career_data, colWidths=[90, 110, 310], repeatRows=1)
    t_careers.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), COLOR_FONDO_HEADER),
        ("TEXTCOLOR", (0, 0), (-1, 0), COLOR_TEXTO_CLARO),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("ALIGN", (0, 0), (1, -1), "CENTER"),
        ("ALIGN", (2, 1), (2, -1), "LEFT"),
        ("FONTSIZE", (0, 1), (-1, -1), 8),
        ("BACKGROUND", (0, 1), (-1, -1), colors.white),
        *[("BACKGROUND", (0, i), (-1, i), COLOR_FONDO_FILA_ALT)
          for i in range(1, len(career_data), 2)],
        ("GRID", (0, 0), (-1, -1), 0.5, COLOR_GRID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t_careers)
    story.append(Spacer(1, 12))

    # ── RESUMEN DATOS FILTRADOS ────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=1, color=COLOR_ACENTO, spaceAfter=10))
    story.append(Paragraph("Detalle de Registros Clasificados", h2_style))
    total = len(df_filtrado)
    story.append(Paragraph(
        f"<b>Total de registros exportados:</b> {total} "
        f"(se muestran hasta 50 en esta tabla)",
        normal_style,
    ))
    story.append(Spacer(1, 6))

    cols_a_mostrar = ["id", "sexo", "dominante", "cluster", "perfil_cluster"]
    cols_existentes = [c for c in cols_a_mostrar if c in df_filtrado.columns]
    df_mostrar = df_filtrado[cols_existentes].head(50).copy()

    if not df_mostrar.empty:
        columnas = list(df_mostrar.columns)
        data = [columnas]
        for _, row in df_mostrar.iterrows():
            data.append([str(item) for item in row])

        t = Table(data, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), COLOR_SECUNDARIO),
            ("TEXTCOLOR", (0, 0), (-1, 0), COLOR_TEXTO_CLARO),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("FONTSIZE", (0, 1), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
            ("BACKGROUND", (0, 1), (-1, -1), colors.white),
            *[("BACKGROUND", (0, i), (-1, i), COLOR_FONDO_FILA_ALT)
              for i in range(1, len(data), 2)],
            ("GRID", (0, 0), (-1, -1), 0.5, COLOR_GRID),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(t)
    else:
        story.append(Paragraph("No hay registros en el conjunto filtrado.", normal_style))

    story.append(Spacer(1, 20))
    story.append(Paragraph(
        "Reporte generado automáticamente por RIASEC Analytics · Sistema de Análisis Vocacional",
        ParagraphStyle("Footer", parent=normal_style, fontSize=8,
                       textColor=colors.HexColor("#94A3B8"), alignment=TA_CENTER),
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer
