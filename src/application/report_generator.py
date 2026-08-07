"""
report_generator.py
Genera un reporte PDF con los datos y resultados filtrados actuales de la sesión.
"""

from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib import colors
import pandas as pd
import json

def generar_reporte_pdf(df_filtrado: pd.DataFrame, modelo_info: dict) -> BytesIO:
    """
    Genera un archivo PDF en memoria conteniendo los datos estadísticos y 
    los registros actuales filtrados.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    # Estilos
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ModernTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        textColor=colors.HexColor("#1E3A8A"),
        spaceAfter=20,
        alignment=TA_CENTER
    )
    
    h2_style = ParagraphStyle(
        "ModernH2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=14,
        textColor=colors.HexColor("#2563EB"),
        spaceBefore=15,
        spaceAfter=10
    )

    Story = []

    # Título
    Story.append(Paragraph("Reporte de Análisis Vocacional RIASEC", title_style))
    Story.append(Spacer(1, 12))

    if modelo_info:
        Story.append(Paragraph("Información del Modelo GMM", h2_style))
        
        # Helper para formatear valores que pueden ser nulos
        def fmt_val(val, d=1): return str(round(val, d)) if val is not None else 'N/A'
        
        texto_modelo = (
            f"<b>Algoritmo:</b> Gaussian Mixture Model<br/>"
            f"<b>Grupos identificados:</b> {modelo_info.get('n_componentes', len(modelo_info.get('etiquetas_texto', {})))}<br/>"
            f"<b>BIC (Criterio de Información Bayesiano):</b> {fmt_val(modelo_info.get('bic'), 1)}<br/>"
            f"<b>AIC:</b> {fmt_val(modelo_info.get('aic'), 1)}<br/>"
            f"<b>Silhouette Score:</b> {fmt_val(modelo_info.get('silhouette'), 3)}"
        )
        Story.append(Paragraph(texto_modelo, styles['Normal']))
        Story.append(Spacer(1, 12))
        
        # Patrones y Sesgos Identificados
        Story.append(Paragraph("Patrones y Clústeres Identificados", h2_style))
        if "etiquetas_texto" in modelo_info:
            for c_idx, label in modelo_info["etiquetas_texto"].items():
                Story.append(Paragraph(f"• <b>Clúster {c_idx}:</b> {label}", styles['Normal']))
        Story.append(Spacer(1, 12))

    # Estadísticas de los datos filtrados
    Story.append(Paragraph("Resumen de Datos Filtrados", h2_style))
    total = len(df_filtrado)
    Story.append(Paragraph(f"<b>Total de registros exportados:</b> {total}", styles['Normal']))
    Story.append(Spacer(1, 12))

    # Mostrar hasta 50 filas de la tabla para no saturar el PDF
    Story.append(Paragraph("Detalle de Registros (muestra parcial si hay más de 50)", styles['Heading3']))
    
    # Seleccionar solo las columnas más importantes para que quepan en el ancho de la hoja PDF
    cols_a_mostrar = ["id", "sexo", "dominante", "cluster", "perfil_cluster"]
    # Si alguna columna no existe en el df (ej. cluster), usar las que existan
    cols_existentes = [c for c in cols_a_mostrar if c in df_filtrado.columns]
    
    df_mostrar = df_filtrado[cols_existentes].head(50).copy()
    # Preparar tabla
    if not df_mostrar.empty:
        # Convertir todo a string para ReportLab
        columnas = list(df_mostrar.columns)
        data = [columnas]
        for _, row in df_mostrar.iterrows():
            data.append([str(item) for item in row])

        t = Table(data)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#3B82F6")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 10),
            ('BOTTOMPADDING', (0,0), (-1,0), 12),
            ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#F9FAFB")),
            ('GRID', (0,0), (-1,-1), 1, colors.HexColor("#E5E7EB"))
        ]))
        Story.append(t)
    else:
        Story.append(Paragraph("No hay registros en el conjunto filtrado.", styles['Normal']))

    doc.build(Story)
    buffer.seek(0)
    return buffer
