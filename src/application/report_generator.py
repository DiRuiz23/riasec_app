"""
report_generator.py
Genera un reporte PDF con los datos y resultados filtrados actuales de la sesión.
"""

from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
import pandas as pd

def generar_reporte_pdf(df_filtrado: pd.DataFrame, modelo_info: dict) -> BytesIO:
    """
    Genera un archivo PDF en memoria conteniendo los datos estadísticos y 
    los registros actuales filtrados.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    Story = []

    # Título
    Story.append(Paragraph("Reporte de Análisis Vocacional RIASEC", styles['Title']))
    Story.append(Spacer(1, 12))

    # Información del modelo (si existe)
    if modelo_info:
        Story.append(Paragraph("Información del Modelo GMM", styles['Heading2']))
        texto_modelo = (
            f"Algoritmo: Gaussian Mixture Model<br/>"
            f"Grupos identificados: {modelo_info.get('n_componentes', 'N/A')}<br/>"
            f"BIC: {modelo_info.get('bic', 'N/A')} | AIC: {modelo_info.get('aic', 'N/A')}<br/>"
            f"Silhouette Score: {modelo_info.get('silhouette', 'N/A')}"
        )
        Story.append(Paragraph(texto_modelo, styles['Normal']))
        Story.append(Spacer(1, 12))

    # Estadísticas de los datos filtrados
    Story.append(Paragraph("Resumen de Datos Filtrados", styles['Heading2']))
    total = len(df_filtrado)
    Story.append(Paragraph(f"Total de registros exportados: {total}", styles['Normal']))
    Story.append(Spacer(1, 12))

    # Mostrar hasta 50 filas de la tabla para no saturar el PDF
    Story.append(Paragraph("Detalle de Registros (muestra parcial si hay más de 50)", styles['Heading3']))
    
    df_mostrar = df_filtrado.head(50).copy()
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
