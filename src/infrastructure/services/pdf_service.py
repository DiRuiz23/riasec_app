import io
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import numpy as np
import matplotlib.pyplot as plt

def generar_grafico_radar(df):
    if df.empty or not all(c in df.columns for c in ["R", "I", "A", "S", "E", "C"]):
        return None
    
    promedios = df[["R", "I", "A", "S", "E", "C"]].mean()
    labels = np.array(["R", "I", "A", "S", "E", "C"])
    stats = promedios.values
    
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    
    stats = np.concatenate((stats, [stats[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    
    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    ax.fill(angles, stats, color='#2a3f5f', alpha=0.3)
    ax.plot(angles, stats, color='#2a3f5f', linewidth=2)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticks([]) 
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', transparent=True)
    plt.close()
    buf.seek(0)
    return buf

def crear_reporte_pdf(df, modelo, registro, etiquetas, fig_plot=None):
    """
    Genera un archivo PDF (BytesIO) que contiene:
    1. Portada/Título
    2. Gráfico interactivo exportado como imagen (si existe)
    3. Interpretación cualitativa (etiquetas de clústeres)
    4. Tabla de datos formateada (Landscape)
    """
    pdf_buffer = io.BytesIO()
    
    # Configuramos el documento en landscape para que la tabla quepa mejor
    doc = SimpleDocTemplate(
        pdf_buffer, 
        pagesize=landscape(letter),
        rightMargin=30, 
        leftMargin=30, 
        topMargin=30, 
        bottomMargin=30
    )
    
    elements = []
    styles = getSampleStyleSheet()
    
    # Estilos personalizados
    titulo_style = styles["Title"]
    subtitulo_style = ParagraphStyle(
        'Subtitulo',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=10,
        textColor=colors.HexColor("#2a3f5f")
    )
    texto_style = styles["Normal"]
    
    # --- 1. TÍTULO Y ENCABEZADO ---
    elements.append(Paragraph("Reporte de Análisis Vocacional RIASEC", titulo_style))
    fecha_str = datetime.now().strftime("%d/%m/%Y %H:%M")
    elements.append(Paragraph(f"<b>Generado el:</b> {fecha_str}", texto_style))
    
    if registro:
        elements.append(Paragraph(f"<b>Modelo Activo:</b> #{registro.id} - {registro.algoritmo}", texto_style))
        elements.append(Paragraph(f"<b>Componentes (Clústeres):</b> {registro.n_componentes}", texto_style))
    elements.append(Spacer(1, 20))
    
    # --- 2. GRÁFICO (IMAGEN) ---
    img_buffer = generar_grafico_radar(df)
    if img_buffer is not None:
        try:
            # Incrustamos la imagen de matplotlib
            img = Image(img_buffer, width=400, height=400)
            elements.append(Paragraph("Promedio RIASEC de los Usuarios Filtrados", subtitulo_style))
            elements.append(img)
            elements.append(Spacer(1, 20))
        except Exception as e:
            elements.append(Paragraph(f"<i>(No se pudo generar la imagen del gráfico: {str(e)})</i>", texto_style))
            elements.append(Spacer(1, 20))

    # --- 3. ANÁLISIS CUALITATIVO ---
    if etiquetas:
        elements.append(Paragraph("Interpretación Cualitativa de Clústeres", subtitulo_style))
        for cluster_id, etiqueta in etiquetas.items():
            elements.append(Paragraph(f"<b>Clúster {cluster_id}:</b> {etiqueta}", texto_style))
        elements.append(Spacer(1, 20))
        
    # --- 4. TABLA DE DATOS ---
    elements.append(Paragraph("Datos de Usuarios (Resumen)", subtitulo_style))
    
    # Preparar datos para la tabla
    columnas = df.columns.tolist()
    # Para evitar que la tabla se rompa, truncamos strings largos
    data = [columnas]
    
    # Limitamos los registros si son demasiados para no generar un PDF de mil paginas
    MAX_ROWS = 100
    df_truncado = df.head(MAX_ROWS)
    
    for _, row in df_truncado.iterrows():
        fila = []
        for val in row.values:
            val_str = str(val)
            if len(val_str) > 30: # truncar
                val_str = val_str[:27] + "..."
            fila.append(val_str)
        data.append(fila)
        
    if len(df) > MAX_ROWS:
        data.append([f"... y {len(df)-MAX_ROWS} registros más"] * len(columnas))
    
    # Ajuste de ancho de columnas básico
    ancho_disponible = landscape(letter)[0] - 60 # Restar margenes
    col_width = ancho_disponible / len(columnas) if len(columnas) > 0 else 50
    
    table = Table(data, repeatRows=1, colWidths=[col_width]*len(columnas))
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2a3f5f")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 10),
        ('BOTTOMPADDING', (0,0), (-1,0), 8),
        ('BACKGROUND', (0,1), (-1,-1), colors.white),
        # Efecto Zebra striping
        * [('BACKGROUND', (0, i), (-1, i), colors.HexColor("#f4f4f4")) for i in range(1, len(data), 2)],
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('FONTSIZE', (0,1), (-1,-1), 8),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('WORDWRAP', (0,0), (-1,-1), 'CJK')
    ]))
    
    elements.append(table)
    
    # Generar PDF
    doc.build(elements)
    pdf_buffer.seek(0)
    return pdf_buffer
