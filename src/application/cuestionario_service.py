"""
cuestionario_service.py
Catalogo fijo de las 18 preguntas RIASEC + 1 pregunta demografica,
función de agregación que convierte respuestas crudas en el vector [R,I,A,S,E,C],
diccionario de carreras sugeridas por dimensión y validación de mapeo CSV.
"""
from typing import Optional

# Cada tupla: (numero, dimension, texto, [(texto_opcion, peso), ...])
PREGUNTAS = [
    (0, "DEMOGRAFICA", "Selecciona tu sexo",
        [("Hombre", None), ("Mujer", None)]),

    (1, "R", "Disfrutas reparar o construir cosas con tus manos",
        [("Sí, me identifica mucho", 2), ("A veces me identifica", 1), ("No me identifica", 0)]),
    (2, "R", "Prefieres actividades físicas o al aire libre antes que permanecer en una oficina",
        [("Sí, lo prefiero", 2), ("Depende de la situación", 1), ("No, prefiero actividades de oficina", 0)]),
    (3, "R", "Te interesa entender cómo funcionan las máquinas o las herramientas",
        [("Me interesa mucho", 2), ("Me interesa un poco", 1), ("No me interesa", 0)]),

    (4, "I", "Antes de tomar una decisión, prefieres analizar la información",
        [("Siempre", 2), ("Algunas veces", 1), ("Casi nunca", 0)]),
    (5, "I", "Disfrutas resolver problemas complejos o rompecabezas",
        [("Mucho", 2), ("De vez en cuando", 1), ("Poco o nada", 0)]),
    (6, "I", "Te gusta investigar temas científicos o técnicos con profundidad",
        [("Sí, frecuentemente", 2), ("Solo cuando el tema me interesa", 1), ("No suele gustarme", 0)]),

    (7, "A", "Prefieres expresarte mediante el arte, la música o la escritura",
        [("Sí, es una de mis formas favoritas", 2), ("En algunas ocasiones", 1), ("No es algo que disfrute", 0)]),
    (8, "A", "Cuando surge un problema, buscas soluciones creativas o poco convencionales",
        [("Casi siempre", 2), ("Algunas veces", 1), ("Prefiero soluciones tradicionales", 0)]),
    (9, "A", "Disfrutas diseñar o crear cosas nuevas",
        [("Mucho", 2), ("Algunas veces", 1), ("Poco o nada", 0)]),

    (10, "S", "Te gusta ayudar a otras personas a resolver sus problemas",
        [("Siempre que puedo", 2), ("Solo en algunas ocasiones", 1), ("Prefiero no involucrarme", 0)]),
    (11, "S", "Te resulta fácil escuchar y comprender las emociones de los demás",
        [("Sí, con facilidad", 2), ("Depende de la persona", 1), ("Me cuesta trabajo", 0)]),
    (12, "S", "Prefieres trabajar en equipo antes que trabajar solo",
        [("Sí, definitivamente", 2), ("Depende de la actividad", 1), ("Prefiero trabajar solo", 0)]),

    (13, "E", "Te gusta convencer a otras personas y dirigir proyectos",
        [("Sí, disfruto liderar", 2), ("Solo cuando es necesario", 1), ("Prefiero no liderar", 0)]),
    (14, "E", "Estás dispuesto(a) a asumir riesgos para alcanzar una meta importante",
        [("Sí, sin problema", 2), ("Solo si el riesgo es razonable", 1), ("Prefiero evitar riesgos", 0)]),
    (15, "E", "Te gustaría iniciar tu propio negocio o proyecto en el futuro",
        [("Sí, definitivamente", 2), ("Tal vez", 1), ("No está en mis planes", 0)]),

    (16, "C", "Prefieres seguir procedimientos y mantener todo organizado",
        [("Sí, siempre", 2), ("Solo cuando es necesario", 1), ("Prefiero trabajar de manera flexible", 0)]),
    (17, "C", "Prefieres tareas con instrucciones claras antes que actividades ambiguas",
        [("Sí, me siento más cómodo(a)", 2), ("Depende de la situación", 1), ("Prefiero tener libertad para decidir", 0)]),
    (18, "C", "Te gusta llevar un registro detallado de tus actividades o gastos",
        [("Sí, de manera constante", 2), ("Solo en algunas ocasiones", 1), ("Casi nunca", 0)]),
]

DIMENSIONES = ["R", "I", "A", "S", "E", "C"]
NOMBRES_DIMENSION = {
    "R": "Realista", "I": "Investigador", "A": "Artístico",
    "S": "Social", "E": "Emprendedor", "C": "Convencional",
}

# ── Carreras sugeridas por dimensión dominante (Holland, 1997) ─────────────────
CARRERAS_RIASEC: dict[str, list[str]] = {
    "R": [
        "Ingeniería Mecánica", "Ingeniería Civil", "Arquitectura",
        "Agronomía", "Técnico Electrónico", "Geología", "Construcción",
    ],
    "I": [
        "Medicina", "Biología", "Física", "Matemáticas",
        "Química", "Ciencias de Datos", "Astronomía", "Neurociencias",
    ],
    "A": [
        "Diseño Gráfico", "Artes Visuales", "Arquitectura", "Música",
        "Comunicación", "Cine y Televisión", "Fotografía", "Publicidad",
    ],
    "S": [
        "Psicología", "Trabajo Social", "Enfermería", "Pedagogía",
        "Terapia Ocupacional", "Sociología", "Orientación Educativa",
    ],
    "E": [
        "Administración de Empresas", "Mercadotecnia", "Derecho",
        "Relaciones Internacionales", "Emprendimiento", "Ventas",
    ],
    "C": [
        "Contabilidad", "Finanzas", "Administración Pública",
        "Estadística", "Logística", "Auditoría", "Sistemas de Información",
    ],
}


def agregar_vector(respuestas_peso: dict) -> dict:
    """
    respuestas_peso: {numero_pregunta: peso_obtenido} para las preguntas 1-18.
    Regresa {'R':x,'I':x,'A':x,'S':x,'E':x,'C':x} con rango 0-6 cada una.
    """
    vector = {d: 0 for d in DIMENSIONES}
    for num, dim, _texto, _opciones in PREGUNTAS:
        if dim == "DEMOGRAFICA":
            continue
        peso = respuestas_peso.get(num, 0)
        vector[dim] += peso
    return vector


def validar_mapeo_csv(df_csv, mapeo_respuestas: dict) -> list[str]:
    """
    Detecta valores en el CSV que no están en el mapeo de respuestas
    y devuelve una lista de advertencias descriptivas.
    
    Esto previene que respuestas no mapeadas queden silenciosamente en 0
    distorsionando los puntajes.
    """
    advertencias = []
    cols = df_csv.columns.tolist()
    # Columnas de preguntas: índices 2-19 (después de marca temporal y sexo)
    cols_preguntas = cols[2:20] if len(cols) >= 20 else cols[2:]
    
    for col in cols_preguntas:
        valores_unicos = df_csv[col].dropna().unique()
        no_mapeados = [v for v in valores_unicos if str(v) not in mapeo_respuestas]
        if no_mapeados:
            advertencias.append(
                f"Columna **'{col}'**: {len(no_mapeados)} valor(es) sin mapeo → "
                f"`{', '.join(str(v) for v in no_mapeados[:3])}`"
                + (" ..." if len(no_mapeados) > 3 else "")
            )
    return advertencias


def obtener_carreras_para_perfil(dim_dominante: str) -> list[str]:
    """
    Retorna las carreras sugeridas para una dimensión dominante.
    Si la dimensión es compuesta (ej. 'R-I'), combina ambas listas.
    """
    dims = dim_dominante.upper().split("-")
    carreras = []
    for d in dims:
        carreras.extend(CARRERAS_RIASEC.get(d, []))
    # Eliminar duplicados preservando orden
    vistas = set()
    resultado = []
    for c in carreras:
        if c not in vistas:
            vistas.add(c)
            resultado.append(c)
    return resultado
