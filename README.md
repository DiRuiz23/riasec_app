# RIASEC App

## ¿Qué es este proyecto?
Este proyecto es una aplicación web hecha con Streamlit para analizar un perfil vocacional basado en las dimensiones RIASEC:

- R: Realista
- I: Investigador
- A: Artístico
- S: Social
- E: Emprendedor
- C: Convencional

La app permite responder un cuestionario, guardar los resultados, visualizar estadísticas y entrenar un modelo de clustering no supervisado para agrupar a los usuarios.

## Estructura principal
- app.py: interfaz principal de la aplicación.
- cuestionario.py: define las preguntas y el cálculo del vector RIASEC.
- db.py: configuración de la base de datos y modelos SQLAlchemy.
- entrenamiento.py: entrenamiento del modelo GMM y generación de resultados.
- estadistica.py: funciones para calcular estadísticas descriptivas.
- seed.py: inserta datos base para probar la app.

## Requisitos
Asegúrate de tener Python instalado y luego instala las dependencias necesarias:

```bash
pip install streamlit pandas plotly sqlalchemy scikit-learn joblib
```

## Cómo usar la aplicación

### 1. Cargar datos iniciales
Antes de abrir la interfaz por primera vez, ejecuta:

```bash
python seed.py
```

Esto crea la base de datos y carga datos de ejemplo para poder probar el sistema.

### 2. Iniciar la app
Ejecuta:

```bash
streamlit run app.py
```

La interfaz se abrirá en tu navegador.

### 3. Usar las funciones principales
En la barra lateral encontrarás varias secciones:

- Responder cuestionario: completa las preguntas y guarda tus respuestas.
- Carga y visualización: puedes subir un archivo CSV con datos externos.
- Estadística descriptiva: revisa resúmenes y distribuciones por categoría.
- Entrenamiento del modelo: entrena un modelo de clustering.
- Resultados: visualiza los grupos obtenidos.
- Comparativa de clústeres: observa los perfiles promedio de cada grupo.
- Metadatos del modelo: revisa información del entrenamiento.
- Descargas: exporta resultados y reportes.

## Archivos generados
- riasec.db: base de datos SQLite creada automáticamente.
- modelos/: carpeta donde se guardan los modelos entrenados.

## Nota importante
Si quieres probar la app con datos reales o de ejemplo, el flujo recomendado es:
1. ejecutar seed.py,
2. abrir la app,
3. responder el cuestionario o cargar un CSV,
4. entrenar el modelo.
