"""
seed.py
Precarga 50 registros base (usuarios + respuestas + vector RIASEC) que sirven
UNICAMENTE como semilla de entrenamiento inicial, para que el modelo tenga con
que entrenar la primera vez que se levanta el sistema.

IMPORTANTE: este script se ejecuta una sola vez desde consola
(python seed.py) durante el despliegue. NO existe ningun boton en la
interfaz que dispare esta funcion ni que genere datos aleatorios bajo
demanda del usuario final.
"""

import random
import datetime
from src.infrastructure.database.db import init_db, get_session, Usuario, Pregunta, OpcionPregunta, Respuesta, VectorRiasec
from src.application.cuestionario_service import PREGUNTAS, agregar_vector

random.seed(42)  # semilla fija: reproducible, no "aleatorio" en cada corrida

CARRERAS_EJEMPLO = [
    "Ingeniería en Sistemas", "Psicología", "Administración", "Diseño Gráfico",
    "Mecatrónica", "Derecho", "Enfermería", "Arquitectura", "Contabilidad",
    "Mercadotecnia", None,
]


def poblar_catalogo_preguntas(session):
    """Inserta el catalogo fijo de preguntas y opciones si aun no existe."""
    if session.query(Pregunta).count() > 0:
        return
    for numero, dimension, texto, opciones in PREGUNTAS:
        p = Pregunta(numero=numero, dimension=dimension, texto=texto)
        session.add(p)
        session.flush()  # para obtener p.id
        for texto_opcion, peso in opciones:
            session.add(OpcionPregunta(pregunta_id=p.id, texto_opcion=texto_opcion, peso=peso))
    session.commit()


def generar_usuario_simulado(session, sesgo_dimension=None):
    """
    Crea un usuario simulado. Si sesgo_dimension se indica (ej. 'S'), las
    respuestas de esa dimension tienden a pesos altos (2) para que el dataset
    semilla contenga los 6 perfiles representados de forma balanceada.
    """
    sexo = random.choice(["Hombre", "Mujer"])
    edad = random.randint(16, 45)
    carrera = random.choice(CARRERAS_EJEMPLO)
    dias_atras = random.randint(0, 180)
    fecha = datetime.datetime.utcnow() - datetime.timedelta(days=dias_atras)

    usuario = Usuario(sexo=sexo, edad=edad, carrera_interes=carrera, fecha_registro=fecha)
    session.add(usuario)
    session.flush()

    preguntas_bd = {p.numero: p for p in session.query(Pregunta).all()}
    respuestas_peso = {}

    for numero, dimension, _texto, opciones in PREGUNTAS:
        pregunta_bd = preguntas_bd[numero]
        opciones_bd = pregunta_bd.opciones

        if dimension == "DEMOGRAFICA":
            opcion_elegida = opciones_bd[0] if sexo == "Hombre" else opciones_bd[1]
            session.add(Respuesta(
                usuario_id=usuario.id, pregunta_id=pregunta_bd.id,
                opcion_id=opcion_elegida.id, peso_obtenido=None
            ))
            continue

        if sesgo_dimension is not None and dimension == sesgo_dimension:
            pesos_posibles = [2, 2, 1]  # sesgado hacia el peso alto
        elif sesgo_dimension is not None:
            pesos_posibles = [0, 1, 1]  # el resto de dimensiones, mas bajo
        else:
            pesos_posibles = [0, 1, 2]  # perfil mixto/neutro

        peso_elegido = random.choice(pesos_posibles)
        opcion_elegida = next(o for o in opciones_bd if o.peso == peso_elegido)

        session.add(Respuesta(
            usuario_id=usuario.id, pregunta_id=pregunta_bd.id,
            opcion_id=opcion_elegida.id, peso_obtenido=peso_elegido
        ))
        respuestas_peso[numero] = peso_elegido

    vector = agregar_vector(respuestas_peso)
    session.add(VectorRiasec(usuario_id=usuario.id, **{k.lower(): v for k, v in vector.items()}))
    session.commit()


def poblar_usuarios_semilla(session, total=50):
    if session.query(Usuario).count() > 0:
        return
    dimensiones = ["R", "I", "A", "S", "E", "C"]
    # ~8 usuarios marcadamente sesgados por dimension (48) + 2 perfiles neutros
    contador = 0
    for dim in dimensiones:
        for _ in range(8):
            generar_usuario_simulado(session, sesgo_dimension=dim)
            contador += 1
    while contador < total:
        generar_usuario_simulado(session, sesgo_dimension=None)
        contador += 1


def main():
    init_db()
    session = get_session()
    try:
        poblar_catalogo_preguntas(session)
        poblar_usuarios_semilla(session, total=50)
        print("Semilla cargada correctamente (catalogo de preguntas + 50 usuarios).")
    finally:
        session.close()


if __name__ == "__main__":
    main()
