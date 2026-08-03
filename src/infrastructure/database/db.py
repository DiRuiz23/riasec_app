"""
db.py
Definicion del esquema de base de datos y motor de conexion (SQLAlchemy).
Compatible con SQLite (desarrollo) y PostgreSQL (produccion) sin cambiar el
codigo del resto de la aplicacion: solo se modifica la variable DATABASE_URL.
"""

import os
import datetime
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Text
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

# --------------------------------------------------------------------------
# Configuracion de conexion
# --------------------------------------------------------------------------
# Para PostgreSQL usar algo como:
# postgresql+psycopg2://usuario:password@localhost:5432/riasec_db
DATABASE_URL = os.environ.get("RIASEC_DATABASE_URL", "sqlite:///riasec.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


# --------------------------------------------------------------------------
# Tablas
# --------------------------------------------------------------------------
class Pregunta(Base):
    """Catalogo fijo de las 18 preguntas ponderadas + 1 demografica (peso null)."""
    __tablename__ = "preguntas"

    id = Column(Integer, primary_key=True)
    numero = Column(Integer, nullable=False, unique=True)  # 0 a 18
    dimension = Column(String(20), nullable=False)  # R, I, A, S, E, C o DEMOGRAFICA
    texto = Column(Text, nullable=False)

    opciones = relationship("OpcionPregunta", back_populates="pregunta")


class OpcionPregunta(Base):
    """Las 3 opciones de respuesta de cada pregunta y su peso (2, 1 o 0)."""
    __tablename__ = "opciones_pregunta"

    id = Column(Integer, primary_key=True)
    pregunta_id = Column(Integer, ForeignKey("preguntas.id"), nullable=False)
    texto_opcion = Column(String(255), nullable=False)
    peso = Column(Integer, nullable=True)  # None para la pregunta demografica

    pregunta = relationship("Pregunta", back_populates="opciones")


class Usuario(Base):
    """Cada persona que contesta el cuestionario."""
    __tablename__ = "usuarios"

    id = Column(Integer, primary_key=True)
    sexo = Column(String(10), nullable=True)          # Hombre / Mujer
    edad = Column(Integer, nullable=True)
    carrera_interes = Column(String(150), nullable=True)
    fecha_registro = Column(DateTime, default=datetime.datetime.utcnow)

    respuestas = relationship("Respuesta", back_populates="usuario")
    vector = relationship("VectorRiasec", back_populates="usuario", uselist=False)


class Respuesta(Base):
    """Respuesta individual de un usuario a una pregunta (guarda el peso obtenido)."""
    __tablename__ = "respuestas"

    id = Column(Integer, primary_key=True)
    usuario_id = Column(Integer, ForeignKey("usuarios.id"), nullable=False)
    pregunta_id = Column(Integer, ForeignKey("preguntas.id"), nullable=False)
    opcion_id = Column(Integer, ForeignKey("opciones_pregunta.id"), nullable=False)
    peso_obtenido = Column(Integer, nullable=True)

    usuario = relationship("Usuario", back_populates="respuestas")


class VectorRiasec(Base):
    """
    Vector agregado [R,I,A,S,E,C] por usuario (0-6 cada dimension).
    Es el input real del modelo de clustering.
    """
    __tablename__ = "vectores_riasec"

    id = Column(Integer, primary_key=True)
    usuario_id = Column(Integer, ForeignKey("usuarios.id"), nullable=False, unique=True)
    r = Column(Integer, nullable=False)
    i = Column(Integer, nullable=False)
    a = Column(Integer, nullable=False)
    s = Column(Integer, nullable=False)
    e = Column(Integer, nullable=False)
    c = Column(Integer, nullable=False)

    usuario = relationship("Usuario", back_populates="vector")


class ModeloEntrenado(Base):
    """Historial de entrenamientos: hiperparametros, metricas y ruta del archivo .pkl."""
    __tablename__ = "modelos_entrenados"

    id = Column(Integer, primary_key=True)
    fecha_entrenamiento = Column(DateTime, default=datetime.datetime.utcnow)
    algoritmo = Column(String(50), default="GaussianMixture")
    n_componentes = Column(Integer, nullable=False)
    covariance_type = Column(String(20), nullable=False)
    n_registros_entrenamiento = Column(Integer, nullable=False)
    silhouette_score = Column(Float, nullable=True)
    bic = Column(Float, nullable=True)
    aic = Column(Float, nullable=True)
    ruta_archivo = Column(String(255), nullable=False)
    activo = Column(Integer, default=1)  # 1 = modelo vigente para inferencia

    resultados = relationship("ResultadoClustering", back_populates="modelo")


class ResultadoClustering(Base):
    """Resultado de asignar un usuario a un cluster con un modelo especifico."""
    __tablename__ = "resultados_clustering"

    id = Column(Integer, primary_key=True)
    usuario_id = Column(Integer, ForeignKey("usuarios.id"), nullable=False)
    modelo_id = Column(Integer, ForeignKey("modelos_entrenados.id"), nullable=False)
    cluster_id = Column(Integer, nullable=False)
    etiqueta_riasec = Column(String(100), nullable=False)
    probabilidades_json = Column(Text, nullable=False)  # JSON string con prob. por cluster

    modelo = relationship("ModeloEntrenado", back_populates="resultados")


def init_db():
    """Crea todas las tablas si no existen."""
    Base.metadata.create_all(engine)


def get_session():
    return SessionLocal()
