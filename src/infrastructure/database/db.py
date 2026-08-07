"""
db.py
Definición del esquema de base de datos (SQLAlchemy).
Cumple con el requerimiento: NO almacenar conjuntos de datos crudos (ni respuestas ni usuarios).
SÓLO almacena estadísticas generadas, características del modelo, fechas y versiones.
"""
import os
import datetime
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Text, Boolean
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

DATABASE_URL = os.environ.get("RIASEC_DATABASE_URL", "sqlite:///riasec.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

class ModeloEntrenado(Base):
    """
    Historial de versiones y ejecuciones del algoritmo.
    Solo guarda características del modelo y fechas, NO guarda al autor.
    """
    __tablename__ = "modelos_entrenados"

    id = Column(Integer, primary_key=True)
    version = Column(Integer, default=1)
    fecha_creacion = Column(DateTime, default=datetime.datetime.utcnow)
    fecha_modificacion = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    # Características del modelo
    algoritmo = Column(String(50), default="GaussianMixture")
    n_componentes = Column(Integer, nullable=False)
    covariance_type = Column(String(20), nullable=False)
    n_registros_entrenamiento = Column(Integer, nullable=False)
    bic = Column(Float, nullable=True)
    aic = Column(Float, nullable=True)
    ruta_archivo = Column(String(255), nullable=True)
    
    activo = Column(Boolean, default=True)

    estadisticas = relationship("EstadisticaClustering", back_populates="modelo", cascade="all, delete-orphan")


class EstadisticaClustering(Base):
    """
    Estadísticas generadas por cada clúster de un modelo específico.
    No almacena el conjunto de datos de los usuarios, solo métricas agregadas.
    """
    __tablename__ = "estadisticas_clustering"

    id = Column(Integer, primary_key=True)
    modelo_id = Column(Integer, ForeignKey("modelos_entrenados.id"), nullable=False)
    cluster_index = Column(Integer, nullable=False)
    
    cantidad_elementos = Column(Integer, nullable=False)
    porcentaje = Column(Float, nullable=False)
    
    # JSON con los patrones y promedios descubiertos para el clúster
    patrones_json = Column(Text, nullable=False) 

    modelo = relationship("ModeloEntrenado", back_populates="estadisticas")


def init_db():
    """Crea todas las tablas si no existen."""
    Base.metadata.create_all(engine)

def get_session():
    return SessionLocal()
