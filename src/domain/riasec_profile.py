"""
riasec_profile.py
Entidad de dominio que encapsula un perfil vocacional RIASEC.
Contiene validación de rangos, cálculo de dimensión dominante con manejo
de empates, y descripciones cualitativas de cada dimensión.
"""

from dataclasses import dataclass, field
from typing import Optional

DIMENSIONES_ORDEN = ["R", "I", "A", "S", "E", "C"]

NOMBRES_DIMENSION = {
    "R": "Realista",
    "I": "Investigador",
    "A": "Artístico",
    "S": "Social",
    "E": "Emprendedor",
    "C": "Convencional",
}

DESCRIPCIONES_DIMENSION = {
    "R": (
        "Prefieren actividades prácticas y concretas: trabajar con herramientas, "
        "máquinas, animales o al aire libre. Valoran la habilidad física y la "
        "destreza manual. Suelen ser francos, persistentes y prácticos."
    ),
    "I": (
        "Prefieren actividades analíticas: observar, investigar y resolver problemas "
        "complejos. Valoran las ciencias y el pensamiento abstracto. Suelen ser "
        "curiosos, metódicos e independientes."
    ),
    "A": (
        "Valoran la creatividad, la autoexpresión artística y los entornos no "
        "estructurados. Se inclinan hacia el arte, la música, la escritura o el "
        "diseño. Suelen ser imaginativos, intuitivos y originales."
    ),
    "S": (
        "Les gusta ayudar, enseñar, orientar o servir a los demás. Tienen fuertes "
        "habilidades interpersonales y empatía. Suelen ser cooperativos, amables "
        "y comprometidos con el bienestar social."
    ),
    "E": (
        "Prefieren persuadir, liderar o dirigir a otros para alcanzar metas. "
        "Valoran el poder, el estatus y la influencia. Suelen ser ambiciosos, "
        "enérgicos y seguros de sí mismos."
    ),
    "C": (
        "Prefieren actividades organizadas y estructuradas: manejo de datos, "
        "registros y rutinas claras. Valoran la precisión y el orden. Suelen "
        "ser responsables, eficientes y meticulosos."
    ),
}


@dataclass
class PerfilRIASEC:
    """
    Representa el vector RIASEC de una persona.
    Cada dimensión tiene rango 0-6 (3 preguntas × peso máximo 2).
    """
    r: float = 0.0
    i: float = 0.0
    a: float = 0.0
    s: float = 0.0
    e: float = 0.0
    c: float = 0.0
    sexo: Optional[str] = None

    RANGO_MIN: float = field(default=0.0, init=False, repr=False)
    RANGO_MAX: float = field(default=6.0, init=False, repr=False)

    def __post_init__(self):
        self._validar()

    def _validar(self):
        """Valida que todos los puntajes estén dentro del rango permitido."""
        for dim in DIMENSIONES_ORDEN:
            val = getattr(self, dim.lower())
            if not (self.RANGO_MIN <= val <= self.RANGO_MAX):
                raise ValueError(
                    f"La dimensión '{dim}' tiene valor {val}, "
                    f"fuera del rango [{self.RANGO_MIN}, {self.RANGO_MAX}]."
                )

    def como_dict(self) -> dict:
        """Devuelve las dimensiones como diccionario {R: val, I: val, ...}."""
        return {dim: getattr(self, dim.lower()) for dim in DIMENSIONES_ORDEN}

    def dimension_dominante(self) -> str:
        """
        Retorna la dimensión con mayor puntaje.
        En caso de empate, retorna la combinación de las dimensiones empatadas
        en orden de aparición del hexágono Holland (RIASEC).
        """
        scores = self.como_dict()
        max_val = max(scores.values())
        empatadas = [d for d in DIMENSIONES_ORDEN if scores[d] == max_val]
        if len(empatadas) == 1:
            return empatadas[0]
        # Empate: devolver las dos primeras en el orden del hexágono
        return "-".join(empatadas[:2])

    def es_perfil_empatado(self) -> bool:
        """Indica si hay empate en la dimensión dominante."""
        scores = self.como_dict()
        max_val = max(scores.values())
        return sum(1 for v in scores.values() if v == max_val) > 1

    def descripcion_dominante(self) -> str:
        """Descripción cualitativa de la dimensión principal."""
        dom = self.dimension_dominante().split("-")[0]
        return DESCRIPCIONES_DIMENSION.get(dom, "")

    def nombre_dominante(self) -> str:
        """Nombre legible de la dimensión dominante."""
        dom = self.dimension_dominante().split("-")[0]
        return NOMBRES_DIMENSION.get(dom, dom)

    def puntaje_total(self) -> float:
        """Suma total de todas las dimensiones."""
        return sum(getattr(self, d.lower()) for d in DIMENSIONES_ORDEN)

    def ranking_dimensiones(self) -> list[tuple[str, float]]:
        """
        Lista de tuplas (dimensión, puntaje) ordenadas de mayor a menor.
        Útil para mostrar el perfil completo en orden de fortaleza.
        """
        scores = self.como_dict()
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    @classmethod
    def desde_dict(cls, datos: dict, sexo: Optional[str] = None) -> "PerfilRIASEC":
        """
        Construye un PerfilRIASEC desde un diccionario.
        Acepta claves en mayúsculas o minúsculas.
        """
        return cls(
            r=float(datos.get("r", datos.get("R", 0))),
            i=float(datos.get("i", datos.get("I", 0))),
            a=float(datos.get("a", datos.get("A", 0))),
            s=float(datos.get("s", datos.get("S", 0))),
            e=float(datos.get("e", datos.get("E", 0))),
            c=float(datos.get("c", datos.get("C", 0))),
            sexo=sexo,
        )
