"""
estadistica.py
Estadistica descriptiva calculada con algoritmos propios (sin usar
funciones ya resueltas como numpy.mean, statistics.mode, pandas.describe, etc.),
tal como lo exige la rubrica.
"""

from collections import Counter


def media(valores):
    if not valores:
        return 0.0
    return sum(valores) / len(valores)


def desviacion_estandar(valores):
    """Desviacion estandar poblacional, calculada paso a paso."""
    if len(valores) < 2:
        return 0.0
    m = media(valores)
    suma_cuadrados = 0.0
    for v in valores:
        suma_cuadrados += (v - m) ** 2
    varianza = suma_cuadrados / len(valores)
    return varianza ** 0.5


def moda(valores):
    """Regresa el/los valores mas frecuentes (puede ser multimodal)."""
    if not valores:
        return []
    conteo = Counter(valores)
    frecuencia_max = max(conteo.values())
    return sorted([v for v, f in conteo.items() if f == frecuencia_max])


def mediana(valores):
    if not valores:
        return 0.0
    ordenados = sorted(valores)
    n = len(ordenados)
    mitad = n // 2
    if n % 2 == 0:
        return (ordenados[mitad - 1] + ordenados[mitad]) / 2
    return ordenados[mitad]


def distribucion_por_categoria(registros, campo):
    """
    registros: lista de dicts (ej. filas de usuarios).
    campo: nombre del campo categorico (ej. 'sexo').
    Regresa {categoria: cantidad, ...} y {categoria: porcentaje}.
    """
    valores = [r.get(campo) for r in registros if r.get(campo) is not None]
    conteo = Counter(valores)
    total = sum(conteo.values()) or 1
    porcentajes = {k: round((v / total) * 100, 2) for k, v in conteo.items()}
    return dict(conteo), porcentajes


def resumen_dimensiones(vectores):
    """
    vectores: lista de dicts con llaves r,i,a,s,e,c (0-6 cada una).
    Regresa un resumen por dimension: media, mediana, moda, desviacion estandar.
    """
    dimensiones = ["r", "i", "a", "s", "e", "c"]
    resumen = {}
    for d in dimensiones:
        valores = [v[d] for v in vectores]
        resumen[d.upper()] = {
            "media": round(media(valores), 2),
            "mediana": mediana(valores),
            "moda": moda(valores),
            "desviacion_estandar": round(desviacion_estandar(valores), 2),
            "minimo": min(valores) if valores else 0,
            "maximo": max(valores) if valores else 0,
        }
    return resumen
