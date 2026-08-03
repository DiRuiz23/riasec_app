import os
import shutil
import sqlite3
import datetime
import json

DB_PATH = 'riasec.db'

def crear_backup():
    if not os.path.exists(DB_PATH):
        print(f"[{datetime.datetime.now()}] La base de datos {DB_PATH} no existe. No hay nada que migrar.")
        return False
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    bak_path = f"{DB_PATH}.bak_{timestamp}"
    shutil.copy2(DB_PATH, bak_path)
    print(f"[{datetime.datetime.now()}] Backup creado exitosamente: {bak_path}")
    print(f"Si algo sale mal, restaura el backup copiando {bak_path} a {DB_PATH}")
    return True

def migrar_esquema():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 1. Crear tabla datasets si no existe
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS datasets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_name VARCHAR(255) NOT NULL,
        columns TEXT NOT NULL,
        records TEXT NOT NULL,
        uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    print("Tabla 'datasets' verificada/creada.")
    
    # 2. Insertar dataset legacy (RIASEC) si la tabla esta vacia o no existe el legacy
    cursor.execute("SELECT id FROM datasets WHERE source_name = 'RIASEC_Legacy'")
    legacy_dataset = cursor.fetchone()
    if not legacy_dataset:
        columns_json = json.dumps(["r", "i", "a", "s", "e", "c"])
        cursor.execute(
            "INSERT INTO datasets (source_name, columns, records) VALUES (?, ?, ?)",
            ("RIASEC_Legacy", columns_json, "[]")
        )
        legacy_dataset_id = cursor.lastrowid
        print(f"Dataset legacy creado con ID {legacy_dataset_id}.")
    else:
        legacy_dataset_id = legacy_dataset[0]
        print(f"Dataset legacy ya existia con ID {legacy_dataset_id}.")
        
    # 3. Anadir dataset_id a modelos_entrenados
    # SQLite no soporta IF EXISTS en ADD COLUMN, asi que usamos PRAGMA
    cursor.execute("PRAGMA table_info(modelos_entrenados)")
    columnas = [info[1] for info in cursor.fetchall()]
    
    if "dataset_id" not in columnas:
        cursor.execute("ALTER TABLE modelos_entrenados ADD COLUMN dataset_id INTEGER REFERENCES datasets(id)")
        print("Columna 'dataset_id' anadida a 'modelos_entrenados'.")
    else:
        print("Columna 'dataset_id' ya existia en 'modelos_entrenados'.")
        
    # 4. Asignar dataset_id = legacy_dataset_id a los modelos existentes
    cursor.execute("UPDATE modelos_entrenados SET dataset_id = ? WHERE dataset_id IS NULL", (legacy_dataset_id,))
    filas_afectadas = cursor.rowcount
    if filas_afectadas > 0:
        print(f"Se actualizaron {filas_afectadas} modelos para usar el dataset legacy.")
    else:
        print("No habia modelos pendientes de actualizar con dataset_id.")
        
    # 5. Asegurar que 'activo' sea 0 por defecto (cambiando el default si es posible,
    # aunque en SQLite es complejo alterar constraints, el comportamiento se manejara en SQLAlchemy).
    # Actualizamos el activo a 0 para todos excepto para el ultimo modelo entrenado del legacy dataset,
    # para garantizar que solo uno esta activo.
    cursor.execute("SELECT id FROM modelos_entrenados WHERE dataset_id = ? AND activo = 1 ORDER BY fecha_entrenamiento DESC", (legacy_dataset_id,))
    activos = cursor.fetchall()
    if len(activos) > 1:
        print(f"Se detectaron {len(activos)} modelos activos. Desactivando anteriores...")
        # Dejar solo el ultimo
        modelo_a_mantener = activos[0][0]
        cursor.execute("UPDATE modelos_entrenados SET activo = 0 WHERE dataset_id = ? AND id != ?", (legacy_dataset_id, modelo_a_mantener))
        
    # 6. Verificacion de que no hay modelos con dataset_id = NULL
    cursor.execute("SELECT COUNT(*) FROM modelos_entrenados WHERE dataset_id IS NULL")
    nulos = cursor.fetchone()[0]
    if nulos > 0:
        raise Exception(f"ERROR: {nulos} modelos quedaron con dataset_id NULL.")
    else:
        print("Verificacion exitosa: Ningun modelo tiene dataset_id NULL.")
        
    conn.commit()
    conn.close()
    print("Migracion completada con exito.")

if __name__ == "__main__":
    if crear_backup():
        try:
            migrar_esquema()
        except Exception as e:
            print(f"Fallo durante la migracion: {e}")
            print("Por favor, restaura el backup.")
