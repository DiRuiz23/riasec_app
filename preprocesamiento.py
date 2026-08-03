import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def detectar_tipos_columnas(df: pd.DataFrame, exclude_cols=None):
    """
    Infiere el tipo de columnas para aplicar el preprocesamiento adecuado.
    """
    if exclude_cols is None:
        exclude_cols = []
        
    numericas = []
    categoricas = []
    no_utilizables = []
    
    for col in df.columns:
        if col in exclude_cols:
            continue
            
        # Revisar si es ID o fecha
        if "id" in col.lower() or pd.api.types.is_datetime64_any_dtype(df[col]):
            no_utilizables.append(col)
            continue
            
        if pd.api.types.is_numeric_dtype(df[col]):
            # Si tiene muy pocos valores unicos en relacion al tamano, podria ser categorica
            # Pero para clustering es seguro usar StandardScaler en numericas discretas.
            numericas.append(col)
        elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
            # Evitar columnas con demasiada cardinalidad (texto libre)
            n_unique = df[col].nunique()
            if n_unique > 50 and n_unique > len(df) * 0.1:
                no_utilizables.append(col)
            else:
                categoricas.append(col)
        else:
            no_utilizables.append(col)
            
    return {
        "numericas": numericas,
        "categoricas": categoricas,
        "no_utilizables": no_utilizables + exclude_cols
    }

def construir_pipeline(columnas_numericas, columnas_categoricas):
    """
    Construye y devuelve un sklearn Pipeline (ColumnTransformer) no ajustado.
    """
    transformers = []
    
    if columnas_numericas:
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        transformers.append(('num', numeric_transformer, columnas_numericas))
        
    if columnas_categoricas:
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        transformers.append(('cat', categorical_transformer, columnas_categoricas))
        
    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
    return preprocessor
