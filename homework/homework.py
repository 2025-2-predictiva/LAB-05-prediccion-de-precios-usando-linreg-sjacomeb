#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#

import os
import json
import pickle
import zipfile
import gzip
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    median_absolute_error,
)

# ===================== PREPROCESAMIENTO ===================== #

def preparar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajustes básicos para el dataset de vehículos:
    - copia del DataFrame
    - crea columna Age a partir de Year
    - elimina columnas poco útiles
    - elimina filas con valores faltantes
    """
    datos = df.copy()
    datos["Age"] = 2021 - datos["Year"]
    datos = datos.drop(columns=["Year", "Car_Name"])
    datos = datos.dropna()
    return datos


# ===================== PIPELINE DEL MODELO ===================== #

def construir_pipeline() -> Pipeline:
    """
    Define el pipeline de regresión:
    - codificación de categóricas
    - escalado de numéricas
    - selección de variables
    - modelo lineal
    """
    cols_categoricas = ["Fuel_Type", "Selling_type", "Transmission"]
    cols_numericas = ["Selling_Price", "Driven_kms", "Age", "Owner"]

    preprocesador = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cols_categoricas),
            ("num", MinMaxScaler(), cols_numericas),
        ],
        remainder="passthrough",
    )

    selector = SelectKBest(score_func=f_regression)

    modelo = Pipeline(
        steps=[
            ("preprocesador", preprocesador),
            ("selector", selector),
            ("regresor", LinearRegression()),
        ]
    )

    return modelo


def ajustar_hiperparametros(
    modelo: Pipeline,
    n_splits: int,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scoring: str,
) -> GridSearchCV:
    """
    Ajusta el número de variables seleccionadas (k) usando GridSearchCV.
    """
    grid = {
        "selector__k": range(1, 13),
    }

    gs = GridSearchCV(
        estimator=modelo,
        param_grid=grid,
        cv=n_splits,
        refit=True,
        scoring=scoring,
    )

    gs.fit(X_train, y_train)
    return gs


# ===================== MÉTRICAS ===================== #

def evaluar_modelo(
    modelo,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
):
    """
    Calcula R², MSE y MAD para train y test.
    """
    y_pred_train = modelo.predict(X_train)
    y_pred_test = modelo.predict(X_test)

    metrics_train = {
        "type": "metrics",
        "dataset": "train",
        "r2": r2_score(y_train, y_pred_train),
        "mse": mean_squared_error(y_train, y_pred_train),
        "mad": median_absolute_error(y_train, y_pred_train),
    }

    metrics_test = {
        "type": "metrics",
        "dataset": "test",
        "r2": r2_score(y_test, y_pred_test),
        "mse": mean_squared_error(y_test, y_pred_test),
        "mad": median_absolute_error(y_test, y_pred_test),
    }

    return metrics_train, metrics_test


# ===================== PERSISTENCIA ===================== #

def guardar_modelo(modelo) -> None:
    """
    Guarda el modelo entrenado comprimido en gzip.
    """
    os.makedirs("files/models", exist_ok=True)
    with gzip.open("files/models/model.pkl.gz", "wb") as f:
        pickle.dump(modelo, f)


def guardar_metricas(lista_metricas) -> None:
    """
    Escribe las métricas en formato JSONL.
    """
    os.makedirs("files/output", exist_ok=True)
    with open("files/output/metrics.json", "w", encoding="utf-8") as f:
        for registro in lista_metricas:
            f.write(json.dumps(registro) + "\n")


# ===================== CARGA DE DATOS ===================== #

def cargar_desde_zip(ruta_zip: str, nombre_interno: str) -> pd.DataFrame:
    """
    Lee un CSV ubicado dentro de un archivo ZIP.
    """
    with zipfile.ZipFile(ruta_zip, "r") as zf:
        with zf.open(nombre_interno) as f:
            return pd.read_csv(f)


# ===================== MAIN ===================== #

if __name__ == "__main__":
    # Carga de datos
    df_test = cargar_desde_zip("files/input/test_data.csv.zip", "test_data.csv")
    df_train = cargar_desde_zip("files/input/train_data.csv.zip", "train_data.csv")

    print("Preparando datos...")
    df_test = preparar_datos(df_test)
    df_train = preparar_datos(df_train)

    X_train, y_train = df_train.drop("Present_Price", axis=1), df_train["Present_Price"]
    X_test, y_test = df_test.drop("Present_Price", axis=1), df_test["Present_Price"]

    print("Construyendo pipeline y ajustando hiperparámetros...")
    base_pipeline = construir_pipeline()
    modelo_ajustado = ajustar_hiperparametros(
        base_pipeline,
        n_splits=10,
        X_train=X_train,
        y_train=y_train,
        scoring="neg_mean_absolute_error",
    )

    print("Guardando modelo...")
    guardar_modelo(modelo_ajustado)

    print("Calculando métricas...")
    m_train, m_test = evaluar_modelo(
        modelo_ajustado, X_train, y_train, X_test, y_test
    )

    print("Guardando métricas...")
    guardar_metricas([m_train, m_test])

    print("Proceso finalizado.")