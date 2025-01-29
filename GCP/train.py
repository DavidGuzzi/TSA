import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import argparse

# Parsear argumentos
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True, help="Ruta al archivo CSV en Cloud Storage")
parser.add_argument("--model_dir", type=str, required=True, help="Directorio donde guardar el modelo")
args = parser.parse_args()

# Cargar los datos
print(f"Cargando datos desde {args.data_path}")
df = pd.read_csv(args.data_path)

# Separar características y target
X = df.drop(columns=["target"])
y = df["target"]

# Dividir en datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluar el modelo
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {accuracy:.4f}")

# Guardar el modelo en Cloud Storage
model_filename = "model.joblib"
joblib.dump(model, model_filename)

# Guardar en Cloud Storage
import subprocess
subprocess.run(["gsutil", "cp", model_filename, f"{args.model_dir}/model.joblib"])

print(f"Modelo guardado en {args.model_dir}/model.joblib")