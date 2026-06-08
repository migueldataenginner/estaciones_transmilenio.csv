
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ==========================================
# PROYECTO: AGRUPAMIENTO DE ESTACIONES
# TRANSMILENIO - APRENDIZAJE NO SUPERVISADO
# ==========================================

# Creación del dataset

datos = {
    "Estacion": [
        "Portal Norte",
        "Calle 100",
        "Calle 72",
        "Avenida Jimenez",
        "Portal Sur",
        "Ricaurte",
        "Calle 26",
        "Suba",
        "Tunal",
        "Banderas"
    ],
    "Pasajeros_Dia": [12000, 8000, 9500, 15000, 11000, 7000, 6000, 13000, 5000, 7500],
    "Congestion": [9, 7, 8, 10, 8, 6, 5, 9, 4, 6],
    "Tiempo_Espera": [8, 5, 6, 9, 7, 4, 3, 8, 3, 4],
    "Rutas_Conectadas": [12, 8, 10, 15, 11, 7, 6, 13, 5, 7]
}

df = pd.DataFrame(datos)

print("\nDATASET UTILIZADO")
print(df)

# Selección de variables numéricas

X = df[[
    "Pasajeros_Dia",
    "Congestion",
    "Tiempo_Espera",
    "Rutas_Conectadas"
]]

# Normalización de datos

scaler = StandardScaler()
X_escalado = scaler.fit_transform(X)

# Modelo K-Means

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_escalado)

# Resultados

print("\nRESULTADOS DEL AGRUPAMIENTO")
print(df[["Estacion", "Cluster"]])

# Interpretación de grupos

print("\nESTACIONES POR CLUSTER")

for cluster in sorted(df["Cluster"].unique()):
    print(f"\nCluster {cluster}:")
    estaciones = df[df["Cluster"] == cluster]["Estacion"]

    for estacion in estaciones:
        print("-", estacion)

# Visualización gráfica

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Pasajeros_Dia"],
    df["Congestion"],
    c=df["Cluster"]
)

for i in range(len(df)):
    plt.text(
        df["Pasajeros_Dia"][i],
        df["Congestion"][i],
        df["Estacion"][i]
    )

plt.xlabel("Pasajeros por Día")
plt.ylabel("Nivel de Congestión")
plt.title("Agrupamiento de Estaciones TransMilenio")
plt.grid(True)

plt.show()
