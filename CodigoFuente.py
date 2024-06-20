import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Datos en formato CSV
data = "Info.txt"

# Leer los datos en un DataFrame de pandas
df = pd.read_csv(data, delimiter=';')

# Mostrar el DataFrame para verificar la columna sin nombre
print("DataFrame original:\n", df)

# Verificar si hay alguna columna sin nombre
if '' in df.columns:
    # Eliminar la columna sin nombre
    df.drop(columns=[''], inplace=True)

# Convertir todas las columnas a tipo numérico
df = df.apply(pd.to_numeric, errors='coerce')

# Reemplazar valores NaN con ceros
df.fillna(0, inplace=True)

# Seleccionar las columnas que se utilizarán para la clasificación
X = df[['fiDiasPermanencia', 'fnTasaRecuperacion', 'fnTasaLiquidacion', 'fnAforoUtilizado', 'fnInteresOrdinario', 'fnColocacionAnual', 'fnYield']]

# Definir el modelo KMeans con 5 clusters
kmeans = KMeans(n_clusters=5, random_state=0)

# Ajustar el modelo a los datos
kmeans.fit(X)

# Asignar las etiquetas de los clusters a los datos
df['Cluster'] = kmeans.labels_

# Mostrar los datos con sus etiquetas de cluster
print("Datos con etiquetas de cluster:\n", df)

# Agrupar por el campo 'Cluster' y contar la cantidad de elementos en cada grupo
cluster_counts = df.groupby('Cluster').size()

# Mostrar los contadores de cada cluster
print("\nContadores de cada cluster:\n", cluster_counts)

# Calcular los valores característicos de cada cluster
cluster_means = df.groupby('Cluster').mean()
cluster_max = df.groupby('Cluster').max()
cluster_min = df.groupby('Cluster').min()

# Combinar medias, máximos y mínimos en un solo DataFrame
cluster_characteristics = cluster_means.copy()
for column in cluster_means.columns:
    cluster_characteristics[column + '_max'] = cluster_max[column]
    cluster_characteristics[column + '_min'] = cluster_min[column]

# Mostrar los valores característicos de cada cluster
print("\nValores característicos de cada cluster (media, máximo, mínimo):\n", cluster_characteristics)

# Realizar una reducción de dimensionalidad con PCA para visualizar en 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Crear un DataFrame para los datos transformados y las etiquetas de los clusters
df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
df_pca['Cluster'] = df['Cluster']

# Graficar los datos agrupados por clusters
plt.figure(figsize=(10, 6))
colors = ['red', 'blue', 'green', 'purple', 'orange']
for cluster in df_pca['Cluster'].unique():
    cluster_data = df_pca[df_pca['Cluster'] == cluster]
    plt.scatter(cluster_data['PC1'], cluster_data['PC2'], label=f'Cluster {cluster}', color=colors[cluster])

plt.title('Visualización de Clusters con PCA')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend()
plt.show()