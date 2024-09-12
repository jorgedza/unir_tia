import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Datos en formato CSV
data = "Info.txt"

# Leer los datos en un DataFrame de pandas
df = pd.read_csv(data, delimiter=';')

df.describe()


nulos_por_columna = df.isnull().sum()

# Mostrar los valores nulos por columna
print(nulos_por_columna)

df = df.drop(columns=[col for col in df.columns if "Unnamed" in col])

nulos_por_columna = df.isnull().sum()

# Mostrar los valores nulos por columna
print(nulos_por_columna)


# Reemplazar valores nulos con 0 para las columnas específicas
df[['fnInteresOrdinario', 'fnColocacionAnual', 'fnYield']] = df[['fnInteresOrdinario', 'fnColocacionAnual', 'fnYield']].fillna(0)

# Reemplazar valores nulos con la media para las columnas específicas
df['fiDiasPermanencia'] = df['fiDiasPermanencia'].fillna(df['fiDiasPermanencia'].mean())
df['fnTasaRecuperacion'] = df['fnTasaRecuperacion'].fillna(df['fnTasaRecuperacion'].mean())
df['fnTasaLiquidacion'] = df['fnTasaLiquidacion'].fillna(df['fnTasaLiquidacion'].mean())

nulos_por_columna = df.isnull().sum()

# Mostrar los valores nulos por columna
print(nulos_por_columna)

from sklearn.preprocessing import MinMaxScaler
# Crear una copia del DataFrame para no modificar el original
df_normalizado = df.copy()

# Seleccionar las columnas que deseas normalizar
columnas_a_normalizar = ['fiDiasPermanencia', 'fnTasaRecuperacion', 'fnTasaLiquidacion', 
                         'fnAforoUtilizado', 'fnInteresOrdinario', 'fnColocacionAnual', 'fnYield']

# Crear un objeto MinMaxScaler
scaler = MinMaxScaler()

# Aplicar la normalización a las columnas seleccionadas
df_normalizado[columnas_a_normalizar] = scaler.fit_transform(df[columnas_a_normalizar])

# Mostrar el DataFrame normalizado
print(df_normalizado)


# Seleccionar las columnas que se utilizarán para la clasificación
X = df_normalizado[['fiDiasPermanencia', 'fnTasaRecuperacion', 'fnTasaLiquidacion', 'fnAforoUtilizado', 'fnInteresOrdinario', 'fnColocacionAnual', 'fnYield']]

# Definir el modelo KMeans
kmeans = KMeans(n_clusters=5, random_state=0)

# Ajustar el modelo a los datos
kmeans.fit(X)


# Asignar las etiquetas de los clusters a los datos
df_normalizado['Cluster'] = kmeans.labels_


# Mostrar los datos con sus etiquetas de cluster
print("Datos con etiquetas de cluster:\n", df_normalizado)

# Agrupar por el campo 'Cluster' y contar la cantidad de elementos en cada grupo
cluster_counts = df_normalizado.groupby('Cluster').size()

# Mostrar los contadores de cada cluster
print("\nContadores de cada cluster:\n", cluster_counts)

# Calcular los valores característicos de cada cluster
cluster_means = df_normalizado.groupby('Cluster').mean()
cluster_max = df_normalizado.groupby('Cluster').max()
cluster_min = df_normalizado.groupby('Cluster').min()

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
df_pca['Cluster'] = df_normalizado['Cluster']

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

# Determinar el rango de k (número de clústeres) a probar
k_range = range(1, 11)  # Probar de 1 a 10 clústeres
inertia = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_pca)
    inertia.append(kmeans.inertia_)

# Graficar el resultado
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia, marker='o')
plt.title('Método del Codo para Determinar el Número Óptimo de Clústeres')
plt.xlabel('Número de Clústeres')
plt.ylabel('Inercia')
plt.grid(True)
plt.show()


# Cálculo de estadísticas descriptivas por clúster
cluster_summary = df_normalizado.groupby('Cluster').agg({
    'fiDiasPermanencia': ['mean', 'std', 'min', 'max'],
    'fnTasaRecuperacion': ['mean', 'std', 'min', 'max'],
    'fnTasaLiquidacion': ['mean', 'std', 'min', 'max'],
    'fnAforoUtilizado': ['mean', 'std', 'min', 'max'],
    'fnInteresOrdinario': ['mean', 'std', 'min', 'max'],
    'fnColocacionAnual': ['mean', 'std', 'min', 'max'],
    'fnYield': ['mean', 'std', 'min', 'max']
}).reset_index()

print(cluster_summary)

cluster_summary.to_csv('cluster_summary.csv', index=False)


import matplotlib.pyplot as plt
import seaborn as sns

# Cálculo de medias de 'fnColocacionAnual' por clúster
mean_colocacion_anual = df_normalizado.groupby('Cluster')['fnColocacionAnual'].mean().reset_index()

# Configuración de la gráfica
plt.figure(figsize=(10, 6))
sns.barplot(x='Cluster', y='fnColocacionAnual', data=mean_colocacion_anual, palette='viridis')

# Personalización de la gráfica
plt.title('Media de Colocación Anual por Clúster')
plt.xlabel('Clúster')
plt.ylabel('Media de Colocación Anual')
plt.xticks(rotation=0)  # Opcional: girar etiquetas del eje x si es necesario

# Mostrar gráfica
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Cálculo de medias de 'fnColocacionAnual' por clúster
mean_colocacion_anual = df_normalizado.groupby('Cluster')['fnTasaRecuperacion'].mean().reset_index()

# Configuración de la gráfica
plt.figure(figsize=(10, 6))
sns.barplot(x='Cluster', y='fnTasaRecuperacion', data=mean_colocacion_anual, palette='viridis')

# Personalización de la gráfica
plt.title('Media de Tasa Recuperacion por Clúster')
plt.xlabel('Clúster')
plt.ylabel('Media de Tasa Recuperacion')
plt.xticks(rotation=0)  # Opcional: girar etiquetas del eje x si es necesario

# Mostrar gráfica
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Cálculo de medias de 'fnColocacionAnual' por clúster
mean_colocacion_anual = df_normalizado.groupby('Cluster')['fiDiasPermanencia'].mean().reset_index()

# Configuración de la gráfica
plt.figure(figsize=(10, 6))
sns.barplot(x='Cluster', y='fiDiasPermanencia', data=mean_colocacion_anual, palette='viridis')

# Personalización de la gráfica
plt.title('Media de Días de Permanencia por Clúster')
plt.xlabel('Clúster')
plt.ylabel('Media de Días de Permanencia')
plt.xticks(rotation=0)  # Opcional: girar etiquetas del eje x si es necesario

# Mostrar gráfica
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

# Cálculo de medias de 'fnColocacionAnual' por clúster
mean_colocacion_anual = df_normalizado.groupby('Cluster')['fnTasaLiquidacion'].mean().reset_index()

# Configuración de la gráfica
plt.figure(figsize=(10, 6))
sns.barplot(x='Cluster', y='fnTasaLiquidacion', data=mean_colocacion_anual, palette='viridis')

# Personalización de la gráfica
plt.title('Media de Tasa de Liquidacion por Clúster')
plt.xlabel('Clúster')
plt.ylabel('Media de Tasa de Liquidacion')
plt.xticks(rotation=0)  # Opcional: girar etiquetas del eje x si es necesario

# Mostrar gráfica
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Cálculo de medias de 'fnColocacionAnual' por clúster
mean_colocacion_anual = df_normalizado.groupby('Cluster')['fnAforoUtilizado'].mean().reset_index()

# Configuración de la gráfica
plt.figure(figsize=(10, 6))
sns.barplot(x='Cluster', y='fnAforoUtilizado', data=mean_colocacion_anual, palette='viridis')

# Personalización de la gráfica
plt.title('Media de Aforo Utilizado por Clúster')
plt.xlabel('Clúster')
plt.ylabel('Media de Aforo Utilizado')
plt.xticks(rotation=0)  # Opcional: girar etiquetas del eje x si es necesario

# Mostrar gráfica
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Cálculo de medias de 'fnColocacionAnual' por clúster
mean_colocacion_anual = df_normalizado.groupby('Cluster')['fnInteresOrdinario'].mean().reset_index()

# Configuración de la gráfica
plt.figure(figsize=(10, 6))
sns.barplot(x='Cluster', y='fnInteresOrdinario', data=mean_colocacion_anual, palette='viridis')

# Personalización de la gráfica
plt.title('Media de Interes Ordinario por Clúster')
plt.xlabel('Clúster')
plt.ylabel('Media de Interes Ordinario')
plt.xticks(rotation=0)  # Opcional: girar etiquetas del eje x si es necesario

# Mostrar gráfica
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Cálculo de medias de 'fnColocacionAnual' por clúster
mean_colocacion_anual = df_normalizado.groupby('Cluster')['fnYield'].mean().reset_index()

# Configuración de la gráfica
plt.figure(figsize=(10, 6))
sns.barplot(x='Cluster', y='fnYield', data=mean_colocacion_anual, palette='viridis')

# Personalización de la gráfica
plt.title('Media de Yield por Clúster')
plt.xlabel('Clúster')
plt.ylabel('Media de Yield')
plt.xticks(rotation=0)  # Opcional: girar etiquetas del eje x si es necesario

# Mostrar gráfica
plt.show()

# Caja de 'fiDiasPermanencia' por clúster
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_normalizado, x='Cluster', y='fiDiasPermanencia',palette='Set1')
plt.title('Dias de Permanencia por Clúster')
plt.xlabel('Clúster')
plt.ylabel('Dias de Permanencia')
plt.show()


# Caja de 'fnTasaRecuperacion' por clúster
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_normalizado, x='Cluster', y='fnTasaRecuperacion',palette='Set1')
plt.title('Tasa de Recuperacion por Clúster')
plt.xlabel('Clúster')
plt.ylabel('Tasa de Recuperacion')
plt.show()

# Caja de 'fnTasaLiquidacion' por clúster
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_normalizado, x='Cluster', y='fnTasaLiquidacion',palette='Set1')
plt.title('Tasa de Liquidacion por Clúster')
plt.xlabel('Clúster')
plt.ylabel('Tasa de Liquidacion')
plt.show()

# Caja de 'fnAforoUtilizado' por clúster
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_normalizado, x='Cluster', y='fnAforoUtilizado',palette='Set1')
plt.title('Aforo Utilizado por Clúster')
plt.xlabel('Clúster')
plt.ylabel('Aforo Utilizado')
plt.show()

# Caja de 'fnInteresOrdinario' por clúster
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_normalizado, x='Cluster', y='fnInteresOrdinario',palette='Set1')
plt.title('Interes Ordinario por Clúster')
plt.xlabel('Clúster')
plt.ylabel('Interes Ordinario')
plt.show()

# Caja de 'fnColocacionAnual' por clúster
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_normalizado, x='Cluster', y='fnColocacionAnual',palette='Set1')
plt.title('Colocacion Anual por Clúster')
plt.xlabel('Clúster')
plt.ylabel('Colocacion Anual')
plt.show()

# Caja de 'fnYield' por clúster
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_normalizado, x='Cluster', y='fnYield',palette='Set1')
plt.title('Yield por Clúster')
plt.xlabel('Clúster')
plt.ylabel('Yield')
plt.show()

