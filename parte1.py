# Importar librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Cargar los datos del archivo Titanic
titanic_data = pd.read_csv('titanic.csv')

# Paso 1: Identificar y corregir valores nulos en la columna "age"
missing_ages_before = titanic_data['age'].isnull().sum()

# Calcular la media de edades según el género (excluyendo nulos)
mean_age_by_gender = titanic_data.groupby('gender')['age'].mean()

# Reemplazar valores nulos en "age" por la media del género correspondiente
titanic_data['age'] = titanic_data.apply(
    lambda row: mean_age_by_gender[row['gender']] if pd.isnull(row['age']) else row['age'], axis=1
)

missing_ages_after = titanic_data['age'].isnull().sum()

# Paso 2: Calcular estadísticas descriptivas
age_mean = titanic_data['age'].mean()
age_median = titanic_data['age'].median()
age_mode = titanic_data['age'].mode()[0]
age_range = titanic_data['age'].max() - titanic_data['age'].min()
age_variance = titanic_data['age'].var()
age_std_dev = titanic_data['age'].std()

age_statistics = {
    "Media": age_mean,
    "Mediana": age_median,
    "Moda": age_mode,
    "Rango": age_range,
    "Varianza": age_variance,
    "Desviación Estándar": age_std_dev
}

# Paso 3: Tasa de supervivencia general y por género
general_survival_rate = titanic_data['survived'].mean()
survival_rate_by_gender = titanic_data.groupby('gender')['survived'].mean()

survival_rates = {
    "Tasa de supervivencia general": general_survival_rate,
    "Tasa de supervivencia por género": survival_rate_by_gender.to_dict()
}

# Paso 4: Histograma de edades por clase
plt.figure(figsize=(10, 6))
for pclass in sorted(titanic_data['p_class'].unique()):
    subset = titanic_data[titanic_data['p_class'] == pclass]
    plt.hist(subset['age'], bins=15, alpha=0.6, label=f'Clase {pclass}')

plt.title('Histograma de Edades por Clase')
plt.xlabel('Edad')
plt.ylabel('Frecuencia')
plt.legend(title='Clases')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Paso 5: Diagramas de cajas para edades de supervivientes y no supervivientes
survived_ages = titanic_data[titanic_data['survived'] == 1]['age']
not_survived_ages = titanic_data[titanic_data['survived'] == 0]['age']

plt.figure(figsize=(10, 6))
plt.boxplot([survived_ages, not_survived_ages], labels=['Supervivientes', 'No supervivientes'], patch_artist=True)
plt.title('Diagrama de Cajas: Edades de Supervivientes vs No Supervivientes')
plt.ylabel('Edad')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Resultados Finales
{
    "Valores nulos antes y después de corrección": (missing_ages_before, missing_ages_after),
    "Estadísticas descriptivas": age_statistics,
    "Tasas de supervivencia": survival_rates
}
