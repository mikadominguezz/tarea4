import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

titanic_data = pd.read_csv('titanic.csv')

# primero se corrigen los valors nulos
missing_ages_before = titanic_data['age'].isnull().sum()

# media segun el genero
mean_age_by_gender = titanic_data.groupby('gender')['age'].mean()

# reemplazo los nulos en "age" por la media calculada arriba
titanic_data['age'] = titanic_data.apply(
    lambda row: mean_age_by_gender[row['gender']] if pd.isnull(row['age']) else row['age'], axis=1
)

missing_ages_after = titanic_data['age'].isnull().sum()

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

general_survival_rate = titanic_data['survived'].mean()
survival_rate_by_gender = titanic_data.groupby('gender')['survived'].mean()

survival_rates = {
    "Tasa de supervivencia general": general_survival_rate,
    "Tasa de supervivencia por género": survival_rate_by_gender.to_dict()
}

# histograma
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

# diagramas de cajas
survived_ages = titanic_data[titanic_data['survived'] == 1]['age']
not_survived_ages = titanic_data[titanic_data['survived'] == 0]['age']

plt.figure(figsize=(10, 6))
plt.boxplot([survived_ages, not_survived_ages], labels=['Supervivientes', 'No supervivientes'], patch_artist=True)
plt.title('Diagrama de Cajas: Edades de Supervivientes vs No Supervivientes')
plt.ylabel('Edad')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

{
    "Valores nulos antes y después de corrección": (missing_ages_before, missing_ages_after),
    "Estadísticas descriptivas": age_statistics,
    "Tasas de supervivencia": survival_rates
}
