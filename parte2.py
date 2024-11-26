import pandas as pd
import numpy as np
from scipy import stats

# Cargar los datos
df = pd.read_csv("titanic.csv")

# 1. Construir un intervalo de confianza (95%) para la edad promedio
def intervalo_confianza_edad(data, confianza=0.95):
    media = data.mean()
    sem = stats.sem(data, nan_policy='omit')  # Error estándar de la media
    intervalo = stats.t.interval(confianza, len(data.dropna()) - 1, loc=media, scale=sem)
    return media, intervalo

# Calcular intervalo para la edad promedio
media_edad, intervalo_edad = intervalo_confianza_edad(df['age'])
print(f"Media de edad: {media_edad:.2f}, Intervalo de confianza al 95%: {intervalo_edad}")

# 2. Promedio de edad de mujeres y hombres > 56 años
def prueba_promedio(data, genero, valor_esperado=56, confianza=0.95):
    subset = data[data['sex'] == genero]['age']
    t_stat, p_valor = stats.ttest_1samp(subset.dropna(), valor_esperado)
    conclusion = p_valor / 2 < 1 - confianza  # Prueba una cola
    return t_stat, p_valor, conclusion

# Pruebas de hipótesis para mujeres y hombres
t_mujeres, p_mujeres, concl_mujeres = prueba_promedio(df, 'female', 56)
t_hombres, p_hombres, concl_hombres = prueba_promedio(df, 'male', 56)
print(f"Mujeres - t: {t_mujeres:.2f}, p: {p_mujeres:.4f}, >56 años: {'Sí' if concl_mujeres else 'No'}")
print(f"Hombres - t: {t_hombres:.2f}, p: {p_hombres:.4f}, >56 años: {'Sí' if concl_hombres else 'No'}")

# 3. Diferencia significativa en la tasa de supervivencia (hombres vs mujeres, clases)
def prueba_diferencia_tasas(data, grupo1, grupo2, col='survived', confianza=0.99):
    surv1 = data[data['sex'] == grupo1][col]
    surv2 = data[data['sex'] == grupo2][col]
    t_stat, p_valor = stats.ttest_ind(surv1, surv2, nan_policy='omit', equal_var=False)
    conclusion = p_valor < 1 - confianza
    return t_stat, p_valor, conclusion

# Tasa de supervivencia por género
t_sex, p_sex, concl_sex = prueba_diferencia_tasas(df, 'male', 'female')
print(f"Tasa de supervivencia - Hombres vs Mujeres - t: {t_sex:.2f}, p: {p_sex:.4f}, Diferencia significativa: {'Sí' if concl_sex else 'No'}")

# Tasa de supervivencia por clase
for clase1 in df['pclass'].unique():
    for clase2 in df['pclass'].unique():
        if clase1 < clase2:
            surv_clase1 = df[df['pclass'] == clase1]['survived']
            surv_clase2 = df[df['pclass'] == clase2]['survived']
            t_clases, p_clases = stats.ttest_ind(surv_clase1, surv_clase2, nan_policy='omit', equal_var=False)
            print(f"Clase {clase1} vs Clase {clase2} - t: {t_clases:.2f}, p: {p_clases:.4f}")

# 4. Mujeres más jóvenes que hombres
def prueba_edad_promedio(data, confianza=0.95):
    edad_mujeres = data[data['sex'] == 'female']['age']
    edad_hombres = data[data['sex'] == 'male']['age']
    t_stat, p_valor = stats.ttest_ind(edad_mujeres, edad_hombres, nan_policy='omit', equal_var=False)
    conclusion = p_valor < 1 - confianza
    return t_stat, p_valor, conclusion

# Prueba para edades
t_edades, p_edades, concl_edades = prueba_edad_promedio(df)
print(f"Edades - Mujeres vs Hombres - t: {t_edades:.2f}, p: {p_edades:.4f}, Mujeres más jóvenes: {'Sí' if concl_edades else 'No'}")
