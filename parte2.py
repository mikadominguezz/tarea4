import pandas as pd
import numpy as np
from scipy import stats

titanic_data = pd.read_csv("titanic.csv")

def confidence_interval_age(data, confidence=0.95):
    mean_age = data.mean()
    sem = stats.sem(data, nan_policy='omit')  # Standard error of the mean
    interval = stats.t.interval(confidence, len(data.dropna()) - 1, loc=mean_age, scale=sem)
    return mean_age, interval

mean_age, age_interval = confidence_interval_age(titanic_data['age'])
print(f"Mean age: {mean_age:.2f}, 95% confidence interval: {age_interval}")

def mean_test(data, gender, expected_value=56, confidence=0.95):
    subset = data[data['gender'] == gender]['age']
    t_stat, p_value = stats.ttest_1samp(subset.dropna(), expected_value)
    conclusion = p_value / 2 < 1 - confidence  # One-tailed test
    return t_stat, p_value, conclusion

t_women, p_women, conclusion_women = mean_test(titanic_data, 'female', 56)
t_men, p_men, conclusion_men = mean_test(titanic_data, 'male', 56)
print(f"Women - t: {t_women:.2f}, p: {p_women:.4f}, >56 years: {'Yes' if conclusion_women else 'No'}")
print(f"Men - t: {t_men:.2f}, p: {p_men:.4f}, >56 years: {'Yes' if conclusion_men else 'No'}")

def survival_rate_difference(data, group1, group2, col='survived', confidence=0.99):
    survival1 = data[data['gender'] == group1][col]
    survival2 = data[data['gender'] == group2][col]
    t_stat, p_value = stats.ttest_ind(survival1, survival2, nan_policy='omit', equal_var=False)
    conclusion = p_value < 1 - confidence
    return t_stat, p_value, conclusion

t_gender, p_gender, conclusion_gender = survival_rate_difference(titanic_data, 'male', 'female')
print(f"Survival rate - Men vs Women - t: {t_gender:.2f}, p: {p_gender:.4f}, Significant difference: {'Yes' if conclusion_gender else 'No'}")

for class1 in titanic_data['p_class'].unique():
    for class2 in titanic_data['p_class'].unique():
        if class1 < class2:
            survival_class1 = titanic_data[titanic_data['p_class'] == class1]['survived']
            survival_class2 = titanic_data[titanic_data['p_class'] == class2]['survived']
            t_classes, p_classes = stats.ttest_ind(survival_class1, survival_class2, nan_policy='omit', equal_var=False)
            print(f"Class {class1} vs Class {class2} - t: {t_classes:.2f}, p: {p_classes:.4f}")

def age_mean_difference(data, confidence=0.95):
    women_age = data[data['gender'] == 'female']['age']
    men_age = data[data['gender'] == 'male']['age']
    t_stat, p_value = stats.ttest_ind(women_age, men_age, nan_policy='omit', equal_var=False)
    conclusion = p_value < 1 - confidence
    return t_stat, p_value, conclusion

t_ages, p_ages, conclusion_ages = age_mean_difference(titanic_data)
print(f"Ages - Women vs Men - t: {t_ages:.2f}, p: {p_ages:.4f}, Women younger: {'Yes' if conclusion_ages else 'No'}")

print(titanic_data.groupby('gender')['survived'].mean())