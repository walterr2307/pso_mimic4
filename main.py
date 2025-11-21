from pso import PSO

lista = ["KNeighborsClassifier", "RandomForestClassifier", "LogisticRegression",
         "XGBClassifier", "LGBMClassifier", "CatBoostClassifier"]

for item in lista:
    pso = PSO(10, 1, item, 'Mimic IV.parquet')

