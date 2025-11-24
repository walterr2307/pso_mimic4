from pso import PSO

lista = ["KNeighborsClassifier", "RandomForestClassifier", "LogisticRegression",
         "XGBClassifier", "LGBMClassifier", "CatBoostClassifier"]

for item in lista:
    print(item)
    PSO(10, 1, item, 'Mimic IV.parquet')
