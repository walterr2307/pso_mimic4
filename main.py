from pso import PSO

lista = ["KNeighborsClassifier", "RandomForestClassifier", "LogisticRegression",
         "XGBClassifier", "LGBMClassifier", "CatBoostClassifier"]

for item in lista:
    pso = PSO(10, 10, lista[5], 'Mimic IV.parquet')
    pso.executar()
