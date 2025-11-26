from pso import PSO

lista = ["KNeighborsClassifier", "RandomForestClassifier", "LogisticRegression",
         "XGBClassifier", "LGBMClassifier", "CatBoostClassifier"]

PSO.definirXY('Mimic IV.parquet')

for item in lista:
    print(item)
    pso = PSO(10, 10, item)
    pso.comecarLoopPrincipal()
