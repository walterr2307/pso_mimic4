from pso import PSO

lista = ["KNeighborsClassifier", "RandomForestClassifier", "LogisticRegression",
         "XGBClassifier", "LGBMClassifier", "CatBoostClassifier"]

pso = PSO(10, 10, 'RandomForestClassifier', 'Mimic IV.parquet', 'recall')
pso.executar()
