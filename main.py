from pso import PSO

pso = PSO(10, 10, 'CatBoostClassifier', 'Mimic IV.parquet', 'recall')
pso.executar()