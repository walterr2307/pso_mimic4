from pso import PSO

pso = PSO(10, 10, 'LGBMClassifier', 'Mimic IV.parquet', 'recall')
pso.executar()