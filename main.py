from pso import PSO

pso = PSO(10, 10, 'LogisticRegression', 'Mimic IV.parquet', 'recall')
pso.executar()