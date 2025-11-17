from algoritmo import Algoritmo
from sklearn.linear_model import LogisticRegression


class LR(Algoritmo):
    def __str__(self):
        return 'LOGISTIC REGRESSION'

    def definirMinimo(self):
        return [0.01, 100, 0, 0]

    def definirMaximo(self):
        return [10, 500, 1, 1]

    def definirIndices(self):
        return [0]

    def gerarModelo(self, pos):
        solvers = ['liblinear', 'saga']
        penalties = ['l1', 'l2']

        return LogisticRegression(C=pos[0], max_iter=pos[1], solver=solvers[pos[2]], penalty=penalties[pos[3]])
