from algoritmo import Algoritmo
from sklearn.linear_model import LogisticRegression


class LR(Algoritmo):
    def __str__(self):
        return 'LOGISTIC REGRESSION'

    def definirMinimo(self):
        return [0.01, 100]

    def definirMaximo(self):
        return [10, 500]

    def definirIndices(self):
        return [0]

    def gerarModelo(self, pos):
        return LogisticRegression(
            C=pos[0],
            max_iter=pos[1],
            solver='liblinear',
            class_weight='balanced',
            penalty='l2'
        )
