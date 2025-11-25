from sklearn.ensemble import RandomForestClassifier
from algoritmo import Algoritmo


class RFC(Algoritmo):
    def __str__(self):
        return 'RANDOM FOREST CLASSIFIER'

    def definirMinimo(self):
        return [100, 5, 2, 1, 0.1]

    def definirMaximo(self):
        return [500, 50, 50, 20, 1]

    def gerarModelo(self, pos):
        return RandomForestClassifier(n_estimators=pos[0], max_depth=pos[1], min_samples_split=pos[2],
                                      min_samples_leaf=pos[3], max_features=pos[4])
