from catboost import CatBoostClassifier
from algoritmo import Algoritmo


class CBC(Algoritmo):
    def __str__(self):
        return "CAT BOOST CLASSIFIER"

    def definirMinimo(self):
        return [50, 0.0001, 3, 0.001, 0.0, 0.0]

    def definirMaximo(self):
        return [2000, 0.3, 10, 20, 10, 1]

    def gerarModelo(self, pos):
        return CatBoostClassifier(iterations=pos[0], learning_rate=pos[1], depth=pos[2],
                                  l2_leaf_reg=pos[3], random_strength=pos[4], bagging_temperature=pos[5])
