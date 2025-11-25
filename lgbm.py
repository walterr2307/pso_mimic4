from lightgbm import LGBMClassifier
from algoritmo import Algoritmo


class LGBM(Algoritmo):
    def __str__(self):
        return "LIGHT GBM CLASSIFIER"

    def definirMinimo(self):
        return [100, 0.01, 16, 3, 0.5, 0.5, 5, 0.0, 0.0]

    def definirMaximo(self):
        return [2000, 0.3, 128, 15, 1, 1, 50, 5, 5]

    def gerarModelo(self, pos):
        return LGBMClassifier(n_estimators=pos[0], learning_rate=pos[1], num_leaves=pos[2],
                              max_depth=pos[3], subsample=pos[4], colsample_bytree=pos[5],
                              min_child_samples=pos[6], reg_lambda=pos[7], reg_alpha=pos[8])
