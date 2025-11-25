from xgboost import XGBClassifier
from algoritmo import Algoritmo


class XGB(Algoritmo):
    def __str__(self):
        return "XGBOOST CLASSIFIER"

    def definirMinimo(self):
        return [100, 0.01, 3, 0.5, 0.5, 0, 0, 0, 1]

    def definirMaximo(self):
        return [1000, 0.3, 12, 1, 1, 5, 10, 10, 10]

    def gerarModelo(self, pos):
        return XGBClassifier(n_estimators=pos[0], learning_rate=pos[1], max_depth=pos[2],
                             subsample=pos[3], colsample_bytree=pos[4], gamma=pos[5],
                             reg_lambda=pos[6], reg_alpha=pos[7], min_child_weight=pos[8])
