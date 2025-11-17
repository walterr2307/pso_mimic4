from random import randint
from algoritmo import Algoritmo
from sklearn.neighbors import KNeighborsClassifier


class KNN(Algoritmo):
    def definirMinimo(self):
        return [1, 0, 0, 10]

    def definirMaximo(self):
        return [30, 1, 2, 100]

    def definirIndices(self):
        return []

    def definirPosicaoAleatoria(self):
        return [randint(1, 30), randint(0, 1), randint(0, 2), randint(10, 100)]

    def gerarModelo(self, pos):
        lista_weights = ['uniform', 'distance']
        lista_metric = ['euclidean', 'manhattan', 'minkowski']

        return KNeighborsClassifier(n_neighbors=pos[0], weights=lista_weights[pos[1]],
                                    metric=lista_metric[pos[2]], leaf_size=pos[3])
