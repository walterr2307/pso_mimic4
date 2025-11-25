from abc import ABC, abstractmethod
from random import randint, uniform


class Algoritmo(ABC):
    def __init__(self):
        self.min = self.definirMinimo()
        self.max = self.definirMaximo()
        self.qtd_atributos = len(self.max)
        self.indices_float = self.definirIndices()

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def definirMinimo(self):
        pass

    @abstractmethod
    def definirMaximo(self):
        pass

    @abstractmethod
    def gerarModelo(self, pos):
        pass

    def definirIndices(self):
        indices = []

        for i in range(self.qtd_atributos):
            if not isinstance(self.min[i], int):
                indices.append(i)

        return indices

    def definirPosicaoInicial(self):
        pos = []

        for i in range(self.qtd_atributos):
            if i in self.indices_float:
                pos.append((self.min[i] + self.max[i]) / 2)
            else:
                pos.append((self.min[i] + self.max[i]) // 2)

        return pos

    def definirPosicaoAleatoria(self):
        pos = []

        for i in range(self.qtd_atributos):
            if i in self.indices_float:
                pos.append(round(uniform(self.min[i], self.max[i]), 4))
            else:
                pos.append(randint(self.min[i], self.max[i]))

        return pos

    def definirVelocidade(self, pos_aleatoria):
        pos_inicial = self.definirPosicaoInicial()
        vel = []

        for i in range(self.qtd_atributos):
            vel.append(pos_aleatoria[i] - pos_inicial[i])

        return vel

    def ajustarPosicao(self, pos):
        for i in range(self.qtd_atributos):
            if i in self.indices_float:
                pos[i] = float(round(pos[i], 4))
            else:
                pos[i] = int(round(pos[i]))

        return pos

    def ajustarLimites(self, pos):
        for i in range(self.qtd_atributos):
            if pos[i] < self.min[i]:
                pos[i] = self.min[i]
            elif pos[i] > self.max[i]:
                pos[i] = self.max[i]

        return pos
