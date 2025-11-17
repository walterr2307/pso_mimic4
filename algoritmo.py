from abc import ABC, abstractmethod


class Algoritmo(ABC):
    def __init__(self):
        self.min = self.definirMinimo()
        self.max = self.definirMaximo()
        self.qtd_atributos = len(self.max)
        self.indices_float = self.definirIndices()

    @abstractmethod
    def definirMinimo(self):
        pass

    @abstractmethod
    def definirMaximo(self):
        pass

    @abstractmethod
    def definirIndices(self):
        pass

    @abstractmethod
    def definirPosicaoAleatoria(self):
        pass

    @abstractmethod
    def gerarModelo(self, pos):
        pass

    def definirPosicaoInicial(self):
        pos = []

        for i in range(self.qtd_atributos):
            pos.append(self.max[i] // 2)

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
                pos[i] = round(pos[i], 4)
            else:
                pos[i] = int(round(pos[i]))

        return pos

    def ajustarLimites(self, pos):
        for i in range(self.qtd_atributos):
            if pos[i] < self.min[i]:
                pos[i] = self.min[i]
            if pos[i] > self.max[i]:
                pos[i] = self.max[i]

        return pos
