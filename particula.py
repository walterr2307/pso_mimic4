import numpy as np
from random import uniform


class Particula:
    def __init__(self, algoritmo):
        self.pos = algoritmo.definirPosicaoAleatoria()
        self.vel = algoritmo.definirVelocidade(self.pos)
        self.melhor_pos = list(self.pos)
        self.melhor_perf = 0

        self.acuracia = 0
        self.precisao = 0
        self.recall = 0

        self.w = uniform(0.5, 1)
        self.c1 = uniform(0, 2)
        self.c2 = uniform(0, 2)

    def __str__(self):
        performance = self.retornarPerformance()

        return ('Posicao: {} -> Acuracia: {}%, Precisao: {}%, Recall: {}%, Performance: {}%'.
                format(self.pos, self.acuracia, self.precisao, self.recall, performance))

    def mover(self, melhor_pos_geral):
        pos = np.array(self.pos, dtype=float)
        vel = np.array(self.vel, dtype=float)
        melhor_pos = np.array(self.melhor_pos, dtype=float)
        melhor_pos_geral = np.array(melhor_pos_geral, dtype=float)

        r1 = uniform(0, 1)
        r2 = uniform(0, 1)

        vel = vel * self.w + r1 * self.c1 * (melhor_pos - pos) + r2 * self.c2 * (melhor_pos_geral - pos)

        return list(pos + vel), list(vel)

    def retornarPerformance(self):
        return round((self.acuracia + 2 * self.precisao + 4 * self.recall) / 7, 1)
