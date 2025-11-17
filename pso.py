from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from pandas import read_parquet
from particula import Particula
from knn import KNN


def definirAlgoritmo(tipo_algoritmo):
    if tipo_algoritmo == "KNeighborsClassifier":
        return KNN()
    else:
        return None


def retornarMetricaEscolhida(particula, metrica):
    if metrica == "acuracia":
        return particula.acuracia
    elif metrica == "precisao":
        return particula.precisao
    else:
        return particula.recall


class PSO:
    def __init__(self, tam_enxame, num_interacoes, tipo_algoritmo, endereco_parquet, metrica):
        self.tam_enxame = tam_enxame
        self.num_interacoes = num_interacoes
        self.algoritmo = definirAlgoritmo(tipo_algoritmo)
        self.dataframe = read_parquet(endereco_parquet)
        self.metrica = metrica

        self.melhor_pos_geral = None
        self.melhor_perf_geral = 0

        self.x_treinamento, self.x_teste, self.y_treinamento, self.y_teste = self.definirXY()

    def definirXY(self):
        x = self.dataframe.iloc[:, :-1].values
        y = self.dataframe.iloc[:, -1].values

        return train_test_split(x, y, test_size=0.3)

    def ajustarMetricas(self, pos):
        modelo = self.algoritmo.gerarModelo(pos)
        modelo.fit(self.x_treinamento, self.y_treinamento)
        previsoes = modelo.predict(self.x_teste)

        return (round(accuracy_score(self.y_teste, previsoes) * 100, 1),
                round(precision_score(self.y_teste, previsoes) * 100, 1),
                round(recall_score(self.y_teste, previsoes) * 100, 1))

    def ajustarMelhoresPosicoes(self, particula):
        particula.acuracia, particula.precisao, particula.recall = self.ajustarMetricas(particula.pos)
        performance = retornarMetricaEscolhida(particula, self.metrica)

        if performance > particula.melhor_perf:
            particula.melhor_perf = performance
            particula.melhor_pos = list(particula.pos)
        if particula.melhor_perf > self.melhor_perf_geral:
            self.melhor_perf_geral = particula.melhor_perf
            self.melhor_pos_geral = list(particula.melhor_pos)

    def gerarEnxame(self):
        enxame = []

        for _ in range(self.tam_enxame):
            particula = Particula(self.algoritmo)
            self.ajustarMelhoresPosicoes(particula)
            enxame.append(particula)

        return enxame

    def mover(self, particula):
        particula.pos, particula.vel = particula.ajustarPosicaoVelocidade(self.melhor_pos_geral)
        particula.pos = self.algoritmo.ajustarPosicao(particula.pos)
        particula.pos = self.algoritmo.ajustarLimites(particula.pos)

        self.ajustarMelhoresPosicoes(particula)

    def escreverPrimeiraInteracao(self, file, enxame):
        file.write("1* Interation:\n")

        for i in range(self.tam_enxame):
            file.write(str(enxame[i]) + "\n")

    def escreverDemaisInteracoes(self, file, enxame):
        for num_interacao in range(1, self.num_interacoes):
            file.write("\n{}* Interation:\n".format(num_interacao + 1))

            for i in range(self.tam_enxame):
                self.mover(enxame[i])
                file.write(str(enxame[i]) + "\n")

    def executar(self):
        enxame = self.gerarEnxame()

        with open("PSO.txt", "w") as file:
            self.escreverPrimeiraInteracao(file, enxame)
            self.escreverDemaisInteracoes(file, enxame)

            file.write("\nBest Position: {}\nBest Perfomance: {}%".
                       format(self.melhor_pos_geral, self.melhor_perf_geral))
