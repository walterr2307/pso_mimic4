from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from particula import Particula
from pandas import read_parquet
from copy import deepcopy

from logistic_regression import LR
from lgbm import LGBM
from knn import KNN
from rfc import RFC
from xgb import XGB
from cbc import CBC


def definirAlgoritmo(tipo_algoritmo):
    if tipo_algoritmo == "KNeighborsClassifier":
        return KNN()
    if tipo_algoritmo == "RandomForestClassifier":
        return RFC()
    if tipo_algoritmo == "LogisticRegression":
        return LR()
    if tipo_algoritmo == "XGBClassifier":
        return XGB()
    if tipo_algoritmo == "LGBMClassifier":
        return LGBM()
    if tipo_algoritmo == "CatBoostClassifier":
        return CBC()

    return None


class PSO:
    x_treinamento = None
    y_treinamento = None
    x_validacao = None
    y_validacao = None
    x_teste = None
    y_teste = None

    def __init__(self, tam_enxame, num_interacoes, tipo_algoritmo):
        self.tam_enxame = tam_enxame
        self.num_interacoes = num_interacoes
        self.algoritmo = definirAlgoritmo(tipo_algoritmo)
        self.melhor_particula_geral = None
        self.melhor_pos_geral = None
        self.melhor_perf_geral = 0

    @staticmethod
    def definirXY(endereco_parquet):
        dataframe = read_parquet(endereco_parquet)
        x = dataframe.iloc[:, :-1].values
        y = dataframe.iloc[:, -1].values
        scaler = MinMaxScaler()

        x_train, x_rest, PSO.y_treinamento, y_rest = train_test_split(
            x, y, test_size=0.3, random_state=42, stratify=y)

        x_val, x_test, PSO.y_validacao, PSO.y_teste = train_test_split(
            x_rest, y_rest, test_size=0.5, random_state=42, stratify=y_rest)

        PSO.x_treinamento = scaler.fit_transform(x_train)
        PSO.x_validacao = scaler.transform(x_val)
        PSO.x_teste = scaler.transform(x_test)

    def comecarLoopPrincipal(self):
        enxame = self.gerarEnxame()

        with open(str(self.algoritmo) + ".txt", "w") as file:
            self.escreverPrimeiraInteracao(file, enxame)
            self.escreverDemaisInteracoes(file, enxame)
            self.avaliarMelhorNoTeste(file)

    def ajustarMetricas(self, particula):
        modelo = self.algoritmo.gerarModelo(particula.pos)
        modelo.fit(PSO.x_treinamento, PSO.y_treinamento)
        previsoes = modelo.predict(PSO.x_validacao)

        particula.acuracia = round(accuracy_score(PSO.y_validacao, previsoes) * 100, 1)
        particula.precisao = round(precision_score(PSO.y_validacao, previsoes, zero_division=0) * 100, 1)
        particula.recall = round(recall_score(PSO.y_validacao, previsoes, zero_division=0) * 100, 1)

    def ajustarMelhoresPosicoes(self, particula):
        self.ajustarMetricas(particula)
        performance = particula.retornarPerformance()

        if performance > particula.melhor_perf:
            particula.melhor_perf = performance
            particula.melhor_pos = list(particula.pos)

        if performance > self.melhor_perf_geral:
            self.melhor_perf_geral = performance
            self.melhor_pos_geral = list(particula.pos)
            self.melhor_particula_geral = deepcopy(particula)

    def gerarEnxame(self):
        enxame = []

        for _ in range(self.tam_enxame):
            particula = Particula(self.algoritmo)
            self.ajustarMelhoresPosicoes(particula)
            enxame.append(particula)

        return enxame

    def mover(self, particula):
        particula.pos, particula.vel = particula.mover(self.melhor_pos_geral)
        particula.pos = self.algoritmo.ajustarPosicao(particula.pos)
        particula.pos = self.algoritmo.ajustarLimites(particula.pos)

        self.ajustarMelhoresPosicoes(particula)

    def avaliarMelhorNoTeste(self, file):
        modelo = self.algoritmo.gerarModelo(self.melhor_pos_geral)
        modelo.fit(PSO.x_treinamento, PSO.y_treinamento)
        previsoes = modelo.predict(PSO.x_teste)

        acuracia = round(accuracy_score(PSO.y_teste, previsoes) * 100, 1)
        precisao = round(precision_score(PSO.y_teste, previsoes, zero_division=0) * 100, 1)
        recall = round(recall_score(PSO.y_teste, previsoes, zero_division=0) * 100, 1)

        file.write("\n\nPerformance no Teste -> Acuracia: {}%, Precisao: {}%, Recall: {}%".
                   format(acuracia, precisao, recall))

    def escreverPrimeiraInteracao(self, file, enxame):
        file.write("1* Interacao:\n")

        for particula in enxame:
            file.write(str(particula) + "\n")

        file.write("\nMelhor {}\n".format(self.melhor_particula_geral))

    def escreverDemaisInteracoes(self, file, enxame):
        for num_interacao in range(1, self.num_interacoes):
            file.write("\n{}* Interacao:\n".format(num_interacao + 1))

            for particula in enxame:
                self.mover(particula)
                file.write(str(particula) + "\n")

            file.write("\nMelhor {}\n".format(self.melhor_particula_geral))
