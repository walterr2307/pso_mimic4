from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from pyarrow.parquet import ParquetFile
from particula import Particula
from pandas import concat
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


def lerEmBlocos(endereco_parquet, tamanho_bloco):
    table = ParquetFile(endereco_parquet)
    dfs = []

    buffer = []
    total_no_buffer = 0

    for i in range(table.num_row_groups):
        df_chunk = table.read_row_group(i).to_pandas()

        while len(df_chunk) > 0:
            faltam = tamanho_bloco - total_no_buffer

            if len(df_chunk) <= faltam:
                buffer.append(df_chunk)
                total_no_buffer += len(df_chunk)
                break

            buffer.append(df_chunk.iloc[:faltam])
            dfs.append(concat(buffer, ignore_index=True))

            df_chunk = df_chunk.iloc[faltam:]
            buffer = []
            total_no_buffer = 0

    if buffer:
        dfs.append(concat(buffer, ignore_index=True))

    return dfs


class PSO:
    def __init__(self, tam_enxame, num_interacoes, tipo_algoritmo, endereco_parquet):
        self.tam_enxame = tam_enxame
        self.num_interacoes = num_interacoes
        self.algoritmo = definirAlgoritmo(tipo_algoritmo)

        self.x_treinamento = None
        self.y_treinamento = None
        self.x_validacao = None
        self.y_validacao = None
        self.x_teste = None
        self.y_teste = None

        self.dataframe = None
        self.melhor_particula_geral = None
        self.melhor_pos_geral = None
        self.melhor_perf_geral = 0
        self.num_interacao = 1

        self.comecarLoopPrincipal(endereco_parquet)

    def definirXY(self):
        x = self.dataframe.iloc[:, :-1].values
        y = self.dataframe.iloc[:, -1].values
        scaler = MinMaxScaler()

        x_train, x_rest, y_train, y_rest = train_test_split(
            x, y, test_size=0.3, random_state=42, stratify=y
        )

        x_val, x_test, y_val, y_test = train_test_split(
            x_rest, y_rest, test_size=0.5, random_state=42, stratify=y_rest
        )

        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)
        x_test = scaler.transform(x_test)

        self.x_treinamento = x_train
        self.y_treinamento = y_train
        self.x_validacao = x_val
        self.y_validacao = y_val
        self.x_teste = x_test
        self.y_teste = y_test

    def comecarLoopPrincipal(self, endereco_parquet):
        blocos = lerEmBlocos(endereco_parquet, tamanho_bloco=10_000)
        enxame = None

        with open(str(self.algoritmo) + ".txt", "w") as file:
            for bloco in blocos:
                self.dataframe = bloco
                self.definirXY()

                if enxame is None:
                    enxame = self.gerarEnxame()

                self.escreverInteracoes(file, enxame)

            self.avaliarMelhorNoTeste(file)

    def ajustarMetricas(self, particula):
        modelo = self.algoritmo.gerarModelo(particula.pos)
        modelo.fit(self.x_treinamento, self.y_treinamento)
        previsoes = modelo.predict(self.x_validacao)

        particula.acuracia = round(accuracy_score(self.y_validacao, previsoes) * 100, 1)
        particula.precisao = round(precision_score(self.y_validacao, previsoes, zero_division=0) * 100, 1)
        particula.recall = round(recall_score(self.y_validacao, previsoes, zero_division=0) * 100, 1)

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
        modelo.fit(self.x_treinamento, self.y_treinamento)
        previsoes = modelo.predict(self.x_teste)

        acuracia = round(accuracy_score(self.y_teste, previsoes) * 100, 1)
        precisao = round(precision_score(self.y_teste, previsoes, zero_division=0) * 100, 1)
        recall = round(recall_score(self.y_teste, previsoes, zero_division=0) * 100, 1)

        print("Performance no teste: {}%, {}%, {}%".format(acuracia, precisao, recall))
        file.write("Performance no teste: {}%, {}%, {}%".format(acuracia, precisao, recall))

    def escreverInteracoes(self, file, enxame):
        for _ in range(self.num_interacoes):
            file.write("\n{}* Interacao:\n".format(self.num_interacao))
            print(str(self.num_interacao) + "* Interacao:\n")

            for i in range(self.tam_enxame):
                if self.num_interacao > 1:
                    self.mover(enxame[i])

                print(str(enxame[i]))
                file.write(str(enxame[i]) + "\n")

            file.write("\nMelhor: {}\n\n".format(self.melhor_particula_geral))
            print('\nMelhor: ' + str(self.melhor_particula_geral) + '\n\n')

            self.num_interacao += 1
