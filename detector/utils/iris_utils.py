import numpy as np

def diametro_iris_4p(pontos):
    centro = np.mean(pontos, axis=0)
    distancias = [np.linalg.norm(np.array(p) - centro) for p in pontos]
    desvio = np.std(distancias)
    if desvio < 2:
        return np.mean(distancias) * 2, True
    return None, False

def diametro_iris_3p(pontos):
    d_h = np.linalg.norm(np.array(pontos[0]) - np.array(pontos[1]))
    centro = (np.array(pontos[0]) + np.array(pontos[1])) / 2
    d_v = np.linalg.norm(centro - np.array(pontos[2])) * 2
    return (d_h + d_v) / 2
