import numpy as np


def dist(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def moranI(points, value):
    W = np.zeros((len(points), len(points)))
    for i in range(len(points)):
        for j in range(len(points)):
            W[i][j] = dist(points[i], points[j])
    W = W**-3
    row, col = np.diag_indices_from(W)
    W[row, col] = 0

    W = np.array(W)
    X = np.array(value)
    X = X.reshape(1, -1)
    W = W / W.sum(axis=1)
    n = W.shape[0]
    Z = X - X.mean()
    S0 = W.sum()
    S1 = 0
    for i in range(n):
        for j in range(n):
            S1 += 0.5 * (W[i, j] + W[j, i]) ** 2
    S2 = 0
    for i in range(n):
        S2 += (W[i, :].sum() + W[:, i].sum()) ** 2
    # global moran index
    I = np.dot(Z, W)
    I = np.dot(I, Z.T)
    I = n / S0 * I / np.dot(Z, Z.T)

    EI_N = -1 / (n - 1)
    VARI_N = (n**2 * S1 - n * S2 + 3 * S0**2) / (S0**2 * (n**2 - 1)) - EI_N**2
    ZI_N = (I - EI_N) / (VARI_N**0.5)

    EI_R = -1 / (n - 1)
    b2 = 0
    for i in range(n):
        b2 += n * Z[0, i] ** 4
    b2 = b2 / ((Z * Z).sum() ** 2)
    VARI_R = n * ((n**2 - 3 * n + 3) * S1 - n * S2 + 3 * S0**2) - b2 * (
        (n**2 - n) * S1 - 2 * n * S2 + 6 * S0**2
    )
    VARI_R = VARI_R / (S0**2 * (n - 1) * (n - 2) * (n - 3)) - EI_R**2
    ZI_R = (I - EI_R) / (VARI_R**0.5)

    Ii = list()
    for i in range(n):
        Ii_ = n * Z[0, i]
        Ii__ = 0
        for j in range(n):
            Ii__ += W[i, j] * Z[0, j]
        Ii_ = Ii_ * Ii__ / ((Z * Z).sum())
        Ii.append(Ii_)
    Ii = np.array(Ii)

    ZIi = list()
    EIi = Ii.mean()
    VARIi = Ii.var()
    for i in range(n):
        ZIi_ = (Ii[i] - EIi) / (VARIi**0.5)
        ZIi.append(ZIi_)
    ZIi = np.array(ZIi)

    return {
        "I": {"value": I[0, 0]},
        "ZI_N": {"value": ZI_N[0, 0]},
        "ZI_R": {"value": ZI_R[0, 0]},
        "Ii": {"value": Ii},
        "ZIi": {"value": ZIi},
    }
