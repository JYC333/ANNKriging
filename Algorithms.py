import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
from scipy.optimize import least_squares
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Bidirectional, Dense
from torch.autograd import Variable

"""
Data structure
[x,y,z-value]
Data with auxiliary variable
[x,y,z-value,auxiliary variables]
"""

np.set_printoptions(suppress=True)


class ANNModel(nn.Module):
    def __init__(self, input_dim, points_dim):
        super(ANNModel, self).__init__()
        self.fc_x1 = nn.Linear(input_dim, points_dim * (input_dim - 2))
        self.fc_x2 = nn.Linear(points_dim, points_dim * (input_dim - 2))

        self.fc_e1 = nn.Linear(input_dim, points_dim * (input_dim - 2))

        self.fc1 = nn.Linear(points_dim * (input_dim - 2) + 1, 1)

        self.fc2 = nn.Linear(points_dim * (input_dim - 2) * 2, 1)

    def forward(self, x, e):
        CC = self.fc_x1(x)
        CC = CC.transpose(1, 2)
        CC = self.fc_x2(CC)
        CC = CC.transpose(1, 2)

        C0 = self.fc_e1(e)
        C0 = C0.view(C0.shape[0], C0.shape[1], 1)

        W = torch.cat([CC, C0], -1)
        W = self.fc1(W)
        W = W.view(W.shape[0], -1)

        vars = x[:, :, 2:].clone().detach().requires_grad_(True)
        vars = vars.view(vars.shape[0], -1)

        out = torch.cat([vars, W], -1)
        out = self.fc2(out).view(-1)

        return out


def ANNKriging(X_train_x, X_train_e, y_train):
    for i in range(len(X_train_x)):
        X_train_x[i, :, :2], X_train_e[i, :2] = NomalizePoint(
            X_train_x[i, :, :2], X_train_e[i, :2]
        )
    X_train_x = torch.from_numpy(X_train_x).type(torch.float)
    X_train_e = torch.from_numpy(X_train_e).type(torch.float)
    y_train = torch.from_numpy(y_train).type(torch.float)

    train = torch.utils.data.TensorDataset(X_train_x, X_train_e, y_train)

    train_loader = torch.utils.data.DataLoader(train, shuffle=False)

    model = ANNModel(X_train_x.shape[2], X_train_x.shape[1])
    error = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    for epoch in range(200):
        for i, (x, e, y) in enumerate(train_loader):
            x = Variable(x)
            e = Variable(e)
            y = Variable(y)

            optimizer.zero_grad()  # Clear gradients
            outputs = model(x, e)  # Forward propagation
            loss = error(outputs, y)  # Calculate softmax and cross entropy loss
            loss.backward()  # Calculating gradients
            optimizer.step()  # Update parameters

    preds = []
    for i, (x, e, y) in enumerate(train_loader):
        x = Variable(x)
        e = Variable(e)

        pred = model(x, e)  # Forward propagation
        preds.append(float(pred))

    return model, np.array(preds)


def BiLSTM(timestep, attributes):
    input = keras.Input(shape=(timestep, attributes))
    h = Bidirectional(LSTM(8, return_sequences=True))(input)
    h = Bidirectional(LSTM(16))(h)
    output = Dense(1)(h)

    model = keras.Model(inputs=input, outputs=output)
    # model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mean_squared_error",
        metrics=["mse", "mae"],
    )

    return model


# Calculating Euclidean Distance
def dist(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# Calculating semi-variogram value using semi-variogram function
def CovMartixSF(points, para, model="Sph"):
    l = len(points)
    cc = np.zeros((l, l))
    for i in range(l):
        for j in range(l):
            cc[i][j] = semi_variogram(para, dist(points[i], points[j]), model)
    return cc


# Calculating semi-variogram value (points pair)
def semi_variogram_points_pair(points, value, covalue=None):
    l = len(points)
    experiments = {}
    dis = []
    for i in range(l):
        for j in range(i + 1, l):
            dis.append(dist(points[i], points[j]))
    dis = np.array(dis)
    bins = np.linspace(0, np.nanmax(dis), 11)[1:]
    for dd in bins:
        experiments[dd] = []

    for i in range(l):
        for j in range(i + 1, l):
            dd = dist(points[i], points[j])
            if (bins > dd).any():
                if covalue is None:
                    experiments[bins[bins > dd][0]].append(pow(value[i] - value[j], 2))
                else:
                    experiments[bins[bins > dd][0]].append(
                        (value[i] - value[j]) * (covalue[i] - covalue[j])
                    )
    for key in experiments.keys():
        experiments[key] = np.average(experiments[key]) / 2

    return bins, experiments


# Semi-variogram function
def semi_variogram(para, dis, model="Sph"):
    C0, a, C = para
    if model == "Sph":
        if dis == 0:
            return 0
        elif 0 < dis <= a:
            return C0 + C * (3 * dis / a - pow(dis / a, 3)) / 2
        elif dis > a:
            return C0 + C
    elif model == "Exp":
        if dis == 0:
            return 0
        elif dis > 0:
            return C0 + C * (1 - np.exp(-dis / a))
    elif model == "Gau":
        if dis == 0:
            return 0
        elif dis > 0:
            return C0 + C * (1 - np.exp(-pow(dis / a, 2)))


# Error function for semi-variogram
def errorS(para, dis, y):
    err = np.zeros(len(dis))
    for i in range(len(dis)):
        err[i] = semi_variogram(para, dis[i], "Sph")
    return err - y


def NomalizePoint(points, estimatePoint=None):
    if estimatePoint is None:
        mx, my = np.mean(points[:, 0]), np.mean(points[:, 1])
        sx, sy = np.std(points[:, 0]), np.std(points[:, 1])
        if sx == 0:
            points[:, 0] = points[:, 0] - mx
        else:
            points[:, 0] = (points[:, 0] - mx) / sx
        if sy == 0:
            points[:, 1] = points[:, 1] - my
        else:
            points[:, 1] = (points[:, 1] - my) / sy

        return points
    else:
        mx, my = np.mean(np.append(points[:, 0], estimatePoint[0])), np.mean(
            np.append(points[:, 1], estimatePoint[1])
        )
        sx, sy = np.std(np.append(points[:, 0], estimatePoint[0])), np.std(
            np.append(points[:, 1], estimatePoint[1])
        )
        if sx == 0:
            points[:, 0] = points[:, 0] - mx
            estimatePoint[0] -= mx
        else:
            points[:, 0] = (points[:, 0] - mx) / sx
            estimatePoint[0] = (estimatePoint[0] - mx) / sx
        if sy == 0:
            points[:, 1] = points[:, 1] - my
            estimatePoint[1] -= my
        else:
            points[:, 1] = (points[:, 1] - my) / sy
            estimatePoint[1] = (estimatePoint[1] - my) / sy

        return points, estimatePoint


def Trace_Variograms(points, value, covalue=None):
    bins, experiments = semi_variogram_points_pair(points, value, covalue)
    bin_max = bins[-1]
    experiments = np.array(list(experiments.values()))
    bins = bins[~(np.isnan(experiments) + (experiments == 0))]
    experiments = experiments[~(np.isnan(experiments) + (experiments == 0))]

    if len(experiments) == 0:
        p0 = np.array([0.5, bin_max, 0.5])
        cc = CovMartixSF(points, p0)
        return cc, p0

    if covalue is None:
        if np.var(value) == 0:
            bounds = [1, bins[-1] / 2, 1]
            p0 = np.array([0.5, bins[-1] / 4, 0.5])
        else:
            bounds = [np.var(value), bins[-1] / 2, np.var(value)]
            p0 = np.array([np.var(value) / 2, bins[-1] / 4, np.var(value) / 2])
    else:
        if np.var(value) == 0 and np.var(covalue) == 0:
            bounds = [1, bins[-1] / 2, 1]
            p0 = np.array([0.5, bins[-1] / 4, 0.5])
        else:
            aa = max(np.var(value), np.var(covalue))
            bounds = [aa, bins[-1] / 2, aa]
            p0 = np.array([aa / 2, bins[-1] / 4, aa / 2])

    para = least_squares(errorS, p0, args=(bins, experiments), bounds=(0, bounds))
    p0 = para.x

    cc = CovMartixSF(points, p0)
    return cc, p0


def UniversalKriging(points, estimatePoint, target):
    points, estimatePoint = NomalizePoint(np.array(points), estimatePoint)

    ll = len(points)

    cc, p0 = Trace_Variograms(points, target)

    if ll <= 6:
        ff = np.c_[np.ones((ll, 1)), points[:, 0], points[:, 1]]
    else:
        ff = np.c_[
            np.ones((ll, 1)),
            points[:, 0],
            points[:, 1],
            pow(points[:, 0], 2),
            points[:, 0] * points[:, 1],
            pow(points[:, 1], 2),
        ]
    cc = np.c_[cc, ff]
    if ll <= 6:
        cc = np.r_[cc, np.c_[ff.T, np.zeros((3, 3))]]
    else:
        cc = np.r_[cc, np.c_[ff.T, np.zeros((6, 6))]]

    C0 = np.zeros((ll, 1))
    for i in range(ll):
        C0[i] = semi_variogram(p0, dist(estimatePoint, points[i]))
    if ll <= 6:
        f0 = np.r_[1, estimatePoint]
    else:
        f0 = np.r_[
            1,
            estimatePoint,
            pow(estimatePoint[0], 2),
            estimatePoint[0] * estimatePoint[1],
            pow(estimatePoint[1], 2),
        ]
    C0 = np.r_[C0, f0.reshape((-1, 1))]

    w = np.dot(np.linalg.pinv(cc), C0)
    estimateValue = 0
    for i in range(ll):
        estimateValue += w[i] * target[i]
    return estimateValue[0]


def CoKriging_Ordinary(points, estimatePoint, target, variables):
    points, estimatePoint = NomalizePoint(np.array(points), estimatePoint)
    # points = np.array(points)
    variables = np.array(variables)
    variables_all = np.c_[target, variables]

    ll = len(points)
    n_vars = variables_all.shape[1]

    cc = []
    pp = []
    for variable in variables_all.T:
        c, p = Trace_Variograms(points, variable)
        cc.append(c)
        pp.append(p)

    cc12 = []
    pp12 = []
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            c, p = Trace_Variograms(points, variables_all[:, i], variables_all[:, j])
            cc12.append(c)
            pp12.append(p)

    ind_cc12 = 0
    for i in range(n_vars):
        if i == 0:
            c = np.c_[cc[i], cc12[ind_cc12]]
            ind_cc12 += 1
            for j in range(ind_cc12, n_vars - i - 1):
                c = np.c_[c, cc12[j]]
            ind_cc12 = n_vars - i - 1
        elif i == n_vars - 1:
            c = np.r_[c, np.c_[np.zeros((cc[0].shape[0], cc[0].shape[1] * i)), cc[i]]]
        else:
            temp = np.c_[np.zeros((cc[0].shape[0], cc[0].shape[1] * i)), cc[i]]
            for j in range(ind_cc12, ind_cc12 + n_vars - i - 1):
                temp = np.c_[temp, cc12[j]]
            ind_cc12 += n_vars - i - 1
            c = np.r_[c, temp]
    c = np.triu(c, 1)
    cc = c + c.T

    n_f = 1
    ff = np.c_[np.ones((ll, n_f)), np.zeros((ll, n_f * (n_vars - 1)))]
    for i in range(1, n_vars - 1):
        ff = np.r_[
            ff,
            np.c_[
                np.zeros((ll, n_f * i)),
                np.ones((ll, n_f)),
                np.zeros((ll, n_f * (n_vars - i - 1))),
            ],
        ]
    ff = np.r_[ff, np.c_[np.zeros((ll, n_f * (n_vars - 1))), np.ones((ll, n_f))]]
    cc = np.r_[np.c_[cc, ff], np.c_[ff.T, np.zeros((n_f * n_vars, n_f * n_vars))]]

    C0 = np.zeros((ll * n_vars, 1))
    for i in range(n_vars):
        for j in range(ll):
            if i == 0:
                C0[j + i * ll] = semi_variogram(pp[i], dist(estimatePoint, points[j]))
            else:
                C0[j + i * ll] = semi_variogram(
                    pp12[i - 1], dist(estimatePoint, points[j])
                )
    C0 = np.r_[C0, np.ones((n_f, 1)), np.zeros((n_f * (n_vars - 1), 1))]

    w = np.dot(np.linalg.pinv(cc), C0)

    estimateValue = 0
    for i in range(n_vars):
        for j in range(ll):
            estimateValue += w[j + i * ll] * variables_all[j, i]
    return estimateValue[0]
