import gc
import itertools
import math
import os
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import pearsonr

from Algorithms import ANNKriging, BiLSTM, CoKriging_Ordinary, UniversalKriging
from MoranI import moranI

np.set_printoptions(suppress=True)
np.random.seed(0)
np.seterr(all="ignore")
warnings.filterwarnings(action="ignore", message="Mean of empty slice")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# Statistic track data information
def statistic_tracks():
    def split_fun(text):
        a = text.split(",")
        b = "".join(a[1].split()).split("-")[0]
        return pd.Series([b, a[0]], index=["ID", "Nr."])

    data = pd.read_csv("../SBB Data/Rail_Data.csv", usecols=list(range(3)))
    print(data.shape)
    for group in data.groupby(by=["Anlagenuntertyp"]):
        rail = group[1].loc[:, ["Nr./ID"]]
        rail = pd.concat(
            [
                pd.DataFrame(rail["Nr./ID"].apply(split_fun), columns=["ID", "Nr."]),
                rail,
            ],
            axis=1,
        )
        rail.drop(columns=["Nr./ID"], inplace=True)
        groups = rail.groupby(by=["Nr."])
        groups_10 = groups.count() > 10
        groups_10 = groups_10[groups_10].dropna()
        print(group[0], len(groups), len(groups_10))


# Randomly sampling the data
def sample_data():
    # Split rail Nr. and ID
    def split_fun(text):
        a = text.split(",")
        b = "".join(a[1].split()).split("-")[0]
        return pd.Series([b, a[0]], index=["ID", "Nr."])

    data = pd.read_csv("../SBB Data/Rail_Data.csv", usecols=list(range(17)))
    data_sample = pd.DataFrame()

    for group in data.groupby(by=["Anlagenuntertyp"]):
        if "Gleis" in group[0]:
            rail = group[1].loc[
                :,
                [
                    "Nr./ID",
                    "Zustand dezimal",
                    "Note Erneuerung",
                    "Note Stopfen",
                    "Note Schleifen",
                    "Note Schienenwechsel",
                    "Note Weichenbauteilwechsel",
                    "Note Sonstiger Substanzerhalt",
                    "Note Versp?tung P",
                    "Note Sicherheitsrisiko",
                ],
            ]
            rail = pd.concat(
                [
                    pd.DataFrame(
                        rail["Nr./ID"].apply(split_fun), columns=["ID", "Nr."]
                    ),
                    rail,
                ],
                axis=1,
            )
            rail.drop(columns=["Nr./ID"], inplace=True)

            for nr in rail.groupby(by=["Nr."]):
                if len(nr[1]) < 10 and np.random.rand() > 0.8:
                    continue
                data_sample = pd.concat([data_sample, nr[1]])
    data_sample.dropna(axis=1, how="all", inplace=True)
    data_sample.to_csv("data_sample.csv", index=False)


# Randomly sampling the data for pre study
def pre_study():
    data = pd.read_csv("./data_sample.csv")
    data_sample = pd.DataFrame()

    for group in data.groupby(by=["Nr."]):
        if np.random.rand() > 0.1:
            continue
        data_sample = pd.concat([data_sample, group[1]])

    print(data_sample.shape)
    data_sample.to_csv("data_pre_study.csv", index=False)


# Statistic used variables
def statistic_variables():
    data = pd.read_csv("./data_sample.csv")

    used = {}
    variables = [i for i in range(3, len(data.columns))]
    for group in data.groupby(by=["Nr."]):
        points = np.array(group[1], dtype=float)

        t = ""
        for j, var in enumerate(variables):
            moran_test = moranI(points[:, :2].copy(), points[:, var])
            if abs(moran_test["ZI_N"]["value"]) > 1.96:
                a, b = pearsonr(points[:, 2], points[:, var])
                if abs(a) >= 0.6 and b <= 0.05:
                    t += str(var) + ","
        t = t[:-1]
        if t not in used:
            used[t] = 1
        else:
            used[t] += 1
        print(group[0], t)
    print(used)
    with open("variables_used.csv", "w") as f:
        for key in used.keys():
            f.write("%s,%s\n" % (key, used[key]))


# Process rail data using kriging
def kriging_method(file, start_nei=6, end_nei=11):
    data = pd.read_csv("./data_" + file + ".csv")
    output_name_pred = "./predict_UK_" + file + ".csv"
    output_name_mse = "./mse_UK_" + file + ".csv"
    data_save = data.copy()

    mse_write = []
    groups = data.groupby(by=["Nr."])
    for usingRange in range(start_nei, end_nei):
        for num, group in enumerate(groups):
            points = np.array(group[1], dtype=float)
            points[:, 1] = 0

            mse = []
            ind = math.floor(usingRange / 2)
            for i in range(len(points) - usingRange + 1):
                points_ = points[i : i + usingRange].copy()
                tx = points_[ind, :2]
                ty = points_[ind, 2]

                points_ = np.delete(points_, ind, axis=0)

                ind_rail = (data_save["ID"] == tx[0]) & (data_save["Nr."] == group[0])
                value = UniversalKriging(
                    points_[:, :2].copy(), tx.copy(), points_[:, 2]
                )
                data_save.loc[ind_rail, "UK_" + str(usingRange)] = value
                mse.append(pow(value - ty, 2))

            print(usingRange, group[0], len(group[1]), "UK", np.average(mse))
            mse_write.append([usingRange, group[0], "UK", str(np.average(mse))])
            if num % 500 == 0:
                data_save.to_csv(output_name_pred, index=False)
                pd.DataFrame(np.array(mse_write)).to_csv(output_name_mse, index=False)
    data_save.to_csv(output_name_pred, index=False)
    pd.DataFrame(np.array(mse_write)).to_csv(output_name_mse, index=False)


def variables_check(file):
    data = pd.read_csv("./data_" + file + ".csv")

    variables_write = {}
    variables_num = len(data.columns)
    groups = data.groupby(by=["Nr."])
    for num, group in enumerate(groups):
        variables = [i for i in range(3, variables_num)]
        points = np.array(group[1], dtype=float)
        points[:, 1] = 0

        used_variables = []
        for j, var in enumerate(variables):
            moran_test = moranI(points[:, :2].copy(), points[:, var])
            if abs(moran_test["ZI_N"]["value"]) > 1.96:
                a, b = pearsonr(points[:, 2], points[:, var])
                if abs(a) >= 0.6 and b <= 0.05:
                    used_variables.append(str(var))
        print(num, group[0], used_variables)
        variables_write[group[0]] = ";".join(used_variables)
    variables_save = pd.DataFrame.from_dict(
        variables_write, orient="index", columns=["vars"]
    )
    variables_save.to_csv("./variables_" + file + ".csv")
    print(variables_save)


# Process rail data using cokriging
def cokriging_method(file, start_nei=6, end_nei=11):
    data = pd.read_csv("./data_" + file + ".csv")
    output_name_pred = "./predict_Co_" + file + ".csv"
    output_name_mse = "./mse_Co_" + file + ".csv"
    data_save = data.copy()

    vars_check = pd.read_csv(
        "./variables_" + file + ".csv", header=0, names=["Nr.", "vars"]
    )

    mse_write = []
    groups = data.groupby(by=["Nr."])
    for usingRange in range(start_nei, end_nei):
        for num, group in enumerate(groups):
            points = np.array(group[1], dtype=float)
            if len(points) < usingRange:
                continue
            points[:, 1] = 0

            variables = vars_check[vars_check["Nr."] == group[0]]["vars"].values[0]
            if type(variables) is str:
                used_variables = list(map(int, variables.split(";")))

                variables = list(itertools.combinations(used_variables, 1))
                for i in range(2, len(used_variables) + 1):
                    variables += list(itertools.combinations(used_variables, i))
            else:
                continue

            mseco = np.zeros((len(variables),))
            covalue = np.zeros((len(points) - usingRange + 1, len(variables)))
            ind = math.floor(usingRange / 2)
            for i in range(len(points) - usingRange + 1):
                points_ = points[i : i + usingRange].copy()
                tx = points_[ind, :2]
                ty = points_[ind, 2]

                points_ = np.delete(points_, ind, axis=0)

                for j, var in enumerate(variables):
                    covalue[i][j] = CoKriging_Ordinary(
                        points_[:, :2].copy(),
                        tx.copy(),
                        points_[:, 2],
                        points_[:, list(var)],
                    )
                    mseco[j] += pow(covalue[i][j] - ty, 2)

            mseco /= len(points) - usingRange + 1
            min_ind = np.argmin(mseco)
            data_save.loc[
                data_save["Nr."] == group[0], "Co_" + str(usingRange)
            ] = np.r_[
                [np.NaN] * ind, covalue[:, min_ind], [np.NaN] * (usingRange - ind - 1)
            ]

            print(usingRange, group[0], len(group[1]), "Co", mseco[min_ind])
            mse_write.append([usingRange, group[0], "Co", str(mseco[min_ind])])
            if num % 1000 == 0:
                data_save.to_csv(output_name_pred, index=False)
                pd.DataFrame(np.array(mse_write)).to_csv(output_name_mse, index=False)

    data_save.to_csv(output_name_pred, index=False)
    pd.DataFrame(np.array(mse_write)).to_csv(output_name_mse, index=False)


def annkriging_method(file, start_nei=6, end_nei=11):
    data = pd.read_csv("./data_" + file + ".csv")
    output_name_pred = "./predict_AK_" + file + ".csv"
    output_name_mse = "./mse_AK_" + file + ".csv"
    data_save = data.copy()

    vars_check = pd.read_csv(
        "./variables_" + file + ".csv", header=0, names=["Nr.", "vars"]
    )

    mse_write = []
    groups = data.groupby(by=["Nr."])
    for usingRange in range(start_nei, end_nei):
        for num, group in enumerate(groups):
            points = np.array(group[1], dtype=float)
            if len(points) < usingRange:
                continue
            points[:, 1] = 0

            variables = vars_check[vars_check["Nr."] == group[0]]["vars"].values[0]
            if type(variables) is str:
                used_variables = list(map(int, variables.split(";")))
            else:
                used_variables = []

            X_train_x = []
            X_train_e = []
            y_train = []

            ind = math.floor(usingRange / 2)
            for i in range(len(points) - usingRange + 1):
                points_ = points[i : i + usingRange, [0, 1, 2] + used_variables].copy()
                ty = points_[ind, 2]

                X_train_e.append(points_[ind])

                points_ = np.delete(points_, ind, axis=0)

                X_train_x.append(points_)
                y_train.append(ty)

            X_train_x = np.array(X_train_x)
            X_train_e = np.array(X_train_e)
            y_train = np.array(y_train)

            mse = []
            preds = []
            for _ in range(1):
                _, pred = ANNKriging(X_train_x, X_train_e, y_train)
                pred = np.array(pred)
                mse.append(np.average(np.power(y_train - pred, 2)))
                preds.append(pred)
            mse = np.average(mse)
            preds = np.average(np.array(preds), axis=0)

            data_save.loc[
                data_save["Nr."] == group[0], "AK_" + str(usingRange)
            ] = np.r_[[np.NaN] * ind, preds, [np.NaN] * (usingRange - ind - 1)]

            print(usingRange, "AK", group[0], len(group[1]), mse)
            mse_write.append([usingRange, group[0], "AK", str(mse)])
            if num % 100 == 0:
                data_save.to_csv(output_name_pred, index=False)
                pd.DataFrame(np.array(mse_write)).to_csv(output_name_mse, index=False)

    data_save.to_csv(output_name_pred, index=False)
    pd.DataFrame(np.array(mse_write)).to_csv(output_name_mse, index=False)


def bilstm_method(file, start_nei=6, end_nei=11):
    data = pd.read_csv("./data_" + file + ".csv")
    output_name_pred = "./predict_Bi_" + file + ".csv"
    output_name_mse = "./mse_Bi_" + file + ".csv"
    output_name_var = "./variables_Bi_" + file + ".csv"
    data_save = data.copy()

    mse_write = []
    variables_write = {}
    variables_num = len(data.columns)

    es = tf.keras.callbacks.EarlyStopping(
        monitor="loss", mode="min", verbose=0, patience=40, min_delta=0.0001
    )

    groups = data.groupby(by=["Nr."])
    for usingRange in range(start_nei, end_nei):
        for num, group in enumerate(groups):
            variables = [i for i in range(3, variables_num)]
            points = np.array(group[1], dtype=float)
            points[:, 1] = 0
            if len(points) < usingRange:
                continue

            used_variables = []
            for j, var in enumerate(variables):
                moran_test = moranI(points[:, :2].copy(), points[:, var])
                if abs(moran_test["ZI_N"]["value"]) > 1.96:
                    a, b = pearsonr(points[:, 2], points[:, var])
                    if abs(a) >= 0.6 and b <= 0.05:
                        used_variables.append(var)

            X_train = []
            X_test = []
            y_train = []

            ind = math.floor(usingRange / 2)
            for i in range(len(points) - usingRange + 1):
                points_ = points[i : i + usingRange, [0, 1, 2] + used_variables].copy()
                ty = points_[ind, 2]

                points_ = np.delete(points_, ind, axis=0)

                X_train.append(points_)
                X_test.append(points_)
                y_train.append(ty)

            X_train = np.array(X_train)
            X_test = np.array(X_test)
            y_train = np.array(y_train)

            mse = []
            preds = []
            for _ in range(1):
                model = BiLSTM(usingRange - 1, X_train.shape[-1])
                model.fit(X_train, y_train, epochs=200, verbose=0, callbacks=[es])
                pred = model.predict(X_test).flatten()
                del model
                gc.collect()
                pred = np.array(pred)
                mse.append(np.average(np.power(y_train - pred, 2)))
                preds.append(pred)
            mse = np.average(mse)
            preds = np.average(np.array(preds), axis=0)

            data_save.loc[
                data_save["Nr."] == group[0], "Bi_" + str(usingRange)
            ] = np.r_[[np.NaN] * ind, preds, [np.NaN] * (usingRange - ind - 1)]

            print(usingRange, "Bi", group[0], len(group[1]), mse)
            mse_write.append([usingRange, group[0], "Bi", str(mse)])
            if num % 100 == 0:
                data_save.to_csv(output_name_pred, index=False)
                pd.DataFrame(np.array(mse_write)).to_csv(output_name_mse, index=False)
                with open(output_name_var, "w") as f:
                    for key in variables_write.keys():
                        f.write("%s,%s\n" % (key, variables_write[key]))

    data_save.to_csv(output_name_pred, index=False)
    pd.DataFrame(np.array(mse_write)).to_csv(output_name_mse, index=False)
    with open(output_name_var, "w") as f:
        for key in variables_write.keys():
            f.write("%s,%s\n" % (key, variables_write[key]))


def mse_handle(file_name):
    data = pd.read_csv("./mse_" + file_name + ".csv")
    data.dropna(axis=0, inplace=True)
    data = data[(1e-5 < data["3"]) & (data["3"] < 1)]
    data.reset_index(drop=True, inplace=True)
    data.columns = [i for i in range(len(data.columns.values))]

    mse = []
    name = []
    for group in data.groupby(by=[0]):
        if name == []:
            for i in range(2, len(data.columns.values) - 1, 2):
                name.append(group[1][i][0])
        temp = []
        for i in range(3, len(data.columns.values), 2):
            temp.append(np.average(group[1][i]))
        mse.append(temp)
    mse = np.r_[np.array(name).reshape((1, -1)), mse]
    pd.DataFrame(mse, index=["Vars"] + [str(i) for i in range(11, 12)]).to_csv(
        "mse_" + file_name + "_average.csv", header=False
    )


pd.set_option("max_columns", 20)

statistic_tracks()
pre_study()
kriging_method("pre", 11, 12)
variables_check("pre")
cokriging_method("pre", 11, 12)
annkriging_method("pre", 11, 12)
bilstm_method("pre", 11, 12)

for file_name in ["UK_pre", "Co_pre", "AK_pre", "Bi_pre"]:
    mse_handle(file_name)
