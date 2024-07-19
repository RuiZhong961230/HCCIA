import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import csv
from intelelm import MhaElmRegressor
import os
from permetrics import RegressionMetric
import warnings
from os import path


warnings.filterwarnings("ignore")

DimSize = 10

DuckPopSize = 60
DuckPop = np.zeros((DuckPopSize, DimSize))
FitDuck = np.zeros(DuckPopSize)
BestDuck = np.zeros(DimSize)
FitBestDuck = 0

FishPopSize = 40
FishPop = np.zeros((FishPopSize, DimSize))
FitFish = np.zeros(FishPopSize)
BestFish = np.zeros(DimSize)
FitBestFish = 0

Prey = np.zeros(DimSize)

CurrentBest = np.zeros(DimSize)
FitCurrentBest = float("inf")

TotalPopSize = DuckPopSize + FishPopSize

LB = [-10] * DimSize  # the maximum value of the variable range
UB = [10] * DimSize  # the minimum value of the variable range
TrialRuns = 30  # the number of independent runs
MaxFEs = 50000

curIter = 0  # the current number of generations
MaxIter = 500

FuncNum = 0
SuiteName = "CEC2020"

X_train = None
y_train = None
model = MhaElmRegressor(hidden_size=10, act_name="elu", obj_name="RMSE")

def open_csv(path):
    data = []
    with open(path) as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            d = []
            for s in row:
                d.append(float(s))
            data.append(d)
    return np.array(data)


def fit_func(indi):
    global X_train, y_train, model
    try:
        Acc = model.fitness_function(indi)
    except Exception:
        Acc = 0
    return Acc


def sigmoid(a):
    return 1 / (1 + np.exp(-a))

P1, P2, P3, P4 = np.random.rand(4)


def Tent(x):
    if x < 0.7:
        return x / 0.7
    else:
        return abs(10 * (1 - x) / 3)


def Sawtooth(x):
    if x == 0.5:
        return np.random.rand()
    else:
        return 2 * x % 1

def Chebyshev(x):
    global curIter
    return abs(np.cos(curIter/np.cos(x)))


def Iterative(x):
    return abs(np.sin(np.random.rand() * np.pi / x))


def Logistic(x):
    if x in [0.25, 0.5, 0.75]:
        return np.random.rand()
    else:
        return abs(4 * x * (1 - x))


Chaos = [Tent, Sawtooth, Chebyshev, Iterative, Logistic]

def Initialization():
    global DuckPop, FitDuck, FishPop, FitFish, BestDuck, BestFish, Prey, CurrentBest, FitCurrentBest, FitBestFish, FitBestDuck, CurrentBest, FitCurrentBest, P1, P2, P3, P4
    DuckPop = np.zeros((DuckPopSize, DimSize))
    FitDuck = np.zeros(DuckPopSize)
    BestDuck = np.zeros(DimSize)
    FishPop = np.zeros((FishPopSize, DimSize))
    FitFish = np.zeros(FishPopSize)
    BestFish = np.zeros(DimSize)
    Prey = np.zeros(DimSize)
    CurrentBest = np.zeros(DimSize)
    FitCurrentBest = 0
    P1, P2, P3, P4 = np.random.rand(4)
    # randomly generate individuals
    for i in range(DuckPopSize):
        for j in range(DimSize):
            DuckPop[i][j] = LB[j] + (UB[j] - LB[j]) * np.random.rand()
            # calculate the fitness of the i-th individual
        FitDuck[i] = -fit_func(DuckPop[i])
    for i in range(FishPopSize):
        for j in range(DimSize):
            FishPop[i][j] = LB[j] + (UB[j] - LB[j]) * np.random.rand()
            # calculate the fitness of the i-th individual
        FitFish[i] = -fit_func(FishPop[i])
    BestDuck = DuckPop[np.argmin(FitDuck)]
    FitBestDuck = np.min(FitDuck)

    BestFish = FishPop[np.argmin(FitFish)]
    FitBestFish = np.min(FitFish)
    if FitBestFish > FitBestDuck:
        CurrentBest = np.copy(BestDuck)
        FitCurrentBest = FitBestDuck
    else:
        CurrentBest = np.copy(BestFish)
        FitCurrentBest = FitBestFish


def HCCIA():
    global DuckPop, FitDuck, BestDuck, FitBestDuck, FishPop, FitFish, BestFish, FitBestFish, Prey, CurrentBest, FitCurrentBest, P1, P2, P3, P4
    # find the best duck and fish
    BestDuck = DuckPop[np.argmin(FitDuck)]
    FitBestDuck = np.min(FitDuck)

    BestFish = FishPop[np.argmin(FitFish)]
    FitBestFish = np.min(FitFish)

    Off = np.zeros(DimSize)

    if FitBestFish > FitBestDuck:
        CurrentBest = np.copy(BestDuck)
        FitCurrentBest = FitBestDuck
    else:
        CurrentBest = np.copy(BestFish)
        FitCurrentBest = FitBestFish

    if np.random.random() < 0.5:
        Prey = np.copy(BestFish)
    else:
        Prey = np.copy(BestDuck)

    diversity = np.mean(np.std(DuckPop, axis=0))
    best_duck = np.argmin(FitDuck)
    best_fish = np.argmin(FitFish)
    A = (1 - curIter / MaxIter)
    for i in range(DuckPopSize):
        if i == best_duck:
            Off = np.copy(BestDuck)
            for j in range(DimSize):
                if np.random.rand() < 0.5:
                    Off[j] = Off[j] + A * P1
                else:
                    Off[j] = Off[j] - A * P1
                chao_idx = np.random.randint(len(Chaos))
                P1 = Chaos[chao_idx](P1)
        else:
            idx = np.random.randint(0, FishPopSize)
            RandFitFish = FitFish[idx]
            RandFish = np.copy(FishPop[idx])
            if FitDuck[i] < RandFitFish:
                Off = DuckPop[i] + P2 * (Prey - DuckPop[i]) * np.sin(2 * np.pi * P2) * A
                chao_idx = np.random.randint(len(Chaos))
                P2 = Chaos[chao_idx](P2)
            else:
                for j in range(DimSize):
                    if np.random.random() < 0.5:
                        Off[j] = DuckPop[i][j] + np.random.normal() * A * diversity
                    else:
                        Off[j] = RandFish[j] + np.random.normal() * A * diversity

            Off = np.clip(Off, LB, UB)
            FitOff = -fit_func(Off)
            if FitOff < FitDuck[i]:
                DuckPop[i] = np.copy(Off)
                FitDuck[i] = FitOff
                if FitOff < FitBestDuck:
                    BestDuck = np.copy(Off)
                    FitBestDuck = FitOff

    for i in range(FishPopSize):
        if i == best_fish:
            Off = np.copy(BestFish)
            for j in range(DimSize):
                if np.random.rand() < 0.5:
                    Off[j] = Off[j] + A * P3
                else:
                    Off[j] = Off[j] - A * P3
                chao_idx = np.random.randint(len(Chaos))
                P3 = Chaos[chao_idx](P3)
        else:
            Off = BestFish - A * (2 * P4 - 1) * (BestDuck - FishPop[i]) - (2 * P4 - 1) * A * (DuckPop[np.random.randint(0, DuckPopSize)] - FishPop[i])
            chao_idx = np.random.randint(len(Chaos))
            P4 = Chaos[chao_idx](P4)
            Off = np.clip(Off, LB, UB)
            FitOff = -fit_func(Off)
            if FitOff < FitFish[i]:
                FishPop[i] = np.copy(Off)
                FitFish[i] = FitOff
                if FitOff < FitBestFish:
                    BestFish = np.copy(Off)
                    FitBestFish = FitOff

    if FitBestFish > FitBestDuck:
        CurrentBest = np.copy(BestDuck)
        FitCurrentBest = FitBestDuck
    else:
        CurrentBest = np.copy(BestFish)
        FitCurrentBest = FitBestFish


def RunHCCIA(file_path):
    global curIter, TrialRuns, DimSize, X_train, y_train, model, DimSize, LB, UB, CurrentBest, FitCurrentBest
    dataset = open_csv(file_path)
    X = dataset[:, 0:8]
    y = dataset[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=100)

    scaler_X = MinMaxScaler()
    scaler_X.fit(X_train)
    X_train = scaler_X.transform(X_train)
    X_test = scaler_X.transform(X_test)

    MAE_train_List = []
    MSE_train_List = []
    RMSE_train_List = []
    R2_train_List = []
    MAPE_train_List = []

    MAE_test_List = []
    MSE_test_List = []
    RMSE_test_List = []
    R2_test_List = []
    MAPE_test_List = []

    for i in range(TrialRuns):

        np.random.seed(1996 + 12 * i)
        model = MhaElmRegressor(hidden_size=10, act_name="elu", obj_name="RMSE")
        model.network, model.obj_scaler = model.create_network(X_train, y_train)
        model.X_temp, model.y_temp = X_train, y_train
        DimSize = len(X_train[0]) * 10 + 10
        LB = [-10] * DimSize
        UB = [10] * DimSize
        curIter = 0
        Initialization()
        while curIter < MaxIter:
            HCCIA()
            curIter += 1
        model.network.update_weights_from_solution(CurrentBest, model.X_temp, model.y_temp)
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        evaluator = RegressionMetric(y_train, y_train_pred)
        Train_results = evaluator.get_metrics_by_list_names(
            list_metric_names=["MAE", "MSE", "RMSE", "R2", "MAPE"],
            list_paras=[{"multi_output": "raw_values"}, ] * 5
        )
        y_test_pred = model.predict(X_test)
        evaluator = RegressionMetric(y_test, y_test_pred)
        Test_results = evaluator.get_metrics_by_list_names(
            list_metric_names=["MAE", "MSE", "RMSE", "R2", "MAPE"],
            list_paras=[{"multi_output": "raw_values"}, ] * 5
        )
        MAE_train_List.append(Train_results["MAE"])
        MSE_train_List.append(Train_results["MSE"])
        RMSE_train_List.append(Train_results["RMSE"])
        R2_train_List.append(Train_results["R2"])
        MAPE_train_List.append(Train_results["MAPE"])

        MAE_test_List.append(Test_results["MAE"])
        MSE_test_List.append(Test_results["MSE"])
        RMSE_test_List.append(Test_results["RMSE"])
        R2_test_List.append(Test_results["R2"])
        MAPE_test_List.append(Test_results["MAPE"])

    np.savetxt("./HCCIA_Data/Train/MAE.csv", MAE_train_List, delimiter=",")
    np.savetxt("./HCCIA_Data/Train/MSE.csv", MSE_train_List, delimiter=",")
    np.savetxt("./HCCIA_Data/Train/RMSE.csv", RMSE_train_List, delimiter=",")
    np.savetxt("./HCCIA_Data/Train/R2.csv", R2_train_List, delimiter=",")
    np.savetxt("./HCCIA_Data/Train/MAPE.csv", MAPE_train_List, delimiter=",")

    np.savetxt("./HCCIA_Data/Test/MAE.csv", MAE_test_List, delimiter=",")
    np.savetxt("./HCCIA_Data/Test/MSE.csv", MSE_test_List, delimiter=",")
    np.savetxt("./HCCIA_Data/Test/RMSE.csv", RMSE_test_List, delimiter=",")
    np.savetxt("./HCCIA_Data/Test/R2.csv", R2_test_List, delimiter=",")
    np.savetxt("./HCCIA_Data/Test/MAPE.csv", MAPE_test_List, delimiter=",")


if __name__ == "__main__":
    if os.path.exists('./HCCIA_Data/Train') == False:
        os.makedirs('./HCCIA_Data/Train')
    if os.path.exists('./HCCIA_Data/Test') == False:
        os.makedirs('./HCCIA_Data/Test')
    this_path = path.dirname(path.abspath(__file__))
    path_file = this_path + "\\data.csv"
    RunHCCIA(path_file)

