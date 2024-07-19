# import packages
import os
import math
from opfunu.cec_based.cec2020 import *
import numpy as np

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
FitCurrentBest = 0
TotalPopSize = DuckPopSize + FishPopSize
LB = [-100] * DimSize  # the maximum value of the variable range
UB = [100] * DimSize  # the minimum value of the variable range
TrialRuns = 30  # the number of independent runs
MaxFEs = 1000 * DimSize  # the maximum number of fitness evaluations
curIter = 0  # the current number of generations
MaxIter = math.ceil(MaxFEs / TotalPopSize)

FuncNum = 0
SuiteName = "CEC2020"

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

def Initialization(func):
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
        FitDuck[i] = func.evaluate(DuckPop[i])
    for i in range(FishPopSize):
        for j in range(DimSize):
            FishPop[i][j] = LB[j] + (UB[j] - LB[j]) * np.random.rand()
            # calculate the fitness of the i-th individual
        FitFish[i] = func.evaluate(FishPop[i])
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


def HCCIA(func):
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
            FitOff = func.evaluate(Off)
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
            FitOff = func.evaluate(Off)
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


def RunHCCIA(func):
    global curIter, MaxFEs, TrialRuns, DimSize
    All_Trial_Best = []
    for i in range(TrialRuns):
        np.random.seed(1996 + 12 * i)
        Best_list = []
        curIter = 0
        Initialization(func)
        curIter = 1
        Best_list.append(FitCurrentBest)
        while curIter < MaxIter:
            HCCIA(func)
            curIter += 1
            Best_list.append(FitCurrentBest)
        All_Trial_Best.append(Best_list)
    np.savetxt("./HCCIA_Data/CEC2020/F" + str(FuncNum) + "_" + str(DimSize) + "D.csv", All_Trial_Best,
               delimiter=",")


def main(dim):
    global FuncNum, DimSize, MaxFEs, MaxIter,SuiteName, LB, UB
    DimSize = dim
    MaxFEs = dim * 1000
    MaxIter = int(MaxFEs / TotalPopSize)
    LB = [-100] * dim
    UB = [100] * dim

    CEC2020 = [F12020(DimSize), F22020(DimSize), F32020(DimSize), F42020(DimSize), F52020(DimSize),
               F62020(DimSize), F72020(DimSize), F82020(DimSize), F92020(DimSize), F102020(DimSize)]

    FuncNum = 0
    for i in range(len(CEC2020)):
        FuncNum = i + 1
        RunHCCIA(CEC2020[i])


if __name__ == "__main__":
    if os.path.exists('./HCCIA_Data/CEC2020') == False:
        os.makedirs('./HCCIA_Data/CEC2020')
    Dims = [30, 50]
    for Dim in Dims:
        main(Dim)


