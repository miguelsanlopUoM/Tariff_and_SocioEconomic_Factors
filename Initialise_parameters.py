import numpy as np
import pandas as pd
import pyomo.environ as pyo
import openpyxl


def Initialise(Excel_file):
    Data_consumers = pd.read_excel(Excel_file,sheet_name='Consumers')
    Data_demand = pd.read_excel(Excel_file,sheet_name='Demand',header=None)
    Data_Out = pd.read_excel(Excel_file, sheet_name='Out', header=None)
    Data_In = pd.read_excel(Excel_file, sheet_name='In', header=None)
    Data_Nodo = pd.read_excel(Excel_file, sheet_name='Nodo', header= None)
    Data_Cmg = pd.read_excel(Excel_file, sheet_name='Cmg', header=None)
    Data_Lines = pd.read_excel(Excel_file, sheet_name='Lines')
    Data_solava= pd.read_excel(Excel_file, sheet_name='Solar_ava', header=None)

    As = np.array(Data_consumers.values[:, 0], dtype=float)
    At = np.array(Data_consumers.values[:, 1], dtype=float)
    Asto = np.array(Data_consumers.values[:, 2], dtype=float)
    Te = np.array(Data_consumers.values[:, 3], dtype=float)
    Td = np.array(Data_consumers.values[:, 4], dtype=float)
    Cv = np.array(Data_consumers.values[:, 5], dtype=float)
    eta = np.array(Data_consumers.values[:, 6], dtype=float)
    T = np.array(Data_consumers.values[:, 7], dtype=float)
    k = np.array(Data_consumers.values[:, 8], dtype=float)
    min_Ps = np.array(Data_consumers.values[:, 9], dtype=float)
    min_Pt = np.array(Data_consumers.values[:, 10], dtype=float)
    min_Sto = np.array(Data_consumers.values[:, 11], dtype=float)

    D = np.array(Data_demand.values[:, :], dtype=float)
    Sol_ava= np.array(Data_solava.values[:, :], dtype=float)

    Out = np.array(Data_Out.values[:, :], dtype=float)

    In = np.array(Data_In.values[:, :], dtype=float)

    Nodo = np.array(Data_Nodo.values[:, :], dtype=float)

    Cmg = np.array(Data_Cmg.values[:], dtype=float)

    Al = np.array(Data_Lines.values[:, 0], dtype=float)
    X = np.array(Data_Lines.values[:, 2], dtype=float)
    R = np.array(Data_Lines.values[:, 1], dtype=float)
    min_T = np.array(Data_Lines.values[:, 3], dtype=float)

    return As, At, Asto, Te, Td, Cv, eta, T, k, D, Out, In, Nodo, Cmg, Al, X, R, Sol_ava, min_Ps, min_Pt, min_Sto, min_T


def Initialise_37_feeder(Excel_file):
    Data_consumers = pd.read_excel(Excel_file,sheet_name='Consumers')
    Data_demand = pd.read_excel(Excel_file,sheet_name='Demand',header=None)
    Data_demand_Q = pd.read_excel(Excel_file, sheet_name='Demand_Q', header=None)
    Data_Out = pd.read_excel(Excel_file, sheet_name='Out', header=None)
    Data_In = pd.read_excel(Excel_file, sheet_name='In', header=None)
    Data_Nodo = pd.read_excel(Excel_file, sheet_name='Nodo', header= None)
    Data_Cmg = pd.read_excel(Excel_file, sheet_name='Cmg', header=None)
    Data_Lines = pd.read_excel(Excel_file, sheet_name='Lines')
    Data_solava= pd.read_excel(Excel_file, sheet_name='Solar_ava', header=None)
    Data_battery_1 = pd.read_excel(Excel_file, sheet_name='Bat_1', header=None)  # OJOOOOO
    Data_battery_2 = pd.read_excel(Excel_file, sheet_name='Bat_2', header=None)
    Data_battery_3 = pd.read_excel(Excel_file, sheet_name='Bat_3', header=None)

    As = np.array(Data_consumers.values[:, 0], dtype=float)
    At = np.array(Data_consumers.values[:, 1], dtype=float)
    Asto = np.array(Data_consumers.values[:, 2], dtype=float)
    Te = np.array(Data_consumers.values[:, 3], dtype=float)
    Td = np.array(Data_consumers.values[:, 4], dtype=float)
    Cv = np.array(Data_consumers.values[:, 5], dtype=float)
    eta = np.array(Data_consumers.values[:, 6], dtype=float)
    T = np.array(Data_consumers.values[:, 7], dtype=float)
    k = np.array(Data_consumers.values[:, 8], dtype=float)
    min_Ps = np.array(Data_consumers.values[:, 9], dtype=float)
    min_Pt = np.array(Data_consumers.values[:, 10], dtype=float)
    min_Sto = np.array(Data_consumers.values[:, 11], dtype=float)

    D = np.array(Data_demand.values[:, :], dtype=float)
    D_R = np.array(Data_demand_Q.values[:, :], dtype=float)
    Sol_ava= np.array(Data_solava.values[:, :], dtype=float)

    Bat_pro_1 = np.array(Data_battery_1.values[:, :], dtype=float)
    Bat_pro_2 = np.array(Data_battery_2.values[:, :], dtype=float)
    Bat_pro_3 = np.array(Data_battery_3.values[:, :], dtype=float)

    Out = np.array(Data_Out.values[:, :], dtype=float)

    In = np.array(Data_In.values[:, :], dtype=float)

    Nodo = np.array(Data_Nodo.values[:, :], dtype=float)

    Cmg = np.array(Data_Cmg.values[:], dtype=float)


    Al = np.array(Data_Lines.values[:, 0], dtype=float)
    X = np.array(Data_Lines.values[:, 2], dtype=float)
    R = np.array(Data_Lines.values[:, 1], dtype=float)
    min_T = np.array(Data_Lines.values[:, 3], dtype=float)

    return As, At, Asto, Te, Td, Cv, eta, T, k, D, D_R, Out, In, Nodo, Cmg, Al, X, R, Sol_ava, min_Ps, min_Pt, min_Sto, min_T, Bat_pro_1, Bat_pro_2, Bat_pro_3

