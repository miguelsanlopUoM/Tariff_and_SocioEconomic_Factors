import numpy as np
import pyomo as pyo
import pandas as pd

from Auxiliar_functions_2 import *
from Auxiliar_functions import *
from Solution_reading import *
from Initialise_parameters import *


## This model contains the Prosumer Investment Model (PIM).
# For each prosumer i, the investment in solar PV and storage is optimised. This way, the number of optimisations is
# the same that the number of prosumers.
# The battery operation (Profile) is extracted. In the case of no battery installation, a 0 vector is considered.

def Prosumer_opt(Ex, As_in,Tau_d,option_energy_tariff,n_days):

    As, At, Asto, Te, Td, CV, eta, T, k, D, D_R, Out, In, Nodo, Cmg, Al, X, R, Sol_ava, min_Ps, min_Pt, min_Sto, min_T, Bat_pro_1, Bat_pro_2, Bat_pro_3 = Initialise_37_feeder(Ex)

    n_consumers = len(At)
    Sol_ava = Sol_ava * (0.4/ 0.4)
    As = np.zeros(n_consumers)
    for i in range(n_consumers):
        As[i] = As_in
    n_time = len(Cmg)
    n_alpha_prosumers=2



    model = pyo.ConcreteModel()
    initialise_consumers_variables(model, n_consumers, n_time)
    Storage_model_pro_free(model, eta,T, n_alpha_prosumers, n_time)
    Thermal_model_2(model, n_alpha_prosumers)
    Solar_pv_model(model, Sol_ava, n_alpha_prosumers)
    Consumer_balance_2(model, D, D_R)

    Tau_R=np.zeros([n_consumers, n_time])
    for i in model.num_consumers:
        for t in model.num_time:
            if t<7:
                Tau_R[i,t]=0
            else:
                Tau_R[i, t] = 0
    Tau_e = energy_tariff_2(Ex, option_energy_tariff, n_days)

    Planning_contraints_2_pro(model, min_Ps, min_Sto, min_Pt, min_T, As, At, Asto, k)
    prosumer_objective_function(model, As, At, Asto, CV, Tau_e, Tau_d, Tau_R)

    opt = pyo.SolverFactory('gurobi', tee=True)
    results = opt.solve(model)
    Profiles=extract_profile(model,Ex)
    print('STO_prosumer_model = '+str(sum(model.Sto[i].value for i in model.num_consumers)))


    return Profiles




