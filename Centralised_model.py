from Initialise_parameters import *
from Auxiliar_functions import *
from Auxiliar_functions_2 import *
from Solution_reading import *
## This code is the centralised expansion model, considering the distritbution power flow (AC)

def centralised_ex(Ex, As_in,factor,n_days):
    # Reading data
    As, At, Asto, Te, Td, CV, eta, T, k, D, D_R, Out, In, Nodo, Cmg, Al, X, R, Sol_ava, min_Ps, min_Pt, min_Sto, min_T, Bat_pro_1, Bat_pro_2, Bat_pro_3 = Initialise_37_feeder(Ex)
    n_consumers = len(At)
    As=np.zeros(n_consumers)
    Al=Al*1.2
    Sol_ava=Sol_ava*(0.4/0.4)
    for i in range(n_consumers):
        As[i]=As_in


    n_time = len(Cmg)
    n_bus = len(Nodo[0, :])
    n_line = len(X)
    n_alpha=10
    n_alpha_prosumers=2
    Cmg_R = Cmg*0.05
    CV_R = CV*0.05


    ## Network initialisation
    file_ini='Data/Ini.xlsx'
    Data_S = pd.read_excel(file_ini, sheet_name='S_ini')
    Data_V = pd.read_excel(file_ini, sheet_name='V_ini')

    voltage = Data_V['V_ini']
    p_from = Data_S['P_ini']
    q_from = Data_S['Q_ini']

    N_time=n_time
    vol=np.zeros([len(voltage),N_time])
    P_ini = np.zeros([len(p_from) ,N_time])
    Q_ini = np.zeros([len(q_from) ,N_time])
    for t in range(n_time):
        for i in range(len(vol)):
            vol[i,t]=Data_V['V_ini'][i]**2
        for i in range(len(P_ini)):
            P_ini[i,t]= Data_S['P_ini'][i]
            Q_ini[i,t]= Data_S['Q_ini'][i]



    # initialising model
    model = pyo.ConcreteModel()
    initialise_consumers_variables(model, n_consumers, n_time)
    initialise_networks_variables(model, n_line, n_bus)
    Storage_model_pro_free(model, eta, T, n_alpha, n_time)
    Thermal_model_2(model, n_alpha_prosumers)
    Solar_pv_model(model, Sol_ava, n_alpha_prosumers)
    Consumer_balance_2(model, D, D_R)
    Linear_AC_power_flow(model, Out, In, Nodo, X, R, n_alpha)
    Objective_function_3(model, As, At, Asto, Cmg, Cmg_R, Al, CV, CV_R,n_days)
    Planning_contraints_2(model, min_Ps, min_Sto, min_Pt, min_T, As, At, Asto, k)
    opt= pyo.SolverFactory('gurobi',tee=True)
    results = opt.solve(model)

    cost_ini=1e15
    while (abs(cost_ini-pyo.value(model.obj))>=0.001): #Level of accuraci of the distflow
        cost_ini=pyo.value(model.obj)
        for t in range(n_time):
            for i in range(len(vol)):
                vol[i,t]=model.V[i,t].value
            for i in range(len(P_ini)):
                P_ini[i,t]= model.Pl[i,t].value
                Q_ini[i,t]= model.Ql[i,t].value

        # This hyperplane apporximate the quadratic equations of Dist flow
        hyperplane_voltage_current(model, vol, P_ini, Q_ini, Out)
        opt= pyo.SolverFactory('gurobi',tee=True)
        results = opt.solve(model)
        print("Total cost [$] = " + str(pyo.value(model.obj)))
    print("Total cost [$] CENTRALIZED= " + str(pyo.value(model.obj)))
    Sol_read(model,Ex,'_c')
    Systemic_financial= systemic_balance(model,Ex,'_c')
    return model







