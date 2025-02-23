from Initialise_parameters import *
from Auxiliar_functions import *
from Auxiliar_functions_2 import *
from Solution_reading import *
import time
# This code is the centralised expansion model, considering a distribution power flow.
# This model considers a predefined value of Tau_d

def decentralised_ex(Ex, Tau_d, As_in,factor,profile,option_energy_tariff, n_days):
    As, At, Asto, Te, Td, CV, eta, T, k, D, D_R, Out, In, Nodo, Cmg, Al, X, R, Sol_ava, min_Ps, min_Pt, min_Sto, min_T, Bat_pro_1, Bat_pro_2, Bat_pro_3 = Initialise_37_feeder(Ex)
    n_consumers = len(At)
    Al=Al*1.2
    Sol_ava = Sol_ava * (0.4/ 0.4)
    As = np.zeros(n_consumers)
    for i in range(n_consumers):
        As[i] = As_in
    n_time = len(Cmg)
    n_bus = len(Nodo[0, :])
    n_line = len(X)
    n_alpha=10
    n_alpha_prosumers=2
    Cmg_R=Cmg*0.05
    CV_R=CV*0.05
    # Fixed operation of the battery
    Bat_pro_1 = profile
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
    Storage_model_2(model, Bat_pro_1) #Notar que en este caso se considera un
    Thermal_model_2(model, n_alpha_prosumers)
    Solar_pv_model(model, Sol_ava, n_alpha_prosumers)
    Consumer_balance_2(model, D, D_R)
    Linear_AC_power_flow(model, Out, In, Nodo, X, R, n_alpha)
    Objective_function_3(model, As, At, Asto, Cmg, Cmg_R, Al, CV, CV_R,n_days)
    Planning_contraints_2(model, min_Ps, min_Sto, min_Pt, min_T, As, At, Asto, k)

    ##Initialising network
    start= time.time()
    opt= pyo.SolverFactory('gurobi',tee=True)
    end= time.time()
    results = opt.solve(model)
    cost_ini=1e15
    while (abs(cost_ini-pyo.value(model.obj))>=0.001):
        start = time.time()
        cost_ini=pyo.value(model.obj)
        for t in range(n_time):
            for i in range(len(vol)):
                vol[i,t]=model.V[i,t].value
            for i in range(len(P_ini)):
                P_ini[i,t]= model.Pl[i,t].value
                Q_ini[i,t]= model.Ql[i,t].value

        hyperplane_voltage_current(model, vol, P_ini, Q_ini, Out)
        opt= pyo.SolverFactory('gurobi',tee=True)
        results = opt.solve(model)
        end = time.time()
    print("Total cost [$] DECEN= " + str(pyo.value(model.obj)))
    # The hyperplane_voltage_current are approximation of the quadratic equations of distribution power flow.

    #For this case of study, we explicity consider a reactive power tariff equals to 0.
    Tau_R=np.zeros([n_consumers, n_time])
    for i in model.num_consumers:
        for t in model.num_time:
            if t<7:
                Tau_R[i,t]=0
            else:
                Tau_R[i, t] = 0
    # Energy tariff initialisation, considering directly the marginal cost in the primary substation (Cmg),
    # two blocks, and the flat tariff.
    Tau_e = energy_tariff_2(Ex,option_energy_tariff,n_days)

    Sl_results=np.zeros(n_line)
    for l in model.num_lines:
        Sl_results[l]=model.Sl[l].value

    #Initialising the prosumers equations

    Dual_variables_2(model, n_alpha_prosumers)
    Integer_variables_2(model, n_alpha_prosumers)
    First_order_condition_2(model, As, At, Asto, Tau_e, Tau_d, Tau_R, CV, CV_R,eta, T, n_alpha_prosumers, Sol_ava,factor, Bat_pro_1)
    Complementary_Slackness_2(model, As, At, Asto,Tau_e, Tau_d, Tau_R, k, eta, T, n_alpha_prosumers, Sol_ava, D, D_R)
    print('Gauss Seidel 1')
    start=time.time()
    opt= pyo.SolverFactory('gurobi',tee=True)
    opt.options['MIPGap'] = 0.0001
    opt.options['timelimit'] = 600
    opt.options['Heuristics'] = 0.2
    opt.options['Cuts'] = 2
    opt.options['Presolve'] = 0
    results=opt.solve(model,tee=True)
    end=time.time()
    print('Gap de = '+str((model.Total_cost.value-results.problem.lower_bound)/model.Total_cost.value))
    print("Tiempo de ejecucion = " + str(end - start))
    print("Total cost [$] = " + str(pyo.value(model.obj)))

    Systemic_financial= systemic_balance(model,Ex,'_d')
    return model