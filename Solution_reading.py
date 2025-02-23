import numpy as np
import pandas as pd
import pyomo.environ as pyo
from Initialise_parameters import *

## Functions to read the solutions of the Pyomo model.

def Sol_read(model,Excel_file,type_model):
    Consumers_inv=dict()
    Consumers_inv['Ps_sol']=np.array([pyo.value(model.Ps[i]) for i in model.num_consumers])
    Consumers_inv['Pt_sol'] = np.array([pyo.value(model.Pt[i]) for i in model.num_consumers])
    Consumers_inv['Sto_sol'] = np.array([pyo.value(model.Sto[i]) for i in model.num_consumers])
    Consumers_inv=pd.DataFrame(Consumers_inv)

    # summary of solution
    Operation_summary = dict()
    Operation_summary['Gsol_sol'] = np.array(
        [sum(pyo.value(model.Gsol_A[i, t]) for i in model.num_consumers) for t in model.num_time])
    Operation_summary['Gt_sol'] = np.array(
        [sum(pyo.value(model.Gt_A[i, t]) for i in model.num_consumers) for t in model.num_time])
    Operation_summary['Gchar_sol'] = np.array(
        [sum(pyo.value(model.Gchar_A[i, t]) for i in model.num_consumers) for t in model.num_time])
    Operation_summary['Gdis_sol'] = np.array(
        [sum(pyo.value(model.Gdis_A[i, t]) for i in model.num_consumers) for t in model.num_time])
    Operation_summary['Gsysin_sol'] = np.array([pyo.value(model.Gsysin_A[t]) for t in model.num_time])
    Operation_summary['Gsysout_sol'] = np.array([pyo.value(model.Gsysout_A[t]) for t in model.num_time])
    Operation_summary = pd.DataFrame(Operation_summary)

    Grid_investment = dict()
    Grid_investment['T_max'] = np.array([pyo.value(model.Sl[l]) for l in model.num_lines])
    Grid_investment = pd.DataFrame(Grid_investment)
    Lines_operation = np.array([[pyo.value(model.Pl[l, t]) for l in model.num_lines] for t in model.num_time])
    Lines_operation_pd = pd.DataFrame(Lines_operation, columns=None)
    Lines_operation_Q = np.array([[pyo.value(model.Ql[l, t]) for l in model.num_lines] for t in model.num_time])
    Lines_operation_Q_pd = pd.DataFrame(Lines_operation_Q, columns=None)
    Current = np.array([[pyo.value(model.il[l, t]) for l in model.num_lines] for t in model.num_time])
    Current_pd = pd.DataFrame(Current, columns=None)
    Voltage = np.array([[pyo.value(model.V[b, t]) for b in model.num_bus] for t in model.num_time])
    Voltage_pd = pd.DataFrame(Voltage, columns=None)

    # Excel writing
    with pd.ExcelWriter(Excel_file, mode='a', if_sheet_exists="replace") as writer:
        Consumers_inv.to_excel(writer, sheet_name='Consumers_inv' + type_model, index=False)
        Operation_summary.to_excel(writer, sheet_name='Operation_Sum' + type_model, index=False)
        Grid_investment.to_excel(writer, sheet_name='Grid_inv' + type_model, index=False)
        Lines_operation_pd.to_excel(writer, sheet_name='Line_operation_P' + type_model)
        Lines_operation_Q_pd.to_excel(writer, sheet_name='Line_operation_Q' + type_model)
        Current_pd.to_excel(writer, sheet_name='Current_' + type_model)
        Voltage_pd.to_excel(writer, sheet_name='Voltage_' + type_model)



def Financial_results(model, Excel_file, type_model,factor, option_energy_tariff,n_days,alpha,T_d):
    As, At, Asto, Te, Td, CV, eta, T, k, D, D_R, Out, In, Nodo, Cmg, Al, X, R, Sol_ava, min_Ps, min_Pt, min_Sto, min_T, Bat_pro_1, Bat_pro_2, Bat_pro_3 = Initialise_37_feeder(Excel_file)
    Te = energy_tariff_2(Excel_file, option_energy_tariff, n_days)
    Financial_out=dict()
    Financial_out['Ps_inv [$]']=np.array([pyo.value(model.Ps[i])* As[i] for i in model.num_consumers])
    Financial_out['Pt_inv [$]']=np.array([pyo.value(model.Pt[i])* At[i] for i in model.num_consumers])
    Financial_out['Sto_inv [$]']=np.array([pyo.value(model.Sto[i])* Asto[i] for i in model.num_consumers])
    Financial_out['Thermal_operating [$]'] = np.array([sum(pyo.value(model.Gt_A[i,t]) * (CV[i]) for t in model.num_time)*365 for i in model.num_consumers])
    Financial_out['Grid_buys_energy [$]'] = np.array([sum(pyo.value(model.Gin_A[i,t]) * (Te[i,t]*0.5) for t in model.num_time)*365 for i in model.num_consumers])
    Financial_out['Grid_buys_distribution [$'] = np.array([sum(pyo.value(model.Gin_A[i,t]) * (T_d[i,t]) for t in model.num_time)*365 for i in model.num_consumers])
    Financial_out['Grid_sells_energy [$]'] = np.array([sum(pyo.value(model.Gout_A[i,t]) * Te[i,t]*0.5*factor for t in model.num_time)*365 for i in model.num_consumers])
    Financial_out['Total [$]'] = (Financial_out['Ps_inv [$]']
                                  + Financial_out['Pt_inv [$]']
                                  + Financial_out['Sto_inv [$]']
                                  + Financial_out['Thermal_operating [$]']
                                  + Financial_out['Grid_buys_energy [$]']
                                  + Financial_out['Grid_buys_distribution [$']
                                  - Financial_out['Grid_sells_energy [$]']
                                  )



    Financial_out_df = pd.DataFrame(Financial_out)
    # Excel writing
    with pd.ExcelWriter(Excel_file, mode='a', if_sheet_exists="replace") as writer:
        Financial_out_df.to_excel(writer, sheet_name='Financial'+type_model, index=False)


def Financial_results_DF(model, Excel_file,As_in,factor,option_energy_tariff,n_days,alpha,T_d):
    As, At, Asto, Te, Td, CV, eta, T, k, D, D_R, Out, In, Nodo, Cmg, Al, X, R, Sol_ava, min_Ps, min_Pt, min_Sto, min_T, Bat_pro_1, Bat_pro_2, Bat_pro_3 = Initialise_37_feeder(Excel_file)
    n_consumers = len(At)
    Al = Al * 1.2
    Sol_ava = Sol_ava * (0.4 / 0.4)
    Te = np.zeros([len(model.num_consumers), len(model.num_time)])
    As = np.zeros(n_consumers)
    Te = energy_tariff_2(Excel_file, option_energy_tariff,n_days)

    for i in range(n_consumers):
        As[i] = As_in
    Financial_out=dict()
    Financial_out['Ps_inv [$]']=np.array([pyo.value(model.Ps[i])* As[i] for i in model.num_consumers])
    Financial_out['Pt_inv [$]']=np.array([pyo.value(model.Pt[i])* At[i] for i in model.num_consumers])
    Financial_out['Sto_inv [$]']=np.array([pyo.value(model.Sto[i])* Asto[i] for i in model.num_consumers])
    Financial_out['Thermal_operating [$]'] = np.array([sum(pyo.value(model.Gt_A[i,t]) * (CV[i]) for t in model.num_time) for i in model.num_consumers])
    Financial_out['Grid_buys_energy [$]'] = np.array([sum(pyo.value(model.Gin_A[i,t]) * (Te[i,t]) for t in model.num_time) for i in model.num_consumers])
    Financial_out['Grid_buys_distribution [$]'] = np.array([sum((pyo.value(model.Gin_A[i,t])-pyo.value(model.Gout_A[i,t])) * (T_d[i,t]) for t in model.num_time) for i in model.num_consumers])
    Financial_out['Grid_sells_energy [$]'] = np.array([sum(pyo.value(model.Gout_A[i,t] * Te[i,t]*factor) for t in model.num_time) for i in model.num_consumers])
    Financial_out['Total [$]'] = (Financial_out['Ps_inv [$]']
                                  + Financial_out['Pt_inv [$]']
                                  + Financial_out['Sto_inv [$]']
                                  + Financial_out['Thermal_operating [$]']
                                  + Financial_out['Grid_buys_energy [$]']
                                  + Financial_out['Grid_buys_distribution [$]']
                                  - Financial_out['Grid_sells_energy [$]']
                                  )
    zona=[4,1,4,4,2,4,2,2,2,4,1,1,4,3,3,4,4,3,3,4,4,3,3,1,4]
    Financial_out['Zone']=np.array(zona)
    return Financial_out


def systemic_balance(model, Excel_file, type_model):
    As, At, Asto, Te, Td, CV, eta, T, k, D, D_R, Out, In, Nodo, Cmg, Al, X, R, Sol_ava, min_Ps, min_Pt, min_Sto, min_T, Bat_pro_1, Bat_pro_2, Bat_pro_3 = Initialise_37_feeder(Excel_file)
    Systemic_financial=dict()
    Systemic_financial['Dist_investment']=np.array([pyo.value(model.Grid_reinforcement)])
    Systemic_financial['DER_investment']=np.array([pyo.value(model.Solar_investment)+ sum(At[i] * pyo.value(model.Pt[i]) for i in model.num_consumers)])
    Systemic_financial['Sto_investment']=np.array([pyo.value(model.Sto_investment)])
    Systemic_financial['Energy_buys']=np.array([pyo.value(model.Energy_buys)])
    Systemic_financial['Energy_sells']=np.array([pyo.value(model.Energy_sells)])
    Systemic_financial['Operating_costs']=np.array([sum(CV[i]*pyo.value(model.Gt_A[i,t]) for i in model.num_consumers for t in model.num_time)*365])
    Systemic_financial['Total_costs']=np.array([pyo.value(model.Total_cost)])

    Systemic_financial=pd.DataFrame(Systemic_financial)
    # Excel writing
    with pd.ExcelWriter(Excel_file, mode='a', if_sheet_exists="replace") as writer:
        Systemic_financial.to_excel(writer, sheet_name='Sys_financial' + type_model, index=False)

def Volumetric_tariff(model):
    Grid_investment=pyo.value(model.Grid_reinforcement)
    Energy_buy_i=np.zeros(len(model.num_consumers))
    for i in model.num_consumers:
        Energy_buy_i[i]=sum(pyo.value(model.Gin_A[i,t]) for t in model.num_time)
    Energy_buy=sum(Energy_buy_i[i] for i in model.num_consumers)
    Td=np.zeros([len(model.num_consumers), len(model.num_time)])
    for i in model.num_consumers:
        for t in model.num_time:
            Td[i,t]=Grid_investment/(Energy_buy*365)
    t_out=0
    return Td,[t_out]


def Peak_tariff(model):
    Grid_investment = pyo.value(model.Grid_reinforcement)
    Energy_matrix=np.zeros([len(model.num_consumers), len(model.num_time)])
    Net_energy=np.zeros(len(model.num_time))
    for i in model.num_consumers:
        for t in model.num_time:
            Energy_matrix[i,t] = pyo.value(model.Gin_A[i,t])-pyo.value(model.Gout_A[i,t])
    for t in model.num_time:
        Net_energy[t]= pyo.value(model.Gsysin_A[t])-pyo.value(model.Gsysout_A[t])

    max_demand=np.max(np.abs(Net_energy))
    t_max_demand=np.argmax(np.abs(Net_energy))
    Td=np.zeros([len(model.num_consumers), len(model.num_time)])
    for i in model.num_consumers:
        Td[i,t_max_demand]=Grid_investment/(Net_energy[t_max_demand]*365)
    return Td,[t_max_demand]

def Multi_inspired_tariff(model):
    k=4
    Grid_investment = pyo.value(model.Grid_reinforcement)
    Energy_matrix = np.zeros([len(model.num_consumers), len(model.num_time)])
    Net_energy = np.zeros(len(model.num_time))
    for i in model.num_consumers:
        for t in model.num_time:
            Energy_matrix[i, t] = pyo.value(model.Gin_A[i, t]) - pyo.value(model.Gout_A[i, t])
    for t in model.num_time:
        Net_energy[t] = pyo.value(model.Gsysin_A[t]) - pyo.value(model.Gsysout_A[t])

    time_ordered = np.argsort(np.abs(Net_energy))
    Td = np.zeros([len(model.num_consumers), len(model.num_time)])
    total_energy_considered= np.sum(np.abs(Net_energy[time_ordered[-k:]]))
    for i in model.num_consumers:
        for t in time_ordered[-k:]:
            Td[i,t]=Grid_investment/(total_energy_considered*365)

    return Td, time_ordered[-k:]


def Mix_inspired_tariff(model):
    k=4
    alpha=0.4
    Grid_investment = pyo.value(model.Grid_reinforcement)
    Energy_matrix = np.zeros([len(model.num_consumers), len(model.num_time)])
    Net_energy = np.zeros(len(model.num_time))
    for i in model.num_consumers:
        for t in model.num_time:
            Energy_matrix[i, t] = pyo.value(model.Gin_A[i, t]) - pyo.value(model.Gout_A[i, t])
    for t in model.num_time:
        Net_energy[t] = pyo.value(model.Gsysin_A[t]) - pyo.value(model.Gsysout_A[t])

    time_ordered = np.argsort(np.abs(Net_energy))
    Td = np.zeros([len(model.num_consumers), len(model.num_time)])
    total_energy_considered= np.sum(np.abs(Net_energy[time_ordered[-k:]]))
    Energy_buy = sum(Net_energy[i] for i in model.num_time)
    for i in model.num_consumers:
        for t in time_ordered[-k:]:
            Td[i,t]=(Grid_investment*alpha/(total_energy_considered*365))
        for t in model.num_time:
            Td[i,t] = Td[i,t] + (Grid_investment*(1-alpha)/(Energy_buy*365))

    return Td, time_ordered[-k:]

def New_capacity_tariff(model):
    alpha=0.7

    Grid_investment = pyo.value(model.Grid_reinforcement)
    Energy_matrix = np.zeros([len(model.num_consumers), len(model.num_time)])
    Net_energy = np.zeros(len(model.num_time))
    for i in model.num_consumers:
        for t in model.num_time:
            Energy_matrix[i, t] = pyo.value(model.Gin_A[i, t]) - pyo.value(model.Gout_A[i, t])
    for t in model.num_time:
        Net_energy[t] = pyo.value(model.Gsysin_A[t]) - pyo.value(model.Gsysout_A[t])

    p= np.percentile(Net_energy,90) # np.percentile(Net_energy,80)np.mean(Net_energy) + 0.0 * np.std(Net_energy)
    Energy_peak = 0
    Energy_buy = sum(Net_energy[i] for i in model.num_time)
    times=[]
    for t in model.num_time:
        if Net_energy[t]>= p:
            Energy_peak += Net_energy[t]
            times.append(t)
    Td = np.zeros([len(model.num_consumers), len(model.num_time)])
    for i in model.num_consumers:
        for t in model.num_time:
            if Net_energy[t] >= p:
                Td[i,t] += (Grid_investment*alpha/(Energy_peak))
            Td[i,t] += (Grid_investment*(1-alpha)/(Energy_buy))

    return Td, times

def extract_profile (model,Excel_file):
    As, At, Asto, Te, Td, CV, eta, T, k, D, D_R, Out, In, Nodo, Cmg, Al, X, R, Sol_ava, min_Ps, min_Pt, min_Sto, min_T, Bat_pro_1, Bat_pro_2, Bat_pro_3 = Initialise_37_feeder(Excel_file)
    Profiles = np.zeros([len(model.num_consumers),len(model.num_time)])
    for t in model.num_time:
        for i in model.num_consumers:
            if model.Sto[i].value==0:
                Profiles[i,t]=Bat_pro_2[i,t]
            else:
                Profiles[i,t]=(model.Gdis_A[i,t].value-(model.Gchar_A[i,t].value ))/model.Sto[i].value

    return Profiles

def energy_tariff(Excel_file,option,n_days):
    As, At, Asto, Te, Td, CV, eta, T, k, D, D_R, Out, In, Nodo, Cmg, Al, X, R, Sol_ava, min_Ps, min_Pt, min_Sto, min_T, Bat_pro_1, Bat_pro_2, Bat_pro_3 = Initialise_37_feeder(
        Excel_file)
    num_consumers=len(As)
    num_time=len(Cmg)
    range_1 = range((n_days - 1) * 24)
    range_2 = range((n_days - 1) * 24, n_days * 24)
    #option == 1 flat, option ==2 UKE7, option == 3 Cmg
    Tau_e=np.zeros([num_consumers, num_time])
    if option==1:
        average=sum(Cmg[t]*sum(D[i,t] for i in range(num_consumers)) for t in range(num_time))/sum(D[i,t] for i in range(num_consumers) for t in range(num_time))
        for i in range(num_time):
            for t in range_1:
                Tau_e[i,t]= average * (365/n_days)

    elif option==2:
        base_1=range(7)
        base_2=range(7,24)
        average_1=sum(Cmg[t]*sum(D[i,t] for i in range(num_consumers)) for t in base_1)/sum(D[i,t] for i in range(num_consumers) for t in base_1)
        average_2 = sum(Cmg[t] * sum(D[i, t] for i in range(num_consumers)) for t in base_2) / sum(D[i, t] for i in range(num_consumers) for t in base_2)
        for i in range(num_consumers):
            for t in base_1:
                Tau_e[i,t]= average_1 * (365/n_days)
            for t in base_2:
                Tau_e[i, t] = average_2 * (365/n_days)

    elif option==3:
        for i in range(num_consumers):
            for t in range(num_time):
                Tau_e[i,t] = Cmg[t] * (365/n_days)

    return Tau_e


def energy_tariff_2(Excel_file,option,n_days):
    As, At, Asto, Te, Td, CV, eta, T, k, D, D_R, Out, In, Nodo, Cmg, Al, X, R, Sol_ava, min_Ps, min_Pt, min_Sto, min_T, Bat_pro_1, Bat_pro_2, Bat_pro_3 = Initialise_37_feeder(
        Excel_file)
    num_consumers=len(As)
    num_time=len(Cmg)
    range_1 = range((n_days - 1) * 24)
    range_2 = range((n_days - 1) * 24, n_days * 24)
    Tau_e=np.zeros([num_consumers, num_time])
    if option==1:
        average=((sum(Cmg[t]*sum(D[i,t] for i in range(num_consumers)) for t in range_1)/sum(D[i,t] for i in range(num_consumers) for t in range_1))*364
                + sum(Cmg[t] * sum(D[i, t] for i in range(num_consumers)) for t in range_2) / sum(D[i, t] for i in range(num_consumers) for t in range_2))/365
        for i in range(num_consumers):
            for t in range_1:
                Tau_e[i,t]= average * (364/(n_days-1))

            for t in range_2:
                Tau_e[i,t]= average

    elif option==2:
        base_1=[]
        base_2=[]
        for t in range(num_time):
            if t%24<7:
                base_1.append(t)
            else:
                base_2.append(t)
        average_1=sum(Cmg[t]*sum(D[i,t] for i in range(num_consumers)) for t in base_1)/sum(D[i,t] for i in range(num_consumers) for t in base_1)
        average_2 = sum(Cmg[t] * sum(D[i, t] for i in range(num_consumers)) for t in base_2) / sum(D[i, t] for i in range(num_consumers) for t in base_2)
        for i in range(num_consumers):
            for t in range(num_time):
                if t< 24* (n_days-1):
                    if t in base_1:
                        Tau_e[i,t]= average_1 * (364/(n_days-1))
                    else:
                        Tau_e[i, t] = average_2 * (364/(n_days-1))
                else:
                    if t in base_1:
                        Tau_e[i, t] = average_1
                    else:
                        Tau_e[i, t] = average_2


    elif option==3:
        for i in range(num_consumers):
            for t in range_1:
                Tau_e[i,t] = Cmg[t] * (364/(n_days-1))
            for t in range_2:
                Tau_e[i,t] = Cmg[t]


    return Tau_e


def New_capacity_tariff_2(model,n_days,alpha):
    Range_h=np.linspace(0,(24*n_days)-1,24*n_days)
    Range_capacity_day = Range_h[24*(n_days-1):]
    Grid_investment = pyo.value(model.Grid_reinforcement)
    Energy_matrix = np.zeros([len(model.num_consumers), len(model.num_time)])
    Net_energy = np.zeros(len(model.num_time))
    for i in model.num_consumers:
        for t in model.num_time:
            Energy_matrix[i, t] = pyo.value(model.Gin_A[i, t]) - pyo.value(model.Gout_A[i, t])
    for t in model.num_time:
        Net_energy[t] = pyo.value(model.Gsysin_A[t]) - pyo.value(model.Gsysout_A[t])

    p = np.percentile([Net_energy[int(t)] for t in Range_capacity_day], 90)  # np.percentile(Net_energy,80)np.mean(Net_energy) + 0.0 * np.std(Net_energy)
    Energy_peak = 0
    Energy_buy = sum(Net_energy[i]*(364/(n_days-1)) for i in range(24*(n_days-1)))+sum(Net_energy[i] for i in range(24*(n_days-1), 24*n_days))
    times = []
    for t in Range_capacity_day:
        if Net_energy[int(t)] >= p:
            Energy_peak += Net_energy[int(t)]
            times.append(int(t))
    Td = np.zeros([len(model.num_consumers), len(model.num_time)])
    for i in model.num_consumers:
        for t in model.num_time:
            if np.isin(times,t).any() :
                Td[i, t] += (Grid_investment * alpha / (Energy_peak))
            if t< 24*(n_days-1):
                Td[i, t] += (Grid_investment * (1 - alpha) / (Energy_buy)) * (364/(n_days-1))
            else:
                Td[i, t] += (Grid_investment * (1 - alpha) / (Energy_buy))

    return Td, times
















