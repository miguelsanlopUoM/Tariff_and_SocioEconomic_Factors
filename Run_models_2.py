from Decentralised_model import *
from Centralised_model import *
from Solution_reading import *
from Prosumer_optimization import *
import pandas as pd
import numpy as np
import time

## This code execute the Centralised Planning Model (CPM) and the Decentralised Investment Model (DIM) - consideering
# the Gauss-Seidel algorithm.

start= time.time()
iterations=[]
factor=0.1
n_days=3
alpha=0.7
Sto_title=""# "" if storage is possible, "_NO_STORAGE"
OPT_TE=[1,2,3]
OPT_ALPHA=[0.7,0,0.9]
OPT_CMG=[ '_Term','_Renew']
Time_centralized=[]
Time_decentralized=[]
GAP_conver=[]
times_conver=[]
Tariff_DF=pd.DataFrame()
for Cmg_case in OPT_CMG:
    Ex='Data/Parameters_JSON_37_v7' + Cmg_case + '.xlsx'
    for option_energy_tariff in OPT_TE:
        if option_energy_tariff==1:
            TE="FLAT"
        elif option_energy_tariff==2:
            TE="UKE7"
        elif option_energy_tariff==3:
            TE="Cmg"
        for alpha in OPT_ALPHA:
            if alpha==0.9:
                Volumetric_share = "VOL10_PEAK90"
            elif alpha==0:
                Volumetric_share = "VOL100_PEAK0"
            elif alpha == 0.7:
                Volumetric_share = "VOL30_PEAK70"
            As_array=np.linspace(100000,200000,11)
            Convergence=[]
            for a in As_array:
                Convergence.append('Solved')
            total_costs_centralised=[]
            total_costs_decentralised=[]
            Ps_total_centralised=[]
            Sto_total_centralised=[]
            Sto_total_decentralised=[]
            Ps_total_decentralised=[]
            Sl_total_centralised=[]
            Sl_total_decentralised=[]
            GAP_FINAL=[]
            caso=0
            Tariff=[]
            Finan_cen=[]
            Finan_dec=[]
            for As in As_array:
                print("As = "+ str(As))
                time_1=time.time()
                # Centralised Planing Model (CPM)
                model=centralised_ex(Ex,As,factor,n_days)
                time_2=time.time()
                Time_centralized.append(time_2-time_1)
                # Initialisation of distribution tariff according to the results of CPM.
                Tau_d,t_out=New_capacity_tariff_2(model,n_days,alpha)
                Profile= extract_profile(model, Ex) #Extracting the operation of the BESS
                taus=[]
                times=[]
                taus.append(Tau_d[0,t_out[0]])
                times.append(t_out[0])
                tau_final=(Tau_d[0,t_out[0]])
                tau_inicial=0
                total_costs_centralised.append(model.Total_cost.value)
                Ps_total_centralised.append(sum(model.Ps[i].value for i in model.num_consumers))
                Sl_total_centralised.append(sum(model.Sl[l].value for l in model.num_lines))
                Sto_total_centralised.append(sum(model.Sto[i].value for i in model.num_consumers))
                financial_out_1= Financial_results_DF(model,Ex,As,factor,option_energy_tariff, n_days, alpha,Tau_d)
                with pd.ExcelWriter(Ex, mode='a', if_sheet_exists="replace") as writer:
                    pd.DataFrame(financial_out_1).to_excel(writer, sheet_name='Financial'+'_c', index=False)

                financial_out_1['As']=As
                Finan_cen.append(pd.DataFrame(financial_out_1))
                i = 1
                GAP=1000

                time_1=time.time()
                conver_time=[]
                gap_conver=[]
                while (abs(GAP)>0.01 and i<10):  # abs(tau_inicial-tau_final)>0.1
                    tau_inicial=tau_final
                    print('ITERACION '+str(i))
                    Tau_d, t_out = New_capacity_tariff_2(model, n_days, alpha)
                    #Note that in this execution, the battery operation (Profile) and distirbution tariff (Tau_d) are fixed
                    model=decentralised_ex(Ex,Tau_d,As,factor, Profile, option_energy_tariff,n_days)
                    print('STO_decentralised_model = ' + str(sum(model.Sto[i].value for i in model.num_consumers)))
                    print('Solar_decentralised_model = ' + str(sum(model.Ps[i].value for i in model.num_consumers)))
                    #Adjusting tariff
                    Tau_d_2,t_out_2=New_capacity_tariff_2(model,n_days,alpha)
                    print("Calculating new profile...........")
                    #New BESS operation usage
                    Profile=Prosumer_opt(Ex,As, Tau_d_2,option_energy_tariff,n_days)
                    times.append(t_out_2[0])
                    taus.append(Tau_d_2[0,t_out_2[0]])
                    tau_final=Tau_d_2[0,t_out_2[0]]
                    # Calculating the gap of the convergence
                    GAP=(tau_inicial-tau_final)/tau_inicial
                    time_pri=time.time()
                    conver_time.append(time_pri-time_1)
                    gap_conver.append(GAP)
                    print("GAP = "+str(abs(GAP*100))+ " %")
                    print(TE + "  " + Volumetric_share+' As = '+str(As))
                    i=i+1
                    if i==10:
                        Convergence[caso]='Not_converge'


                #Calculation to save the results of this iteration
                time_2=time.time()
                Time_decentralized.append(time_2-time_1)
                GAP_conver.append(gap_conver)
                times_conver.append(conver_time)
                caso=caso+1
                Tau_e_DF= energy_tariff_2(Ex, option_energy_tariff, n_days)
                Tariff_DF['Ene_'+TE+ "_" + Volumetric_share]=pd.DataFrame(Tau_e_DF[0,:])
                Tariff_DF['Dist_' + TE+ "_" + Volumetric_share] = pd.DataFrame(Tau_d_2[0,:])
                Sol_read(model, Ex, '_d')
                Tariff.append(tau_final)
                GAP_FINAL.append((tau_inicial-tau_final)/tau_inicial)
                Sol_read(model, Ex, '_d')
                financial_out = Financial_results_DF(model, Ex, As, factor, option_energy_tariff,n_days,alpha,Tau_d)
                with pd.ExcelWriter(Ex, mode='a', if_sheet_exists="replace") as writer:
                    pd.DataFrame(financial_out).to_excel(writer, sheet_name='Financial'+'_d', index=False)
                total_costs_decentralised.append(model.Total_cost.value)
                Ps_total_decentralised.append(sum(model.Ps[i].value for i in model.num_consumers))
                Sl_total_decentralised.append(sum(model.Sl[l].value for l in model.num_lines))
                financial_out['As'] = As
                Finan_dec.append(pd.DataFrame(financial_out))

                Sto_total_decentralised.append(sum(model.Sto[i].value for i in model.num_consumers))
                iterations.append(i)




            end= time.time()

            Data_final=pd.DataFrame()
            Data_final['As']=np.array(As_array)
            Data_final['total_costs_centralised']=np.array(total_costs_centralised)
            Data_final['total_costs_decentralised']=np.array(total_costs_decentralised)
            Data_final['Ps_total_centralised']=np.array(Ps_total_centralised)
            Data_final['Ps_total_decentralised']=np.array(Ps_total_decentralised)
            Data_final['Sto_total_centralised']=np.array(Sto_total_centralised)
            Data_final['Sto_total_decentralised']=np.array(Sto_total_decentralised)
            Data_final['Sl_total_centralised']=np.array(Sl_total_centralised)
            Data_final['Sl_total_decentralised']=np.array(Sl_total_decentralised)
            pd.concat(Finan_cen).to_csv('output/FinCen_'+ Sto_title+TE +'_'+Volumetric_share+Cmg_case+'_P90_1602_Renew.csv')
            pd.concat(Finan_dec).to_csv('output/FinDec_'+ Sto_title+TE +'_'+Volumetric_share+Cmg_case+'_P90_1602_Renew.csv')
            Data_final['Gap_final']=np.array(GAP_FINAL)
            Data_final['Status']=Convergence
            Data_final.to_csv('output/Resultados_'+ Sto_title+TE +'_'+Volumetric_share+Cmg_case+'_P90_1602_Renew.csv')
            Total_time= end-start
            IT=np.array(iterations)
    Tariff_DF.to_csv('Output/Tariff_1007.csv')#Location where it is saved