import pyomo.environ as pyo
import numpy as np
import pandapower

## Set of constrants for Centralised and Decentralised investment models
def initialise_consumers_variables(model, n_consumers, n_time):
    model.num_consumers=range(n_consumers)
    model.num_time=range(n_time)
    # variables initialisations GENERAL
    model.Ps = pyo.Var(model.num_consumers, domain=pyo.NonNegativeReals)
    model.Pt = pyo.Var(model.num_consumers, domain=pyo.NonNegativeReals)
    model.Sto = pyo.Var(model.num_consumers, domain=pyo.NonNegativeReals)
    model.Gout_A = pyo.Var(model.num_consumers, model.num_time, domain=pyo.NonNegativeReals)
    model.Gin_A = pyo.Var(model.num_consumers, model.num_time, domain=pyo.NonNegativeReals)
    model.Gt_A = pyo.Var(model.num_consumers, model.num_time, domain=pyo.NonNegativeReals)
    model.Gsol_A = pyo.Var(model.num_consumers, model.num_time, domain=pyo.NonNegativeReals)
    model.Soc = pyo.Var(model.num_consumers, model.num_time, domain=pyo.NonNegativeReals)
    model.Gchar_A = pyo.Var(model.num_consumers, model.num_time, domain=pyo.NonNegativeReals)
    model.Gdis_A = pyo.Var(model.num_consumers, model.num_time, domain=pyo.NonNegativeReals)
    model.Gout_R = pyo.Var(model.num_consumers, model.num_time, domain=pyo.NonNegativeReals)
    model.Gin_R = pyo.Var(model.num_consumers, model.num_time, domain=pyo.NonNegativeReals)
    model.Gt_R = pyo.Var(model.num_consumers, model.num_time, domain=pyo.Reals)
    model.Gsol_R = pyo.Var(model.num_consumers, model.num_time, domain=pyo.Reals)
    model.Gsto_R = pyo.Var(model.num_consumers, model.num_time, domain=pyo.Reals)

def initialise_networks_variables(model,n_line, n_bus):
    model.num_lines = range(n_line)
    model.num_bus = range(n_bus)

    model.Gsysout_A = pyo.Var(model.num_time, domain=pyo.NonNegativeReals)
    model.Gsysin_A = pyo.Var(model.num_time, domain=pyo.NonNegativeReals)
    model.Gsysout_R = pyo.Var(model.num_time, domain=pyo.NonNegativeReals)
    model.Gsysin_R = pyo.Var(model.num_time, domain=pyo.NonNegativeReals)
    model.Sl = pyo.Var(model.num_lines, domain=pyo.NonNegativeReals)
    model.V = pyo.Var(model.num_bus, model.num_time, domain=pyo.NonNegativeReals)
    model.il = pyo.Var(model.num_lines, model.num_time, domain=pyo.NonNegativeReals)
    model.Pl = pyo.Var(model.num_lines, model.num_time, domain=pyo.Reals)
    model.Ql = pyo.Var(model.num_lines, model.num_time, domain=pyo.Reals)
    model.Dn_A = pyo.Var(model.num_bus, model.num_time, domain=pyo.Reals)
    model.Dn_R = pyo.Var(model.num_bus, model.num_time, domain=pyo.Reals)
    model.Q_compensation_pos = pyo.Var(model.num_bus, model.num_time, domain=pyo.NonNegativeReals)
    model.Q_compensation_neg = pyo.Var(model.num_bus, model.num_time, domain=pyo.NonNegativeReals)
    model.Solar_investment = pyo.Var(domain=pyo.NonNegativeReals)
    model.Thermal_investment = pyo.Var(domain=pyo.NonNegativeReals)
    model.Sto_investment = pyo.Var(domain=pyo.NonNegativeReals)
    model.Energy_sells = pyo.Var( domain=pyo.NonNegativeReals)
    model.Reactive_buys = pyo.Var(domain=pyo.NonNegativeReals)
    model.Reactive_sells = pyo.Var( domain=pyo.NonNegativeReals)
    model.Energy_buys = pyo.Var( domain=pyo.NonNegativeReals)
    model.thermal_costs_A = pyo.Var( domain=pyo.NonNegativeReals)
    model.thermal_costs_R = pyo.Var(domain=pyo.NonNegativeReals)
    model.Grid_reinforcement = pyo.Var(domain=pyo.NonNegativeReals)

def Storage_model(model, eta, T, n_alpha, n_time):
    model.G_char_up = pyo.ConstraintList()
    model.G_char_down = pyo.ConstraintList()
    model.G_dis_up = pyo.ConstraintList()
    model.G_dis_down = pyo.ConstraintList()
    model.SoC_max = pyo.ConstraintList()
    model.SoC_inventory = pyo.ConstraintList()
    model.SoC_neutrality = pyo.ConstraintList()
    model.Sto_MAX= pyo.ConstraintList()
    STOMAX=10

    for i in model.num_consumers:
        model.SoC_neutrality.add(
            model.Soc[i,0] == model.Soc[i,n_time-1] + eta[i]*model.Gchar_A[i,n_time-1] - model.Gdis_A[i,n_time-1]
        )
        model.Sto_MAX.add(
            model.Sto[i]<=STOMAX
        )
        for t in range(n_time-1):
            model.SoC_inventory.add(
                model.Soc[i,t+1]==
                model.Soc[i,t]
                + eta[i]*model.Gchar_A[i,t]
                - model.Gdis_A[i,t]
            )
        for t in model.num_time:
            model.SoC_max.add(
                model.Soc[i, t] <= T[i] * model.Sto[i]
            )
    alpha_array=np.linspace(0,1,n_alpha)

    for a in alpha_array:
        for i in model.num_consumers:
            for t in model.num_time:
                model.G_char_up.add(
                    -1*(((-a*model.Gchar_A[i,t])+model.Sto[i]))
                    <=
                    model.Gsto_R[i,t]*(np.sqrt(1-a*a))
                )

                model.G_char_up.add(
                    model.Gsto_R[i, t]*(np.sqrt(1 - a * a))
                    <=
                    (((-a * model.Gchar_A[i, t]) + model.Sto[i]))
                )

                model.G_dis_up.add(
                    -1 * (((-a * model.Gdis_A[i, t]) + model.Sto[i]))
                    <=
                    model.Gsto_R[i, t]*(np.sqrt(1 - a * a))
                )

                model.G_dis_up.add(
                    model.Gsto_R[i, t]* (np.sqrt(1 - a * a))
                    <=
                    (((-a * model.Gdis_A[i, t]) + model.Sto[i]))
                )

def Solar_pv_model(model, Sol_ava, n_alpha):
    model.G_solar_max_ava=pyo.ConstraintList()
    model.G_solar_up=pyo.ConstraintList()
    model.G_solar_down = pyo.ConstraintList()
    model.G_NO_reactives=pyo.ConstraintList()
    model.Ps_MAX=pyo.ConstraintList()
    alpha_array=np.linspace(0,1,n_alpha)
    PS_MAX=10

    for i in model.num_consumers:
        for t in model.num_time:
            model.G_solar_max_ava.add(
                model.Gsol_A[i,t] <= model.Ps[i]*Sol_ava[i,t]
            )
            model.G_NO_reactives.add(
                model.Gsol_R[i,t]==0
            )
            model.Ps_MAX.add(
                model.Ps[i]<=PS_MAX
            )


def Thermal_model(model, n_alpha):
    model.G_ther_up = pyo.ConstraintList()
    model.G_ther_down = pyo.ConstraintList()
    alpha_array = np.linspace(0, 1, n_alpha)
    model.Pt_MAX=pyo.ConstraintList()
    Pt_MAX=0 #Imposing that thermal prosumers do not install thermal generators

    for i in model.num_consumers:
        model.Pt_MAX.add(
            model.Pt[i]<=Pt_MAX
        )
        for t in model.num_time:
            for a in alpha_array:
                model.G_ther_up.add(
                    -1 * (((-a * model.Gt_A[i, t]) + model.Pt[i]))
                    <=
                    model.Gt_R[i, t]* (np.sqrt(1 - a * a))
                )

                model.G_ther_down.add(
                    model.Gt_R[i, t]* (np.sqrt(1 - a * a))
                    <=
                    (((-a * model.Gt_A[i, t]) + model.Pt[i]))
                )

def Consumer_balance(model, D, D_R):
    model.energy_balance_A = pyo.ConstraintList()
    model.energy_balance_R = pyo.ConstraintList()

    for t in model.num_time:
        for i in model.num_consumers:
            model.energy_balance_A.add(
                D[i,t]
                + model.Gout_A[i,t]
                + model.Gchar_A[i,t]
                ==
                model.Gin_A[i,t] +
                model.Gdis_A[i,t] +
                model.Gsol_A[i,t] +
                model.Gt_A[i,t]
            )

            model.energy_balance_R.add(
                (D_R[i, t])
                + model.Gout_R[i, t]
                ==
                model.Gin_R[i, t] +
                model.Gsto_R[i, t] +
                model.Gsol_R[i, t] +
                model.Gt_R[i, t]
            )

def Objective_function(model, As, At, Asto, Cmg, Cmg_R, Al, CV, CV_R):
    model.Solar_def = pyo.ConstraintList()
    model.Thermal_def = pyo.ConstraintList()
    model.Sto_def = pyo.ConstraintList()
    model.Energy_sell_def = pyo.ConstraintList()
    model.Energy_buy_def = pyo.ConstraintList()
    model.Reactive_sell_def = pyo.ConstraintList()
    model.Reactive_buy_def = pyo.ConstraintList()
    model.Grid_reinfor_def = pyo.ConstraintList()
    model.thermal_costs_A_def = pyo.ConstraintList()
    model.thermal_costs_R_def = pyo.ConstraintList()
    model.solar_costs_R_def = pyo.ConstraintList()
    model.storage_costs_R_def = pyo.ConstraintList()
    model.Total_cost = pyo.Var(domain=pyo.Reals)
    model.solar_costs_R = pyo.Var(domain=pyo.Reals)
    model.storage_costs_R = pyo.Var(domain=pyo.Reals)
    model.Total_cost_definition = pyo.ConstraintList()
    CV_R_sol = np.zeros(len(model.num_consumers))
    CV_R_sto = np.zeros(len(model.num_consumers))
    for i in model.num_consumers:
        CV_R_sol[i]=0
        CV_R_sto[i]=0


    model.Solar_def.add(
        model.Solar_investment ==
        sum(As[i]*model.Ps[i] for i in model.num_consumers)
    )
    model.Thermal_def.add(
        model.Thermal_investment ==
        sum(At[i] * model.Pt[i] for i in model.num_consumers)
    )
    model.Sto_def.add(
        model.Sto_investment ==
        sum(Asto[i] * model.Sto[i] for i in model.num_consumers)
    )
    model.Energy_sell_def.add(
        model.Energy_sells == sum(model.Gsysout_A[t]*((Cmg[t]-0.05)*0.5) for t in model.num_time)*365
    )
    model.Energy_buy_def.add(
        model.Energy_buys == sum(model.Gsysin_A[t] * (Cmg[t]*0.5) for t in model.num_time)*365
    )
    model.Reactive_sell_def.add(
        model.Reactive_sells == sum(model.Gsysout_R[t] * ((Cmg_R[t]-0.05)*0.5) for t in model.num_time)*365
    )
    model.Reactive_buy_def.add(
        model.Reactive_buys == sum(model.Gsysin_R[t] * (Cmg_R[t]*0.5) for t in model.num_time)*365
    )
    model.Grid_reinfor_def.add(
        model.Grid_reinforcement == sum(Al[l]*model.Sl[l] for l in model.num_lines)
    )
    model.thermal_costs_A_def.add(
        model.thermal_costs_A == sum(CV[i]*model.Gt_A[i,t] for i in model.num_consumers for t in model.num_time)*365
    )
    model.thermal_costs_R_def.add(
        model.thermal_costs_R == sum(CV_R[i] * model.Gt_R[i, t] for i in model.num_consumers for t in model.num_time)*365
    )

    model.solar_costs_R_def.add(
        model.solar_costs_R == sum(CV_R_sol[i] * model.Gsol_R[i, t] for i in model.num_consumers for t in model.num_time)*365
    )

    model.storage_costs_R_def.add(
        model.storage_costs_R == sum(CV_R_sto[i] * model.Gsto_R[i, t] for i in model.num_consumers for t in model.num_time)*365
    )

    model.Total_cost_definition.add(
        model.Total_cost
        ==
        + model.Solar_investment
        + model.Thermal_investment
        + model.Sto_investment
        - model.Energy_sells
        + model.Energy_buys
        - model.Reactive_sells
        + model.Reactive_buys
        + model.Grid_reinforcement
        + model.thermal_costs_A
        + model.thermal_costs_R
        + model.solar_costs_R
        + model.storage_costs_R
    )
    model.obj = pyo.Objective(expr=(model.Total_cost), sense=pyo.minimize)

def Linear_AC_power_flow(model, Out, In, Nodo, X, R, n_alpha):
    model.nodal_voltage = pyo.ConstraintList()
    model.active_nodal_balance = pyo.ConstraintList()
    model.reactive_nodal_balance = pyo.ConstraintList()
    model.Net_demand_A = pyo.ConstraintList()
    model.Net_demand_R = pyo.ConstraintList()
    model.Reactive_up = pyo.ConstraintList()
    model.Reactive_down = pyo.ConstraintList()
    model.hyperplane = pyo.ConstraintList()
    model.V_lim_up = pyo.ConstraintList()
    model.V_lim_down = pyo.ConstraintList()
    model.il_upper = pyo.ConstraintList()
    model.Gsys_aux = pyo.ConstraintList()
    model.Compensator_11_pos = pyo.ConstraintList()
    model.Compensator_11_neg = pyo.ConstraintList()


    base=0.5

    for j in model.num_bus:
        for t in model.num_time:
            model.Net_demand_A.add(
                model.Dn_A[j,t] ==
                sum((model.Gin_A[i,t]-model.Gout_A[i,t])*Nodo[i,j] for i in model.num_consumers)/base
            )
            model.Net_demand_R.add(
                model.Dn_R[j, t] ==
                (sum((model.Gin_R[i, t] - model.Gout_R[i, t]) * Nodo[i, j] for i in model.num_consumers)+ model.Q_compensation_pos[j,t] - model.Q_compensation_neg[j,t])/base
            )##  + model.Q_compensation_pos[j,t] - model.Q_compensation_neg[j,t]
            if (j!=16):
                model.Compensator_11_pos.add(
                    model.Q_compensation_pos[j,t] == 0
                )
                model.Compensator_11_neg.add(
                    model.Q_compensation_neg[j, t] == 0
                )

            if j==0:
                model.V_lim_up.add(
                    model.V[j, t]
                    <=
                    1.03**2
                )
                model.V_lim_down.add(
                    model.V[j, t]
                    >=
                    1.03**2
                )
            else:
                model.V_lim_up.add(
                    model.V[j,t]
                    <=
                    1.05**2
                )
                model.V_lim_down.add(
                    model.V[j,t]
                    >=
                    0.95**2
                )


    for l in model.num_lines:
        for t in model.num_time:
            model.nodal_voltage.add(
                sum(model.V[j,t]*Out[j,l] for j in model.num_bus)
                - sum(model.V[k,t]*In[k,l] for k in model.num_bus)
                ==
                2*(model.Pl[l,t]*R[l] + model.Ql[l,t]*X[l])
                - (R[l]**2 + X[l]**2)*model.il[l,t]
            )
            model.il_upper.add(
                model.il[l,t]<=30
            )

    for j in model.num_bus:
        for t in model.num_time:
            if j==0:
                model.active_nodal_balance.add(
                    sum(model.Pl[l, t] * Out[j, l] for l in model.num_lines)
                    + (model.Gsysout_A[t]/base) ==
                    (model.Gsysin_A[t]/base)
                )
                model.reactive_nodal_balance.add(
                    sum(model.Ql[l, t] * Out[j, l] for l in model.num_lines)
                    + (model.Gsysout_R[t]/base) ==
                    (model.Gsysin_R[t]/base)
                )

            else:
                model.active_nodal_balance.add(
                    sum(model.Pl[l,t]*Out[j,l] for l in model.num_lines)
                    + model.Dn_A[j,t] ==
                    sum((model.Pl[l,t] - R[l]*model.il[l,t])*In[j,l] for l in model.num_lines)#ojo
                )
                model.reactive_nodal_balance.add(
                    sum(model.Ql[l, t] * Out[j, l] for l in model.num_lines)
                    + model.Dn_R[j, t] ==
                    sum((model.Ql[l, t] - X[l] * model.il[l, t]) * In[j, l] for l in model.num_lines)
                )

    alpha_array = np.linspace(0, 1, n_alpha)

    for a in alpha_array:
        for l in model.num_lines:
            for t in model.num_time:
                model.Reactive_up.add(
                    model.Ql[l,t]* np.sqrt(1-a**2)
                    <=
                    (-a*model.Pl[l,t]+(model.Sl[l]/base))
                )
                model.Reactive_down.add(
                    -1*(-a * model.Pl[l, t] + (model.Sl[l]/base))
                    <=
                    model.Ql[l,t]* np.sqrt(1 - a ** 2)
                )

def initial_conditions(file,model):
    N_time=len(model.num_time)
    A = pandapower.converter.from_mpc(file, f_hz=50, casename_mpc_file='mpc', validate_conversion=False)
    pandapower.runopp(A)
    voltage = A.res_bus['vm_pu']
    p_from = A.res_line['p_from_mw']
    q_from = A.res_line['q_from_mvar']
    vol=np.zeros([len(voltage)+1,N_time])
    P_ini = np.zeros([len(p_from) + 1,N_time])
    Q_ini = np.zeros([len(q_from) + 1,N_time])
    for t in model.num_time:
        for i in range(len(vol)):
            if i==0:
                vol[i,t]=1
            else:
                vol[i,t]=voltage[i-1]**2
        for i in range(len(P_ini)):
            if i==0:
                P_ini[i,t]=A.res_ext_grid['p_mw'][0]
                Q_ini[i,t]=A.res_ext_grid['q_mvar'][0]
            else:
                P_ini[i,t]= p_from[i-1]
                Q_ini[i,t]= q_from[i-1]
    return vol, P_ini, Q_ini

def hyperplane_voltage_current(model, vol, P_ini, Q_ini, Out):
    for t in model.num_time:
        for l in model.num_lines:
            model.hyperplane.add(
                sum(vol[j,t]*Out[j,l] for j in model.num_bus)*model.il[l,t]
                >=
                P_ini[l,t]**2
                +2*P_ini[l,t]*(model.Pl[l,t]-P_ini[l,t])
                +Q_ini[l,t]**2
                +2*Q_ini[l,t]*(model.Ql[l,t]-Q_ini[l,t])
            )

def Operational_contraints(model, Ps_val, Sto_val, Pt_val, Sl_val):
    model.Solar_fixing=pyo.ConstraintList()
    model.Storage_fixing = pyo.ConstraintList()
    model.Thermal_fixing = pyo.ConstraintList()
    model.Network_fixing = pyo.ConstraintList()

    for i in model.num_consumers:
        if (i==100):
            model.Solar_fixing.add(
                model.Ps[i] >= 0
            )
            model.Storage_fixing.add(
                model.Sto[i] >= 0
            )
            model.Thermal_fixing.add(
                model.Pt[i] >= 0
            )
        else:
            model.Solar_fixing.add(
                model.Ps[i] == Ps_val[i]
            )
            model.Storage_fixing.add(
                model.Sto[i] == Sto_val[i]
            )
            model.Thermal_fixing.add(
                model.Pt[i] == Pt_val[i]
            )

    for l in model.num_lines:
        model.Network_fixing.add(
            model.Sl[l] >= 0
        )

def Planning_contraints(model, Ps_val, Sto_val, Pt_val, Sl_val, As, At, Asto, k):
    model.Solar_fixing=pyo.ConstraintList()
    model.Storage_fixing = pyo.ConstraintList()
    model.Thermal_fixing = pyo.ConstraintList()
    model.Thermal_aux = pyo.ConstraintList()
    model.Network_fixing = pyo.ConstraintList()
    model.Budget_constraint_2 = pyo.ConstraintList()

    for i in model.num_consumers:
        model.Budget_constraint_2.add(
            As[i] * model.Ps[i]  + Asto[i] * model.Sto[i] <= k[i]
        )
        model.Solar_fixing.add(
            model.Ps[i] >= Ps_val[i]
        )
        model.Storage_fixing.add(
            model.Sto[i] >= Sto_val[i]
        )


    for l in model.num_lines:
        model.Network_fixing.add(
            model.Sl[l] >= 0# Sl_val[i]
        )

def fixing_network(model, Sl_results):
    model.fixing_network=pyo.ConstraintList()
    for l in model.num_lines:
        model.fixing_network.add(
            model.Sl[l] == Sl_results[l]
        )

def Integer_variables(model, n_alpha):
    model.u_phi_upA_solava=pyo.Var(model.num_consumers, model.num_time,domain=pyo.Binary)
    model.u_phi_up_soc=pyo.Var(model.num_consumers, model.num_time,domain=pyo.Binary)
    model.u_beta=pyo.Var(model.num_consumers,domain=pyo.Binary)
    model.u_phi_upalpha_T=pyo.Var(model.num_consumers, model.num_time, range(n_alpha), domain=pyo.Binary)
    model.u_phi_downalpha_T = pyo.Var(model.num_consumers, model.num_time, range(n_alpha), domain=pyo.Binary)
    model.u_phi_upalpha_char = pyo.Var(model.num_consumers, model.num_time, range(n_alpha), domain=pyo.Binary)
    model.u_phi_downalpha_char = pyo.Var(model.num_consumers, model.num_time, range(n_alpha), domain=pyo.Binary)
    model.u_phi_upalpha_dis = pyo.Var(model.num_consumers, model.num_time, range(n_alpha), domain=pyo.Binary)
    model.u_phi_downalpha_dis = pyo.Var(model.num_consumers, model.num_time, range(n_alpha), domain=pyo.Binary)
    model.u_sigma_sol = pyo.Var(model.num_consumers, domain=pyo.Binary)
    model.u_sigma_T=pyo.Var(model.num_consumers, domain=pyo.Binary)
    model.u_sigma_sto = pyo.Var(model.num_consumers, domain=pyo.Binary)
    model.u_sigma_A_in= pyo.Var(model.num_consumers, model.num_time, domain=pyo.Binary)
    model.u_sigma_A_out = pyo.Var(model.num_consumers, model.num_time, domain=pyo.Binary)
    model.u_sigma_R_in = pyo.Var(model.num_consumers, model.num_time, domain=pyo.Binary)
    model.u_sigma_R_out = pyo.Var(model.num_consumers, model.num_time, domain=pyo.Binary)
    model.u_phi_downA_char = pyo.Var(model.num_consumers, model.num_time, domain=pyo.Binary)
    model.u_phi_downA_dis = pyo.Var(model.num_consumers, model.num_time, domain=pyo.Binary)
    model.u_phi_downA_sol = pyo.Var(model.num_consumers, model.num_time, domain=pyo.Binary)
    model.u_phi_downA_T = pyo.Var(model.num_consumers, model.num_time, domain=pyo.Binary)
    model.u_phi_downA_soc = pyo.Var(model.num_consumers, model.num_time, domain=pyo.Binary)
    model.u_SIGMA_sol_MAX = pyo.Var(model.num_consumers, domain=pyo.Binary)
    model.u_SIGMA_t_MAX = pyo.Var(model.num_consumers, domain=pyo.Binary)
    model.u_SIGMA_sto_MAX = pyo.Var(model.num_consumers, domain=pyo.Binary)

def Dual_variables(model, n_alpha):
    model.phi_upA_solava = pyo.Var(model.num_consumers, model.num_time, domain=pyo.NonNegativeReals)
    model.gamma = pyo.Var(model.num_consumers, model.num_time, domain=pyo.Reals)
    model.phi_up_soc = pyo.Var(model.num_consumers, model.num_time, domain=pyo.NonNegativeReals)
    model.gamma_neutral = pyo.Var(model.num_consumers,  domain=pyo.Reals)
    model.beta = pyo.Var(model.num_consumers, domain=pyo.NonNegativeReals)
    model.lambda_R = pyo.Var(model.num_consumers, model.num_time, domain=pyo.Reals)
    model.lambda_A = pyo.Var(model.num_consumers, model.num_time, domain=pyo.Reals)
    model.phi_upalpha_T = pyo.Var(model.num_consumers, model.num_time, range(n_alpha), domain=pyo.NonNegativeReals)
    model.phi_downalpha_T = pyo.Var(model.num_consumers, model.num_time, range(n_alpha), domain=pyo.NonNegativeReals)
    model.phi_upalpha_char = pyo.Var(model.num_consumers, model.num_time, range(n_alpha), domain=pyo.NonNegativeReals)
    model.phi_downalpha_char = pyo.Var(model.num_consumers, model.num_time, range(n_alpha), domain=pyo.NonNegativeReals)
    model.phi_upalpha_dis = pyo.Var(model.num_consumers, model.num_time, range(n_alpha), domain=pyo.NonNegativeReals)
    model.phi_downalpha_dis = pyo.Var(model.num_consumers, model.num_time, range(n_alpha), domain=pyo.NonNegativeReals)
    model.sigma_sol = pyo.Var(model.num_consumers, domain=pyo.NonNegativeReals)
    model.sigma_T = pyo.Var(model.num_consumers, domain=pyo.NonNegativeReals)
    model.sigma_sto = pyo.Var(model.num_consumers, domain=pyo.NonNegativeReals)
    model.sigma_A_in = pyo.Var(model.num_consumers, model.num_time, domain=pyo.NonNegativeReals)
    model.sigma_A_out = pyo.Var(model.num_consumers, model.num_time, domain=pyo.NonNegativeReals)
    model.sigma_R_in = pyo.Var(model.num_consumers, model.num_time, domain=pyo.NonNegativeReals)
    model.sigma_R_out = pyo.Var(model.num_consumers, model.num_time, domain=pyo.NonNegativeReals)
    model.phi_downA_char = pyo.Var(model.num_consumers, model.num_time, domain=pyo.NonNegativeReals)
    model.phi_downA_dis = pyo.Var(model.num_consumers, model.num_time, domain=pyo.NonNegativeReals)
    model.phi_downA_sol = pyo.Var(model.num_consumers, model.num_time, domain=pyo.NonNegativeReals)
    model.phi_downA_T = pyo.Var(model.num_consumers, model.num_time, domain=pyo.NonNegativeReals)
    model.phi_downA_soc = pyo.Var(model.num_consumers, model.num_time, domain=pyo.NonNegativeReals)
    model.SIGMA_sol_MAX=pyo.Var(model.num_consumers, domain=pyo.NonNegativeReals)
    model.SIGMA_t_MAX = pyo.Var(model.num_consumers, domain=pyo.NonNegativeReals)
    model.SIGMA_sto_MAX = pyo.Var(model.num_consumers, domain=pyo.NonNegativeReals)

def First_order_condition(model, As, At, Asto, Tau_e, Tau_d, Tau_R, CV, CV_R,eta, T, n_alpha, Sol_ava, sell_factor):
    model.dLdPs = pyo.ConstraintList()
    model.dLdPt = pyo.ConstraintList()
    model.dLdSto = pyo.ConstraintList()
    model.dLdG_A_in = pyo.ConstraintList()
    model.dLdG_A_out = pyo.ConstraintList()
    model.dLdG_A_T = pyo.ConstraintList()
    model.dLdG_A_sol = pyo.ConstraintList()
    model.dLdG_A_dis = pyo.ConstraintList()
    model.dLdG_A_char = pyo.ConstraintList()
    model.dLdSoc = pyo.ConstraintList()
    model.dLdG_R_in = pyo.ConstraintList()
    model.dLdG_R_out = pyo.ConstraintList()
    model.dLdG_R_T = pyo.ConstraintList()
    model.dLdG_R_sol = pyo.ConstraintList()
    model.dLdG_R_sto = pyo.ConstraintList()
    CV_R_sol=np.zeros(len(model.num_consumers))
    CV_R_sto = np.zeros(len(model.num_consumers))
    for i in model.num_consumers:
        CV_R_sol[i]=0
        CV_R_sto[i]=0


    alpha_array = np.linspace(0, 1, n_alpha)

    for i in model.num_consumers:

        model.dLdPs.add(
            As[i]
            + As[i] * model.beta[i]
            - sum(Sol_ava[i, t] * model.phi_upA_solava[i, t] for t in model.num_time)
            - model.sigma_sol[i]
            + model.SIGMA_sol_MAX[i]
            == 0
        )

        model.dLdPt.add(
            At[i]
            + At[i]*model.beta[i]
            - sum(model.phi_upalpha_T[i,t,a]  for a in range(n_alpha) for t in model.num_time)
            - sum(model.phi_downalpha_T[i,t,a]  for a in range(n_alpha) for t in model.num_time)
            - model.sigma_T[i]
            + model.SIGMA_t_MAX[i]
            == 0
        )

        model.dLdSto.add(
            Asto[i]
            + Asto[i]*model.beta[i]
            - sum(T[i]*model.phi_up_soc[i,t] for t in model.num_time)
            - sum(model.phi_upalpha_char[i,t,a] for a in range(n_alpha) for t in model.num_time)
            - sum(model.phi_downalpha_char[i,t,a] for a in range(n_alpha) for t in model.num_time)
            - sum(model.phi_upalpha_dis[i, t, a]  for a in range(n_alpha) for t in model.num_time)
            - sum(model.phi_downalpha_dis[i, t, a] for a in range(n_alpha) for t in model.num_time)
            - model.sigma_sto[i]
            + model.SIGMA_sto_MAX[i]
            == 0
        )

        for t in model.num_time:
            model.dLdG_A_in.add(
                (Tau_e[i,t]*0.5 + Tau_d[i,t])*365
                - model.lambda_A[i,t]
                - model.sigma_A_in[i,t]
                == 0
            )

            model.dLdG_A_out.add(
                - Tau_e[i,t]*0.5*365*sell_factor
                + model.lambda_A[i,t]
                - model.sigma_A_out[i,t]
                == 0
            )

            model.dLdG_A_T.add(
                CV[i]*365*0.5
                - model.lambda_A[i,t]
                + sum((alpha_array[a]*model.phi_upalpha_T[i,t,a]) for a in range(n_alpha))
                + sum((alpha_array[a] * model.phi_downalpha_T[i, t, a]) for a in range(n_alpha))
                - model.phi_downA_T[i,t]
                == 0
            )


            model.dLdG_A_sol.add(
                model.phi_upA_solava[i, t]
                - model.lambda_A[i, t]
                - model.phi_downA_sol[i, t]
                == 0
            )

            model.dLdG_R_in.add(
                Tau_R[i,t]*0.5*365
                - model.lambda_R[i,t]
                - model.sigma_R_in[i,t]
                ==0
            )

            model.dLdG_R_out.add(
                - Tau_R[i,t]*0.5*365
                + model.lambda_R[i,t]
                - model.sigma_R_out[i,t]
                == 0
            )

            model.dLdG_R_T.add(
                CV_R[i]*365*0.5
                - model.lambda_R[i,t]
                + sum(model.phi_upalpha_T[i,t,a]*np.sqrt(1-alpha_array[a]**2) for a in range(n_alpha))
                - sum(model.phi_downalpha_T[i,t,a]*np.sqrt(1-alpha_array[a]**2) for a in range(n_alpha))
                ==0
            )


            model.dLdG_R_sto.add(
                CV_R_sto[i]*365*0.5
                - model.lambda_R[i, t]
                + sum(model.phi_upalpha_char[i, t, a]*np.sqrt(1-alpha_array[a]**2) for a in range(n_alpha))
                - sum(model.phi_downalpha_char[i, t, a]*np.sqrt(1-alpha_array[a]**2) for a in range(n_alpha))
                + sum(model.phi_upalpha_dis[i, t, a]*np.sqrt(1-alpha_array[a]**2) for a in range(n_alpha))
                - sum(model.phi_downalpha_dis[i, t, a]*np.sqrt(1-alpha_array[a]**2) for a in range(n_alpha))
                ==0
            )

            if t==0:
                model.dLdSoc.add(
                    model.phi_up_soc[i,t]
                    + model.gamma_neutral[i]
                    - model.gamma[i,t]
                    + model.phi_downA_soc[i,t]
                    == 0
                )

            elif t==len(model.num_time)-1:
                model.dLdSoc.add(
                    model.phi_up_soc[i,t]
                    + model.gamma[i, t]
                    - model.gamma_neutral[i]
                    + model.phi_downA_soc[i,t]
                    == 0
                )

                model.dLdG_A_dis.add(
                    model.gamma_neutral[i]
                    - model.lambda_A[i,t]
                    + sum((alpha_array[a]*model.phi_upalpha_dis[i,t,a]) for a in range(n_alpha))
                    + sum((alpha_array[a] * model.phi_downalpha_dis[i, t, a]) for a in range(n_alpha))
                    - model.phi_downA_dis[i,t]
                    == 0
                )

                model.dLdG_A_char.add(
                    - eta[i] * model.gamma_neutral[i]
                    + model.lambda_A[i, t]
                    + sum((alpha_array[a] * model.phi_upalpha_char[i, t, a]) for a in range(n_alpha))
                    + sum((alpha_array[a] * model.phi_downalpha_char[i, t, a]) for a in range(n_alpha))
                    - model.phi_downA_char[i, t]
                    == 0
                )

            else:
                model.dLdG_A_dis.add(
                    model.gamma[i,t]
                    - model.lambda_A[i, t]
                    + sum((alpha_array[a] * model.phi_upalpha_dis[i, t, a])  for a in range(n_alpha))
                    + sum((alpha_array[a] * model.phi_downalpha_dis[i, t, a]) for a in range(n_alpha))
                    - model.phi_downA_dis[i, t]
                    == 0
                )

                model.dLdG_A_char.add(
                    - eta[i] * model.gamma[i,t]
                    + model.lambda_A[i, t]
                    + sum((alpha_array[a] * model.phi_upalpha_char[i, t, a]) for a in range(n_alpha))
                    + sum((alpha_array[a] * model.phi_downalpha_char[i, t, a]) for a in range(n_alpha))
                    - model.phi_downA_char[i, t]
                    == 0
                )

                model.dLdSoc.add(
                    model.phi_up_soc[i,t]
                    + model.gamma[i, t - 1]
                    - model.gamma[i, t]
                    + model.phi_downA_soc[i,t]
                    == 0
                )

def Complementary_Slackness(model, As, At, Asto,Tau_e, Tau_d, Tau_R, k, eta, T, n_alpha, Sol_ava, D, D_R):
    model.phi_upA_solava_1 = pyo.ConstraintList()#
    model.phi_up_soc_1 = pyo.ConstraintList()#
    model.beta_1 = pyo.ConstraintList()#
    model.phi_upalpha_T_1 = pyo.ConstraintList()#
    model.phi_downalpha_T_1 = pyo.ConstraintList()#
    model.phi_upalpha_char_1 = pyo.ConstraintList()#
    model.phi_downalpha_char_1 = pyo.ConstraintList()#
    model.phi_upalpha_dis_1 = pyo.ConstraintList()#
    model.phi_downalpha_dis_1 = pyo.ConstraintList()#
    model.sigma_sol_1 = pyo.ConstraintList()#
    model.sigma_T_1 = pyo.ConstraintList()#
    model.sigma_sto_1 = pyo.ConstraintList()#
    model.sigma_A_in_1 = pyo.ConstraintList()#
    model.sigma_A_out_1 = pyo.ConstraintList()#
    model.sigma_R_in_1 = pyo.ConstraintList()#
    model.sigma_R_out_1 = pyo.ConstraintList()#
    model.phi_downA_char_1 = pyo.ConstraintList()#
    model.phi_downA_dis_1 = pyo.ConstraintList()#
    model.phi_downA_sol_1 = pyo.ConstraintList()#
    model.phi_downA_T_1 = pyo.ConstraintList()#
    model.phi_downA_soc_1 = pyo.ConstraintList()#
    model.phi_upA_solava_2 = pyo.ConstraintList()
    model.phi_up_soc_2 = pyo.ConstraintList()
    model.beta_2 = pyo.ConstraintList()
    model.phi_upalpha_T_2 = pyo.ConstraintList()
    model.phi_downalpha_T_2 = pyo.ConstraintList()
    model.phi_upalpha_char_2 = pyo.ConstraintList()
    model.phi_downalpha_char_2 = pyo.ConstraintList()
    model.phi_upalpha_dis_2 = pyo.ConstraintList()
    model.phi_downalpha_dis_2 = pyo.ConstraintList()
    model.sigma_sol_2 = pyo.ConstraintList()
    model.sigma_T_2 = pyo.ConstraintList()
    model.sigma_sto_2 = pyo.ConstraintList()
    model.sigma_A_in_2 = pyo.ConstraintList()
    model.sigma_A_out_2 = pyo.ConstraintList()
    model.sigma_R_in_2 = pyo.ConstraintList()
    model.sigma_R_out_2 = pyo.ConstraintList()
    model.phi_downA_char_2 = pyo.ConstraintList()
    model.phi_downA_dis_2 = pyo.ConstraintList()
    model.phi_downA_sol_2 = pyo.ConstraintList()
    model.phi_downA_T_2 = pyo.ConstraintList()
    model.phi_downA_soc_2 = pyo.ConstraintList()
    model.help_sol=pyo.ConstraintList()
    model.help_sto=pyo.ConstraintList()
    model.SIGMA_sol_MAX_1=pyo.ConstraintList()
    model.SIGMA_sol_MAX_2 = pyo.ConstraintList()
    model.SIGMA_t_MAX_1 = pyo.ConstraintList()
    model.SIGMA_t_MAX_2 = pyo.ConstraintList()
    model.SIGMA_sto_MAX_1 = pyo.ConstraintList()
    model.SIGMA_sto_MAX_2 = pyo.ConstraintList()
    model.aux = pyo.ConstraintList()
    M=1e5*10000
    M1=1e5*10000
    M_beta=10
    Tmax=len(model.num_time)
    alpha_array = np.linspace(0, 1, n_alpha)

    PS_MAX=100
    PT_MAX=0
    STO_MAX=0
    for i in model.num_consumers:
        model.SIGMA_sol_MAX_1.add(
            -1*(model.Ps[i]-PS_MAX)<=M*model.u_SIGMA_sol_MAX[i]
        )
        model.SIGMA_sol_MAX_2.add(
            model.SIGMA_sol_MAX[i]<=M*(1-model.u_SIGMA_sol_MAX[i])
        )

        model.SIGMA_t_MAX_1.add(
            -1 * (model.Pt[i] - PT_MAX) <= M * model.u_SIGMA_t_MAX[i]
        )
        model.SIGMA_t_MAX_2.add(
            model.SIGMA_t_MAX[i] <= M * (1 - model.u_SIGMA_t_MAX[i])
        )

        model.SIGMA_sto_MAX_1.add(
            -1 * (model.Sto[i] - STO_MAX) <= M * model.u_SIGMA_sto_MAX[i]
        )
        model.SIGMA_sto_MAX_2.add(
            model.SIGMA_sto_MAX[i] <= M * (1 - model.u_SIGMA_sto_MAX[i])
        )
        # dldPs
        model.beta_1.add(
            As[i] * model.Ps[i]
            + Asto[i] * model.Sto[i]
            + At[i] * model.Pt[i]
            - k[i]
            >=
            - k[i] * model.u_beta[i]
        )

        model.beta_2.add(
            model.beta[i]
            <= M * (1 - model.u_beta[i])
        )
        # dldGsol
        model.sigma_sol_1.add(
            model.Ps[i]<= (k[i]/As[i])*model.u_sigma_sol[i]
        )

        model.sigma_sol_2.add(
            model.sigma_sol[i]<= M * (1-model.u_sigma_sol[i])
        )
        # dldGt
        model.sigma_T_1.add(
            model.Pt[i]
            <= (k[i]/At[i]) * model.u_sigma_T[i]
        )

        model.sigma_T_2.add(
            model.sigma_T[i]
            <= M * (1-model.u_sigma_T[i])
        )
        #dLdSto
        model.sigma_sto_1.add(
            model.Sto[i]
            <= (k[i]/Asto[i]) * model.u_sigma_sto[i]
        )

        model.sigma_sto_2.add(
            model.sigma_sto[i]
            <= M * (1- model.u_sigma_sto[i])
        )


        for t in model.num_time:
            #solar availability
            model.phi_upA_solava_1.add(
                -1*(model.Gsol_A[i,t]
                - model.Ps[i] *Sol_ava[i,t])
                <=
                (k[i]/As[i]) * model.u_phi_upA_solava[i,t] #(k[i]*Sol_ava[i,t]/As[i])
            )

            model.phi_upA_solava_2.add(
                model.phi_upA_solava[i,t]
                <= M1*(1-model.u_phi_upA_solava[i,t])#(Tau_e[i,t]*0.5 + Tau_d[i,t])*365
            )
            #positiveness solar injections
            model.phi_downA_sol_1.add(
                model.Gsol_A[i, t]
                <= (k[i]/As[i]+1) * model.u_phi_downA_sol[i, t]
            )

            model.phi_downA_sol_2.add(
                model.phi_downA_sol[i, t]
                <= M1 * (1 - model.u_phi_downA_sol[i, t])
            )
            #Help constraint
            model.help_sol.add(
                model.u_phi_downA_sol[i,t]
                + model.u_phi_upA_solava[i,t]
                <=
                1
            )
            # Energy Storage capacity
            model.phi_up_soc_1.add(
                model.Soc[i,t]
                - T[i] *model.Sto[i]
                >= -((k[i]*T[i])/Asto[i]) * model.u_phi_up_soc[i,t]
            )

            model.phi_up_soc_2.add(
                model.phi_up_soc[i,t]
                <=M * (1-model.u_phi_up_soc[i,t])
            )
            #postiveness Energy withdraws
            model.sigma_A_in_1.add(
                model.Gin_A[i,t]
                <= (D[i,t]+(k[i]/Asto[i])) *model.u_sigma_A_in[i,t]
            )

            model.sigma_A_in_2.add(
                model.sigma_A_in [i,t]
                <= M* (1-model.u_sigma_A_in[i,t])
            )
            # postiveness Energy injections
            model.sigma_A_out_1.add(
                model.Gout_A[i, t]
                <= (-D[i,t]+(k[i]/At[i])+(k[i]/As[i])+(k[i]/Asto[i])) * model.u_sigma_A_out[i, t]
            )

            model.sigma_A_out_2.add(
                model.sigma_A_out[i, t]
                <= M * (1 - model.u_sigma_A_out[i, t])
            )
            # postiveness reactives withdraws
            model.sigma_R_in_1.add(
                model.Gin_R[i, t]
                <= (D_R[i,t]+(k[i]/At[i])+(k[i]/Asto[i])) * model.u_sigma_R_in[i, t]
            )

            model.sigma_R_in_2.add(
                model.sigma_R_in[i, t]
                <= M * (1 - model.u_sigma_R_in[i, t])
            )
            # postiveness reactives injections
            model.sigma_R_out_1.add(
                model.Gout_R[i, t]
                <= (-D_R[i,t]+(k[i]/At[i])+(k[i]/Asto[i])) * model.u_sigma_R_out[i, t]
            )

            model.sigma_R_out_2.add(
                model.sigma_R_out[i, t]
                <= M * (1 - model.u_sigma_R_out[i, t])
            )
            #positiveness charging
            model.phi_downA_char_1.add(
                model.Gchar_A[i,t]
                <= (k[i]/Asto[i]) * model.u_phi_downA_char[i,t]
            )

            model.phi_downA_char_2.add(
                model.phi_downA_char[i,t]
                <= M * (1- model.u_phi_downA_char[i,t])
            )
            # positiveness discharging
            model.phi_downA_dis_1.add(
                model.Gdis_A[i, t]
                <= (k[i]/Asto[i]) * model.u_phi_downA_dis[i, t]
            )

            model.phi_downA_dis_2.add(
                model.phi_downA_dis[i, t]
                <= M * (1 - model.u_phi_downA_dis[i, t])

            )
            # positiveness thermal injections
            model.phi_downA_T_1.add(
                model.Gt_A[i, t]
                <= (k[i]/At[i]) * model.u_phi_downA_T[i, t]
            )

            model.phi_downA_T_2.add(
                model.phi_downA_T[i, t]
                <= M * (1 - model.u_phi_downA_T[i, t])
            )
        #     # Positiveness SOC
            model.phi_downA_soc_1.add(
                model.Soc[i, t]
                <= (k[i]*T[i]/Asto[i]) * model.u_phi_downA_soc[i, t]
            )

            model.phi_downA_soc_2.add(
                model.phi_downA_soc[i, t]
                <= M * (1 - model.u_phi_downA_soc[i, t])
            )

            model.help_sto.add(
                sum((model.u_phi_upalpha_char[i, t, a]) for a in range(n_alpha))
                + sum((model.u_phi_downalpha_char[i, t, a]) for a in range(n_alpha))
                >= 2 * n_alpha - 1
            )
            model.help_sto.add(
                sum((model.u_phi_upalpha_dis[i, t, a]) for a in range(n_alpha))
                + sum((model.u_phi_downalpha_dis[i, t, a]) for a in range(n_alpha))
                >= 2 * n_alpha - 1
            )
            model.help_sto.add(
                sum((model.u_phi_upalpha_char[i, t, a]) for a in range(n_alpha))
                + sum((model.u_phi_downalpha_dis[i, t, a]) for a in range(n_alpha))
                >= 2 * n_alpha - 1
            )
            model.help_sto.add(
                sum((model.u_phi_upalpha_dis[i, t, a]) for a in range(n_alpha))
                + sum((model.u_phi_downalpha_char[i, t, a]) for a in range(n_alpha))
                >= 2 * n_alpha - 1
            )

            for a in range(n_alpha):
                # Thermal circle up
                model.phi_upalpha_T_1.add(
                    model.Gt_R[i,t]*np.sqrt(1-alpha_array[a]**2)
                    - (-alpha_array[a]*model.Gt_A[i,t]+model.Pt[i])
                    >= -(k[i]/At[i]) * model.u_phi_upalpha_T[i,t,a]
                )

                model.phi_upalpha_T_2.add(
                    model.phi_upalpha_T[i,t,a]
                    <= M* (1-model.u_phi_upalpha_T[i,t,a])
                )
                # Thermal circle down
                model.phi_downalpha_T_1.add(
                    - model.Gt_R[i, t] * np.sqrt(1-alpha_array[a]**2)
                    - (-alpha_array[a] * model.Gt_A[i, t] + model.Pt[i])
                    >= -(k[i]/At[i]) * model.u_phi_downalpha_T[i, t, a]
                )

                model.phi_downalpha_T_2.add(
                    model.phi_downalpha_T[i, t, a]
                    <= M * (1-model.u_phi_downalpha_T[i, t, a])
                )
                # charging circle up
                model.phi_upalpha_char_1.add(
                    model.Gsto_R[i, t] * np.sqrt(1 - alpha_array[a] ** 2)
                    - (-alpha_array[a] * model.Gchar_A[i, t] + model.Sto[i])
                    >= -4*(k[i]/Asto[i]) *M * model.u_phi_upalpha_char[i, t, a]
                )

                model.phi_upalpha_char_2.add(
                    model.phi_upalpha_char[i,t,a]
                    <= M* (1-model.u_phi_upalpha_char[i,t,a])
                )
                #charging circle down
                model.phi_downalpha_char_1.add(
                    - model.Gsto_R[i, t] * np.sqrt(1 - alpha_array[a] ** 2)
                    - (-alpha_array[a] * model.Gchar_A[i, t] + model.Sto[i])
                    >= -4*(k[i]/Asto[i]) *M * model.u_phi_downalpha_char[i, t, a]
                )

                model.phi_downalpha_char_2.add(
                    model.phi_downalpha_char[i, t, a]
                    <= M * (1 - model.u_phi_downalpha_char[i, t, a])

                )
                # discharging circle up
                model.phi_upalpha_dis_1.add(
                    model.Gsto_R[i, t] * np.sqrt(1 - alpha_array[a] ** 2)
                    - (-alpha_array[a] * model.Gdis_A[i, t] + model.Sto[i])
                    >= -4*(k[i]/Asto[i]) *M  * model.u_phi_upalpha_dis[i, t, a]
                )

                model.phi_upalpha_dis_2.add(
                    model.phi_upalpha_dis[i, t, a]
                    <= M * (1 - model.u_phi_upalpha_dis[i, t, a])
                )
                # discharging circle down
                model.phi_downalpha_dis_1.add(
                    - model.Gsto_R[i, t] * np.sqrt(1 - alpha_array[a] ** 2)
                    - (-alpha_array[a] * model.Gdis_A[i, t] + model.Sto[i])
                    >= -4*(k[i]/Asto[i]) *M  * model.u_phi_downalpha_dis[i, t, a]
                )

                model.phi_downalpha_dis_2.add(
                    model.phi_downalpha_dis[i, t, a]
                    <= M * (1 - model.u_phi_downalpha_dis[i, t, a])
                )

def deleting_first_order_condition(model):
    del model.dLdPs
    del model.dLdPt
    del model.dLdSto
    del model.dLdG_A_in
    del model.dLdG_A_out
    del model.dLdG_A_T
    del model.dLdG_A_sol
    del model.dLdG_A_dis
    del model.dLdG_A_char
    del model.dLdSoc
    del model.dLdG_R_in
    del model.dLdG_R_out
    del model.dLdG_R_T
    del model.dLdG_R_sol
    del model.dLdG_R_sto
    del model.dLdPs_index
    del model.dLdPt_index
    del model.dLdSto_index
    del model.dLdG_A_in_index
    del model.dLdG_A_out_index
    del model.dLdG_A_T_index
    del model.dLdG_A_sol_index
    del model.dLdG_A_dis_index
    del model.dLdG_A_char_index
    del model.dLdSoc_index
    del model.dLdG_R_in_index
    del model.dLdG_R_out_index
    del model.dLdG_R_T_index
    del model.dLdG_R_sol_index
    del model.dLdG_R_sto_index

def pairing_by_branches(model):
    model.Ps_pairing_1 = pyo.ConstraintList()
    model.Ps_pairing_2 = pyo.ConstraintList()
    model.Ps_pairing_3 = pyo.ConstraintList()
    model.Ps_pairing_4 = pyo.ConstraintList()
    model.Ps_pairing_5 = pyo.ConstraintList()
    model.Ps_pairing_6 = pyo.ConstraintList()
    model.Sto_pairing_1 = pyo.ConstraintList()
    model.Sto_pairing_2 = pyo.ConstraintList()
    model.Sto_pairing_3 = pyo.ConstraintList()
    model.Sto_pairing_4 = pyo.ConstraintList()
    model.Sto_pairing_5 = pyo.ConstraintList()
    model.Sto_pairing_6 = pyo.ConstraintList()

    #Medium/Rich
    model.Ps_pairing_1.add(
        model.Ps[1]==model.Ps[23]
    )
    model.Sto_pairing_1.add(
        model.Sto[1] == model.Sto[23]
    )
    model.Ps_pairing_1.add(
        model.Ps[1] == model.Ps[11]
    )
    model.Sto_pairing_1.add(
        model.Sto[1] == model.Sto[11]
    )
    #Rich/Rich
    model.Ps_pairing_2.add(
        model.Ps[6] == model.Ps[7]
    )
    model.Sto_pairing_2.add(
        model.Sto[6] == model.Sto[7]
    )
    model.Ps_pairing_2.add(
        model.Ps[6] == model.Ps[8]
    )
    model.Sto_pairing_2.add(
        model.Sto[6] == model.Sto[8]
    )
    #Rich
    model.Ps_pairing_3.add(
        model.Ps[8] == model.Ps[4]
    )
    model.Sto_pairing_3.add(
        model.Sto[8] == model.Sto[4]
    )
    #Medium
    model.Ps_pairing_4.add(
        model.Ps[11] == model.Ps[10]
    )
    model.Sto_pairing_4.add(
        model.Sto[11] == model.Sto[10]
    )
    model.Ps_pairing_4.add(
        model.Ps[10] == model.Ps[13]
    )
    model.Sto_pairing_4.add(
        model.Sto[10] == model.Sto[13]
    )
    model.Ps_pairing_4.add(
        model.Ps[13] == model.Ps[14]
    )
    model.Sto_pairing_4.add(
        model.Sto[13] == model.Sto[14]
    )
    #poor
    model.Ps_pairing_5.add(
        model.Ps[18] == model.Ps[17]
    )
    model.Sto_pairing_5.add(
        model.Sto[18] == model.Sto[17]
    )
    model.Ps_pairing_5.add(
        model.Ps[18] == model.Ps[21]
    )
    model.Sto_pairing_5.add(
        model.Sto[18] == model.Sto[21]
    )
    #poor/poor
    model.Ps_pairing_5.add(
        model.Ps[21] == model.Ps[22]
    )
    model.Sto_pairing_5.add(
        model.Sto[21] == model.Sto[22]
    )





































