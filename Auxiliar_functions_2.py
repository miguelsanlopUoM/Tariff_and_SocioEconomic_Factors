import pyomo.environ as pyo
import numpy as np

## Set of constrants for Centralised and Decentralised investment models

def Storage_model_2(model, Bat_pro_1):
    model.Sto_MAX= pyo.ConstraintList()
    model.Batt_pos= pyo.ConstraintList()
    model.Batt_neg = pyo.ConstraintList()
    model.Sto_No_Reactives= pyo.ConstraintList()

    STOMAX=10

    model.Sto_1 = pyo.Var(model.num_consumers, domain=pyo.NonNegativeReals)

    model.Sto_sum = pyo.ConstraintList()

    Bat_positive_1 = np.zeros(Bat_pro_1.shape)
    Bat_negative_1 = np.zeros(Bat_pro_1.shape)


    for i in model.num_consumers:
        model.Sto_MAX.add(
            model.Sto[i]<=STOMAX
        )
        model.Sto_sum.add(
            model.Sto_1[i]  == model.Sto[i]
        )#model.Sto_1[i] + model.Sto_2[i] + model.Sto_3[i] == model.Sto[i]
        for t in model.num_time:
            if Bat_pro_1[i,t]>0:
                Bat_positive_1[i,t]=Bat_pro_1[i,t]
            else:
                Bat_negative_1[i,t]=Bat_pro_1[i,t]

        for t in model.num_time:
            model.Batt_pos.add(
                model.Gdis_A[i, t] ==
                Bat_positive_1[i,t]*model.Sto_1[i]
            ) #phi_STO_EQ_pos + Bat_positive_2[i,t]*model.Sto_2[i] + Bat_positive_3[i,t]*model.Sto_3[i]
            model.Batt_neg.add(
                - model.Gchar_A[i, t] ==
                Bat_negative_1[i, t] * model.Sto_1[i]
            ) #phi_STO_EQ_neg                 + Bat_negative_2[i, t] * model.Sto_2[i] + Bat_negative_3[i, t] * model.Sto_3[i]
            model.Sto_No_Reactives.add(
                model.Gsto_R[i,t]==0
            )

def Thermal_model_2(model, n_alpha):
    model.G_ther_up = pyo.ConstraintList()
    model.G_ther_down = pyo.ConstraintList()
    alpha_array = np.linspace(0, 1, n_alpha)
    model.Pt_MAX=pyo.ConstraintList()
    Pt_MAX=0

    for i in model.num_consumers:
        model.Pt_MAX.add(
            model.Pt[i]<=Pt_MAX
        )
        for t in model.num_time:
            model.G_ther_up.add(
                model.Gt_A[i, t]
                <=
                model.Pt[i]
            )

            model.G_ther_down.add(
                model.Gt_R[i, t]
                <=
                model.Pt[i]
            )

def Dual_variables_2(model, n_alpha):
    model.phi_upA_solava = pyo.Var(model.num_consumers, model.num_time, domain=pyo.NonNegativeReals)
    model.phi_up_soc = pyo.Var(model.num_consumers, model.num_time, domain=pyo.NonNegativeReals)
    model.beta = pyo.Var(model.num_consumers, domain=pyo.NonNegativeReals)
    model.lambda_R = pyo.Var(model.num_consumers, model.num_time, domain=pyo.Reals)
    model.lambda_A = pyo.Var(model.num_consumers, model.num_time, domain=pyo.Reals)
    model.phi_upalpha_T = pyo.Var(model.num_consumers, model.num_time, range(n_alpha), domain=pyo.NonNegativeReals)
    model.phi_downalpha_T = pyo.Var(model.num_consumers, model.num_time, range(n_alpha), domain=pyo.NonNegativeReals)
    model.sigma_sol = pyo.Var(model.num_consumers, domain=pyo.NonNegativeReals)
    model.sigma_T = pyo.Var(model.num_consumers, domain=pyo.NonNegativeReals)
    model.sigma_sto = pyo.Var(model.num_consumers, domain=pyo.Reals)
    model.sigma_A_in = pyo.Var(model.num_consumers, model.num_time, domain=pyo.NonNegativeReals)
    model.sigma_A_out = pyo.Var(model.num_consumers, model.num_time, domain=pyo.NonNegativeReals)
    model.sigma_R_in = pyo.Var(model.num_consumers, model.num_time, domain=pyo.NonNegativeReals)
    model.sigma_R_out = pyo.Var(model.num_consumers, model.num_time, domain=pyo.NonNegativeReals)
    model.phi_downA_char = pyo.Var(model.num_consumers, model.num_time, domain=pyo.NonNegativeReals)
    model.phi_downA_dis = pyo.Var(model.num_consumers, model.num_time, domain=pyo.NonNegativeReals)
    model.phi_downA_sol = pyo.Var(model.num_consumers, model.num_time, domain=pyo.NonNegativeReals)
    model.phi_downA_T = pyo.Var(model.num_consumers, model.num_time, domain=pyo.NonNegativeReals)
    model.SIGMA_sol_MAX=pyo.Var(model.num_consumers, domain=pyo.NonNegativeReals)
    model.SIGMA_t_MAX = pyo.Var(model.num_consumers, domain=pyo.NonNegativeReals)
    model.SIGMA_sto_MAX = pyo.Var(model.num_consumers, domain=pyo.NonNegativeReals)
    model.phi_STO_EQ_pos = pyo.Var(model.num_consumers, model.num_time, domain = pyo.Reals)
    model.phi_STO_EQ_neg=pyo.Var(model.num_consumers, model.num_time, domain=pyo.Reals)
    model.alpha_sto = pyo.Var(model.num_consumers, domain=pyo.Reals)
    model.sigma_sto_1= pyo.Var(model.num_consumers, domain=pyo.NonNegativeReals)
    model.sigma_sto_2= pyo.Var(model.num_consumers, domain=pyo.NonNegativeReals)
    model.sigma_sto_3= pyo.Var(model.num_consumers, domain=pyo.NonNegativeReals)

def First_order_condition_2(model, As, At, Asto, Tau_e, Tau_d, Tau_R, CV, CV_R,eta, T, n_alpha, Sol_ava, sell_factor,Bat_pro_1):
    model.dLdPs = pyo.ConstraintList()
    model.dLdPt = pyo.ConstraintList()
    model.dLdSto = pyo.ConstraintList()
    model.dLdSto_1 = pyo.ConstraintList()
    model.dLdSto_2 = pyo.ConstraintList()
    model.dLdSto_3 = pyo.ConstraintList()
    model.dLdG_A_in = pyo.ConstraintList()
    model.dLdG_A_out = pyo.ConstraintList()
    model.dLdG_A_T = pyo.ConstraintList()
    model.dLdG_A_sol = pyo.ConstraintList()
    model.dLdG_A_dis = pyo.ConstraintList()
    model.dLdG_A_char = pyo.ConstraintList()
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

    Bat_positive_1 = np.zeros(Bat_pro_1.shape)
    Bat_negative_1 = np.zeros(Bat_pro_1.shape)


    alpha_array = np.linspace(0, 1, n_alpha)

    for i in model.num_consumers:
        for t in model.num_time:
            if Bat_pro_1[i, t] > 0:
                Bat_positive_1[i, t] = Bat_pro_1[i, t]
            else:
                Bat_negative_1[i, t] = Bat_pro_1[i, t]


        model.dLdPs.add(
            As[i]
            + As[i] * model.beta[i]
            - sum(Sol_ava[i, t] * model.phi_upA_solava[i, t] for t in model.num_time)
            - model.sigma_sol[i]
            + model.SIGMA_sol_MAX[i]
            == 0
        )


        model.dLdSto.add(
            Asto[i]
            + Asto[i]*model.beta[i]
            - model.alpha_sto[i]
            + model.SIGMA_sto_MAX[i]
            ==0
        )

        model.dLdSto_1.add(
            model.alpha_sto[i]
            - sum(Bat_positive_1[i, t]*model.phi_STO_EQ_pos[i,t] for t in model.num_time)
            - sum(Bat_negative_1[i, t]*model.phi_STO_EQ_neg[i,t] for t in model.num_time)
            - model.sigma_sto_1[i]
            ==0
        )

        for t in model.num_time:
            model.dLdG_A_in.add(
                ((Tau_e[i, t]+ Tau_d[i, t]))
                + model.lambda_A[i, t]
                - model.sigma_A_in[i, t]
                == 0
            )

            model.dLdG_A_out.add(
                (-1 * (Tau_e[i, t]*sell_factor+ Tau_d[i, t]))
                - model.lambda_A[i, t]
                - model.sigma_A_out[i, t]
                == 0
            ) #####################

            model.dLdG_A_sol.add(
                model.phi_upA_solava[i, t]
                + model.lambda_A[i, t]
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

            model.dLdG_A_dis.add(
                model.phi_STO_EQ_pos[i, t]
                + model.lambda_A[i, t]
                == 0
            )#- model.phi_downA_dis[i,t] - model.sigma_Gdis_A[i, t]

            model.dLdG_A_char.add(
                - model.phi_STO_EQ_neg[i, t]
                - model.lambda_A[i, t]
                == 0
            )#- model.phi_downA_char[i, t]- model.sigma_Gchar_A[i, t]


def Integer_variables_2(model, n_alpha):
    model.u_phi_upA_solava=pyo.Var(model.num_consumers, model.num_time,domain=pyo.Binary)
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
    model.u_SIGMA_sol_MAX = pyo.Var(model.num_consumers, domain=pyo.Binary)
    model.u_SIGMA_t_MAX = pyo.Var(model.num_consumers, domain=pyo.Binary)
    model.u_SIGMA_sto_MAX = pyo.Var(model.num_consumers, domain=pyo.Binary)
    model.u_sigma_sto_1 = pyo.Var(model.num_consumers, domain=pyo.Binary)
    model.u_sigma_sto_2 = pyo.Var(model.num_consumers, domain=pyo.Binary)
    model.u_sigma_sto_3 = pyo.Var(model.num_consumers, domain=pyo.Binary)

def Complementary_Slackness_2(model, As, At, Asto,Tau_e, Tau_d, Tau_R, k, eta, T, n_alpha, Sol_ava, D, D_R ):
    model.phi_upA_solava_1 = pyo.ConstraintList()#
    model.beta_1 = pyo.ConstraintList()#
    model.phi_upalpha_T_1 = pyo.ConstraintList()#
    model.phi_downalpha_T_1 = pyo.ConstraintList()#
    model.phi_upalpha_char_1 = pyo.ConstraintList()#
    model.phi_downalpha_char_1 = pyo.ConstraintList()#
    model.phi_upalpha_dis_1 = pyo.ConstraintList()#
    model.phi_downalpha_dis_1 = pyo.ConstraintList()#
    model.sigma_sol_1 = pyo.ConstraintList()#
    model.sigma_T_1 = pyo.ConstraintList()#
    model.v_sigma_sto_1 = pyo.ConstraintList()#
    model.v1_sigma_sto_1 = pyo.ConstraintList()
    model.v2_sigma_sto_1 = pyo.ConstraintList()
    model.v3_sigma_sto_1 = pyo.ConstraintList()
    model.sigma_A_in_1 = pyo.ConstraintList()#
    model.sigma_A_out_1 = pyo.ConstraintList()#
    model.sigma_R_in_1 = pyo.ConstraintList()#
    model.sigma_R_out_1 = pyo.ConstraintList()#
    model.phi_downA_char_1 = pyo.ConstraintList()#
    model.phi_downA_dis_1 = pyo.ConstraintList()#
    model.phi_downA_sol_1 = pyo.ConstraintList()#
    model.phi_downA_T_1 = pyo.ConstraintList()#
    model.phi_upA_solava_2 = pyo.ConstraintList()
    model.beta_2 = pyo.ConstraintList()
    model.phi_upalpha_T_2 = pyo.ConstraintList()
    model.phi_downalpha_T_2 = pyo.ConstraintList()
    model.phi_upalpha_char_2 = pyo.ConstraintList()
    model.phi_downalpha_char_2 = pyo.ConstraintList()
    model.phi_upalpha_dis_2 = pyo.ConstraintList()
    model.phi_downalpha_dis_2 = pyo.ConstraintList()
    model.sigma_sol_2 = pyo.ConstraintList()
    model.sigma_T_2 = pyo.ConstraintList()
    model.v_sigma_sto_2 = pyo.ConstraintList()
    model.v1_sigma_sto_2 = pyo.ConstraintList()
    model.v2_sigma_sto_2 = pyo.ConstraintList()
    model.v3_sigma_sto_2 = pyo.ConstraintList()
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
    model.help_out_A= pyo.ConstraintList()
    model.help_out_R = pyo.ConstraintList()
    model.help_battery=pyo.ConstraintList()
    model.help_SIGMA_sol_MAX = pyo.ConstraintList()
    model.help_SIGMA_t_MAX = pyo.ConstraintList()
    model.help_SIGMA_sto_MAX = pyo.ConstraintList()
    model.help_sigma_T = pyo.ConstraintList()
    model.help_phi_downA_T= pyo.ConstraintList()
    model.solar_help_2 =pyo.ConstraintList()


    M=1e6
    M1=1e5
    M_beta=M1
    Tmax=len(model.num_time)
    alpha_array = np.linspace(0, 1, n_alpha)

    PS_MAX=10
    PT_MAX=0
    STO_MAX=10
    for i in model.num_consumers:
        model.SIGMA_sol_MAX_1.add(
            -1*(model.Ps[i]-PS_MAX)<=PS_MAX*model.u_SIGMA_sol_MAX[i]
        )
        model.SIGMA_sol_MAX_2.add(
            model.SIGMA_sol_MAX[i]<=M*(1-model.u_SIGMA_sol_MAX[i])
        )


        model.SIGMA_sto_MAX_1.add(
            -1 * (model.Sto[i] - STO_MAX) <= STO_MAX * model.u_SIGMA_sto_MAX[i]
        )
        model.SIGMA_sto_MAX_2.add(
            model.SIGMA_sto_MAX[i] <= M * (1 - model.u_SIGMA_sto_MAX[i])
        )


        # dldPs
        model.beta_1.add(
            As[i] * model.Ps[i]
            + Asto[i] * model.Sto[i]
            - k[i]
            >=
            - k[i] * model.u_beta[i]
        )

        model.beta_2.add(
            model.beta[i]
            <= M_beta * (1 - model.u_beta[i])
        )
        # dldGsol
        model.sigma_sol_1.add(
            model.Ps[i]<= (k[i]/As[i])*model.u_sigma_sol[i]
        )

        model.sigma_sol_2.add(
            model.sigma_sol[i]<= M * (1-model.u_sigma_sol[i])
        )

        model.v1_sigma_sto_1.add(
            model.Sto_1[i]
            <= (k[i] / Asto[i] ) * model.u_sigma_sto_1[i]
        )

        model.v1_sigma_sto_2.add(
            model.sigma_sto_1[i]
            <= M * (1 - model.u_sigma_sto_1[i])
        )

        for t in model.num_time:
            # solar availability
            model.phi_upA_solava_1.add(
                -1*(model.Gsol_A[i,t]
                - model.Ps[i] * Sol_ava[i,t])
                <=
                (k[i]/As[i]) * model.u_phi_upA_solava[i,t]  # (k[i]*Sol_ava[i,t]/As[i])
            )



            model.phi_upA_solava_2.add(
                model.phi_upA_solava[i,t]
                <= M1*(1-model.u_phi_upA_solava[i,t])  # (Tau_e[i,t]*0.5 + Tau_d[i,t])*365
            )
            # positiveness solar injections
            model.phi_downA_sol_1.add(
                model.Gsol_A[i, t]
                <= (k[i]/As[i]) * model.u_phi_downA_sol[i, t]
            )

            model.phi_downA_sol_2.add(
                model.phi_downA_sol[i, t]
                <= M * (1 - model.u_phi_downA_sol[i, t])
            )

            #postiveness Energy withdraws
            model.sigma_A_in_1.add(
                model.Gin_A[i,t]
                <= (D[i,t]+(k[i]/At[i])+(k[i]/Asto[i])) *model.u_sigma_A_in[i,t]
            )

            model.sigma_A_in_2.add(
                model.sigma_A_in [i,t]
                <= M* (1-model.u_sigma_A_in[i,t])
            )
            # postiveness Energy injections
            model.sigma_A_out_1.add(
                model.Gout_A[i, t]
                <= (D[i,t]+(k[i]/At[i])+(k[i]/Asto[i])) * model.u_sigma_A_out[i, t]
            )

            model.sigma_A_out_2.add(
                model.sigma_A_out[i, t]
                <= M * (1 - model.u_sigma_A_out[i, t])
            )


            #help
            # model.help_out_A.add(
            #     model.u_sigma_A_in[i,t] + model.u_sigma_A_out[i, t]<=1
            # )

            # postiveness reactives withdraws
            model.sigma_R_in_1.add(
                model.Gin_R[i, t]
                <= (D_R[i,t]+(k[i]/At[i])+(k[i]/Asto[i])+1) * model.u_sigma_R_in[i, t]
            )

            model.sigma_R_in_2.add(
                model.sigma_R_in[i, t]
                <= M * (1 - model.u_sigma_R_in[i, t])
            )
            # postiveness reactives injections
            model.sigma_R_out_1.add(
                model.Gout_R[i, t]
                <= (D_R[i,t]+(k[i]/At[i])+(k[i]/Asto[i])) * model.u_sigma_R_out[i, t]
            )

            model.sigma_R_out_2.add(
                model.sigma_R_out[i, t]
                <= M * (1 - model.u_sigma_R_out[i, t])
            )
            #help

            model.help_out_R.add(
                model.u_sigma_R_in[i, t]+ model.u_sigma_R_out[i, t]<=1
            )

def Consumer_balance_2(model, D, D_R):
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
                model.Gsol_A[i,t]
            )

            model.energy_balance_R.add(
                (D_R[i, t])
                + model.Gout_R[i, t]
                ==
                model.Gin_R[i, t] +
                model.Gsto_R[i, t] +
                model.Gsol_R[i, t]
            )

def Objective_function_2(model, As, At, Asto, Cmg, Cmg_R, Al, CV, CV_R,n_days):
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
    model.Compensation_losses = pyo.Var(domain=pyo.NonNegativeReals)
    model.Comp_loss_def = pyo.ConstraintList()
    model.Total_cost_definition = pyo.ConstraintList()
    CV_R_sol = np.zeros(len(model.num_consumers))
    CV_R_sto = np.zeros(len(model.num_consumers))
    N_hours_2= n_days*24
    range_1=range((n_days-1)*24)
    range_2=range((n_days-1)*24,n_days*24)
    for i in model.num_consumers:
        CV_R_sol[i]=0
        CV_R_sto[i]=0
    model.Solar_def.add(
        model.Solar_investment ==
        sum(As[i]*model.Ps[i] for i in model.num_consumers)
    )
    model.Sto_def.add(
        model.Sto_investment ==
        sum(Asto[i] * model.Sto[i] for i in model.num_consumers)
    )
    model.Energy_sell_def.add(
        model.Energy_sells == sum(model.Gsysout_A[t]*((Cmg[t]-0.05)*0.5) for t in model.num_time)*(365/n_days)
    )
    model.Energy_buy_def.add(
        model.Energy_buys == sum(model.Gsysin_A[t] * (Cmg[t]*0.5) for t in model.num_time)*(365/n_days)
    )
    model.Reactive_sell_def.add(
        model.Reactive_sells == sum(model.Gsysout_R[t] * ((Cmg_R[t]-0.05)*0.5) for t in model.num_time)*(365/n_days)
    )
    model.Reactive_buy_def.add(
        model.Reactive_buys == sum(model.Gsysin_R[t] * (Cmg_R[t]*0.5) for t in model.num_time)*(365/n_days)
    )
    model.Grid_reinfor_def.add(
        model.Grid_reinforcement == sum(Al[l]*model.Sl[l] for l in model.num_lines)
    )
    model.solar_costs_R_def.add(
        model.solar_costs_R == sum(CV_R_sol[i] * model.Gsol_R[i, t] for i in model.num_consumers for t in model.num_time)*(365/n_days)
    )
    model.storage_costs_R_def.add(
        model.storage_costs_R == sum(CV_R_sto[i] * model.Gsto_R[i, t] for i in model.num_consumers for t in model.num_time)*(365/n_days)
    )
    model.Comp_loss_def.add(
        model.Compensation_losses == sum(sum(500* (model.Q_compensation_pos[j,t] + model.Q_compensation_neg[j,t]) for j in model.num_bus) for t in model.num_time) *(365/n_days)
    )

    model.Total_cost_definition.add(
        model.Total_cost
        ==
        + model.Solar_investment
        + model.Sto_investment
        - model.Energy_sells
        + model.Energy_buys
        - model.Reactive_sells
        + model.Reactive_buys
        + model.Grid_reinforcement
        + model.solar_costs_R
        + model.storage_costs_R
        + model.Compensation_losses
    )
    model.obj = pyo.Objective(expr=(model.Total_cost), sense=pyo.minimize)

def Objective_function_3(model, As, At, Asto, Cmg, Cmg_R, Al, CV, CV_R,n_days):
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
    model.Compensation_losses = pyo.Var(domain=pyo.NonNegativeReals)
    model.Comp_loss_def = pyo.ConstraintList()
    model.Total_cost_definition = pyo.ConstraintList()
    CV_R_sol = np.zeros(len(model.num_consumers))
    CV_R_sto = np.zeros(len(model.num_consumers))
    N_hours_2= n_days*24
    range_1=range((n_days-1)*24)
    range_2=range((n_days-1)*24,n_days*24)
    for i in model.num_consumers:
        CV_R_sol[i]=0
        CV_R_sto[i]=0
    model.Solar_def.add(
        model.Solar_investment ==
        sum(As[i]*model.Ps[i] for i in model.num_consumers)
    )
    model.Sto_def.add(
        model.Sto_investment ==
        sum(Asto[i] * model.Sto[i] for i in model.num_consumers)
    )
    model.Energy_sell_def.add(
        model.Energy_sells == (sum(model.Gsysout_A[t]*((Cmg[t]-0.05)) for t in range_1)*(364/(n_days-1))+
        sum(model.Gsysout_A[t] * ((Cmg[t] - 0.05) ) for t in range_2))
    )
    model.Energy_buy_def.add(
        model.Energy_buys == sum(model.Gsysin_A[t] * (Cmg[t]) for t in range_1)*(364/(n_days-1))
        + sum(model.Gsysin_A[t] * (Cmg[t]) for t in range_2)
    )
    model.Reactive_sell_def.add(
        model.Reactive_sells == sum(model.Gsysout_R[t] * ((Cmg_R[t]-0.05)) for t in range_1)*(364/(n_days-1))
        + sum(model.Gsysout_R[t] * ((Cmg_R[t] - 0.05) ) for t in range_2)
    )
    model.Reactive_buy_def.add(
        model.Reactive_buys == sum(model.Gsysin_R[t] * (Cmg_R[t]) for t in range_1)*(364/(n_days-1))
        + sum(model.Gsysin_R[t] * (Cmg_R[t]) for t in range_2)
    )
    model.Grid_reinfor_def.add(
        model.Grid_reinforcement == sum(Al[l]*model.Sl[l] for l in model.num_lines)
    )
    model.solar_costs_R_def.add(
        model.solar_costs_R == sum(CV_R_sol[i] * model.Gsol_R[i, t] for i in model.num_consumers for t in range_1)*(364/(n_days-1))
        + sum(CV_R_sol[i] * model.Gsol_R[i, t] for i in model.num_consumers for t in range_2)
    )
    model.storage_costs_R_def.add(
        model.storage_costs_R == sum(CV_R_sto[i] * model.Gsto_R[i, t] for i in model.num_consumers for t in range_1)*(364/(n_days-1))
        + sum(CV_R_sto[i] * model.Gsto_R[i, t] for i in model.num_consumers for t in range_2)
    )
    model.Comp_loss_def.add(
        model.Compensation_losses == sum(sum(500* (model.Q_compensation_pos[j,t] + model.Q_compensation_neg[j,t]) for j in model.num_bus) for t in range_1) *(364/(n_days-1))
        + sum(sum(500 * (model.Q_compensation_pos[j, t] + model.Q_compensation_neg[j, t]) for j in model.num_bus) for t in range_2)
    )

    model.Total_cost_definition.add(
        model.Total_cost
        ==
        + model.Solar_investment
        + model.Sto_investment
        - model.Energy_sells
        + model.Energy_buys
        - model.Reactive_sells
        + model.Reactive_buys
        + model.Grid_reinforcement
        + model.solar_costs_R
        + model.storage_costs_R
        + model.Compensation_losses
    )
    model.obj = pyo.Objective(expr=(model.Total_cost), sense=pyo.minimize)

def Planning_contraints_2(model, Ps_val, Sto_val, Pt_val, Sl_val, As, At, Asto, k):
    model.Solar_fixing = pyo.ConstraintList()
    model.Storage_fixing = pyo.ConstraintList()
    model.Thermal_fixing = pyo.ConstraintList()
    model.Thermal_aux = pyo.ConstraintList()
    model.Network_fixing = pyo.ConstraintList()
    model.Budget_constraint_2 = pyo.ConstraintList()

    for i in model.num_consumers:
        model.Budget_constraint_2.add(
            As[i] * model.Ps[i] + Asto[i] * model.Sto[i] <= k[i]
        )
        model.Solar_fixing.add(
            model.Ps[i] >=  Ps_val[i]
        )
        model.Storage_fixing.add(
            model.Sto[i] >=Sto_val[i]
        )
    for l in model.num_lines:
        model.Network_fixing.add(
            model.Sl[l] >= 0  # Sl_val[i]
        )

def Planning_contraints_2_pro(model, Ps_val, Sto_val, Pt_val, Sl_val, As, At, Asto, k):
    model.Solar_fixing = pyo.ConstraintList()
    model.Storage_fixing = pyo.ConstraintList()
    model.Thermal_fixing = pyo.ConstraintList()
    model.Thermal_aux = pyo.ConstraintList()
    model.Network_fixing = pyo.ConstraintList()
    model.Budget_constraint_2 = pyo.ConstraintList()

    for i in model.num_consumers:
        model.Budget_constraint_2.add(
            As[i] * model.Ps[i] + Asto[i] * model.Sto[i] <= k[i]
        )
        model.Solar_fixing.add(
            model.Ps[i] >= Ps_val[i]
        )
        model.Storage_fixing.add(
            model.Sto[i] >= Sto_val[i]
        )

def Storage_model_pro_free(model, eta, T, n_alpha, n_time):
    model.G_char_up = pyo.ConstraintList()
    model.G_dis_up = pyo.ConstraintList()
    model.G_sto_R_fix = pyo.ConstraintList()
    model.SoC_max = pyo.ConstraintList()
    model.SoC_inventory = pyo.ConstraintList()
    model.SoC_neutrality = pyo.ConstraintList()
    model.Sto_MAX= pyo.ConstraintList()
    model.SoC_daily=pyo.ConstraintList()
    STOMAX=10

    for i in model.num_consumers:
        model.SoC_neutrality.add(
            model.Soc[i,0] == model.Soc[i,n_time-1] + eta[i]*model.Gchar_A[i,n_time-1] - model.Gdis_A[i,n_time-1]
        )
        model.SoC_daily.add(
            model.Soc[i,0] == model.Soc[i,23]
        )
        model.SoC_daily.add(
            model.Soc[i, 24] == model.Soc[i, 24*2-1]
        )
        model.SoC_daily.add(
            model.Soc[i, 24*2] == model.Soc[i, 24 * 3 - 1]
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
            model.G_char_up.add(
                model.Gchar_A[i,t] <= model.Sto[i]
            )
            model.G_dis_up.add(
                model.Gdis_A[i,t] <= model.Sto[i]
            )
            model.G_sto_R_fix.add(
                model.Gsto_R[i,t] == 0
            )

def prosumer_objective_function(model, As, At, Asto, CV, Tau_e, Tau_d, Tau_R):
    model.Solar_def = pyo.ConstraintList()
    model.Sto_def = pyo.ConstraintList()
    model.Prosumer_energy_sell_def = pyo.ConstraintList()
    model.Prosumer_energy_buy_def = pyo.ConstraintList()
    model.Prosumer_reactive_sell_def = pyo.ConstraintList()
    model.Prosumer_reactive_buy_def = pyo.ConstraintList()
    model.Prosumer_distribution_costs_def = pyo.ConstraintList()
    model.Prosumer_distribution_benefits_def = pyo.ConstraintList()
    model.Prosumer_total_costs_def = pyo.ConstraintList()

    model.Solar_investment = pyo.Var(domain=pyo.NonNegativeReals)
    model.Sto_investment = pyo.Var(domain=pyo.NonNegativeReals)
    model.Prosumer_energy_sells = pyo.Var( domain=pyo.NonNegativeReals)
    model.Prosumer_reactive_buys = pyo.Var(domain=pyo.NonNegativeReals)
    model.Prosumer_reactive_sells = pyo.Var( domain=pyo.NonNegativeReals)
    model.Prosumer_energy_buys = pyo.Var( domain=pyo.NonNegativeReals)
    model.Prosumer_distribution_costs = pyo.Var( domain=pyo.NonNegativeReals)
    model.Prosumer_distribution_benefits =  pyo.Var( domain=pyo.NonNegativeReals)
    model.Prosumer_total_costs =  pyo.Var( domain=pyo.Reals)

    sell_factor = 0.1

    model.Solar_def.add(
        model.Solar_investment ==
        sum(As[i]*model.Ps[i] for i in model.num_consumers)
    )

    model.Sto_def.add(
        model.Sto_investment ==
        sum(Asto[i] * model.Sto[i] for i in model.num_consumers)
    )

    model.Prosumer_energy_sell_def.add(
        model.Prosumer_energy_sells == sum(model.Gout_A[i,t]*Tau_e[i,t]* sell_factor  for t in model.num_time for i in model.num_consumers)
    )
    model.Prosumer_energy_buy_def.add(
        model.Prosumer_energy_buys == sum(model.Gin_A[i,t] * Tau_e[i,t]  for t in model.num_time for i in model.num_consumers)
    )
    model.Prosumer_reactive_sell_def.add(
        model.Prosumer_reactive_sells == sum(model.Gout_R[i,t]*Tau_R[i,t] for t in model.num_time for i in model.num_consumers)*365
    )
    model.Prosumer_reactive_buy_def.add(
        model.Prosumer_reactive_buys == sum(model.Gin_R[i,t]*Tau_R[i,t] for t in model.num_time for i in model.num_consumers)*365
    )

    model.Prosumer_distribution_costs_def.add(
        model.Prosumer_distribution_costs == sum(model.Gin_A[i,t]*Tau_d[i,t] for t in model.num_time for i in model.num_consumers)
    )

    model.Prosumer_distribution_benefits_def.add(
        model.Prosumer_distribution_benefits == sum(model.Gout_A[i, t] * Tau_d[i, t] for t in model.num_time for i in model.num_consumers)
    )

    model.Prosumer_total_costs_def.add(
        model.Prosumer_total_costs
        ==
        + model.Solar_investment
        + model.Sto_investment
        - model.Prosumer_energy_sells
        + model.Prosumer_energy_buys
        - model.Prosumer_reactive_sells
        + model.Prosumer_reactive_buys
        - model.Prosumer_distribution_benefits
        + model.Prosumer_distribution_costs
    )
    model.obj = pyo.Objective(expr=(model.Prosumer_total_costs), sense=pyo.minimize)




