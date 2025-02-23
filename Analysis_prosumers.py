import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl


Technology = ['_NO_STORAGE','']
Technology = ['','']
Energy_tariff = ['UKE7', 'Cmg', 'FLAT']
Volumetric_share = ['Vol100_PEAK0', 'Vol10_PEAK90', 'Vol30_PEAK70']
OPT_CMG=['_Renew', '_Term']
OPT_CMG=[ '_Renew']
id='1007'

Case_order=["_NO_STORAGE_Cmg_Vol100_PEAK0_Term",
"_NO_STORAGE_UKE7_Vol100_PEAK0_Term",
"_NO_STORAGE_FLAT_Vol100_PEAK0_Term",
"_NO_STORAGE_Cmg_Vol30_PEAK70_Term",
"_NO_STORAGE_UKE7_Vol30_PEAK70_Term",
"_NO_STORAGE_FLAT_Vol30_PEAK70_Term",
"_NO_STORAGE_Cmg_Vol10_PEAK90_Term",
"_NO_STORAGE_UKE7_Vol10_PEAK90_Term",
"_NO_STORAGE_FLAT_Vol10_PEAK90_Term",
"_Cmg_Vol100_PEAK0_Term",
"_UKE7_Vol100_PEAK0_Term",
"_FLAT_Vol100_PEAK0_Term",
"_Cmg_Vol30_PEAK70_Term",
"_UKE7_Vol30_PEAK70_Term",
"_FLAT_Vol30_PEAK70_Term",
"_Cmg_Vol10_PEAK90_Term",
"_UKE7_Vol10_PEAK90_Term",
"_FLAT_Vol10_PEAK90_Term",
]

Case_order=["_NO_STORAGE_Cmg_Vol100_PEAK0_Renew",
"_NO_STORAGE_UKE7_Vol100_PEAK0_Renew",
"_NO_STORAGE_FLAT_Vol100_PEAK0_Renew",
"_NO_STORAGE_Cmg_Vol30_PEAK70_Renew",
"_NO_STORAGE_UKE7_Vol30_PEAK70_Renew",
"_NO_STORAGE_FLAT_Vol30_PEAK70_Renew",
"_NO_STORAGE_Cmg_Vol10_PEAK90_Renew",
"_NO_STORAGE_UKE7_Vol10_PEAK90_Renew",
"_NO_STORAGE_FLAT_Vol10_PEAK90_Renew",
"_Cmg_Vol100_PEAK0_Renew",
"_UKE7_Vol100_PEAK0_Renew",
"_FLAT_Vol100_PEAK0_Renew",
"_Cmg_Vol30_PEAK70_Renew",
"_UKE7_Vol30_PEAK70_Renew",
"_FLAT_Vol30_PEAK70_Renew",
"_Cmg_Vol10_PEAK90_Renew",
"_UKE7_Vol10_PEAK90_Renew",
"_FLAT_Vol10_PEAK90_Renew",
]

Case_format=[
        ('PV','Vol 100','MgC'),
('PV','Vol 100','2-b'),
('PV','Vol 100','Flat'),
('PV','Vol 30 Peak 70','MgC'),
('PV','Vol 30 Peak 70','2-b'),
('PV','Vol 30 Peak 70','Flat'),
('PV','Vol 10 Peak 90','MgC'),
('PV','Vol 10 Peak 90','2-b'),
('PV','Vol 10 Peak 90','Flat'),
('PV + Storage','Vol 100','MgC'),
('PV + Storage','Vol 100','2-b'),
('PV + Storage','Vol 100','Flat'),
('PV + Storage','Vol 30 Peak 70','MgC'),
('PV + Storage','Vol 30 Peak 70','2-b'),
('PV + Storage','Vol 30 Peak 70','Flat'),
('PV + Storage','Vol 10 Peak 90','MgC'),
('PV + Storage','Vol 10 Peak 90','2-b'),
('PV + Storage','Vol 10 Peak 90','Flat')
]


DF=[]
Result_df=[]
for cmg in OPT_CMG:
        i=0
        for sto in Technology:
            i=i+1
            if i==1:
                sto1='_NO_STORAGE'
            elif i==2:
                sto1=''
            for tau_e in Energy_tariff:
                for vol in Volumetric_share:
                        fincen = pd.read_csv('output/FinCen' + sto + '_' + tau_e + '_' + vol + cmg +'_P90_'+ id +'_Renew.csv')
                        findec = pd.read_csv('output/FinDec' + sto + '_' + tau_e + '_' + vol + cmg +'_P90_'+ id +'_Renew.csv')
                        fincen=fincen.add_suffix('_c')
                        findec=findec.add_suffix('_d')
                        fincen.rename(columns={'Unnamed: 0_c': 'Cliente','As_c':'As'}, inplace=True)
                        findec.rename(columns={'Unnamed: 0_d': 'Cliente','As_d':'As'}, inplace=True)
                        Finan = pd.merge(fincen, findec, on=['Cliente', 'As'], how='inner')
                        Finan['Total_cost_rate'] = Finan['Total [$]_d'] / Finan['Total [$]_c']
                        Finan['Total_grid_cost_rate'] = Finan['Grid_buys_distribution [$]_d']/Finan['Grid_buys_distribution [$]_c']
                        Finan['case']= sto1 + '_' + tau_e + '_' + vol + cmg
                        Res = pd.read_csv('output/Resultados' + sto + '_' + tau_e + '_' + vol + cmg +'_P90_'+ id +'_Renew.csv')
                        Res['case'] = sto1 + '_' + tau_e + '_' + vol + cmg
                        Finan = pd.merge(Finan,Res, on=['As','case'], how= 'inner')
                        DF.append(Finan)

A=pd.concat(DF)
# Grouping by 'case' and 'Zone_c' and calculating the average total cost for each group

As_selected=150000


######################### FIRST CHART
B = A.groupby(['case', 'Zone_c']).apply(lambda x: x[x['As'] == As_selected]['Total [$]_d'].mean()).unstack()

# If you want to reset the index and have 'case' as a regular column instead of index
B.reset_index(inplace=True)

B = B.set_index('case').reindex(Case_order).reset_index()
index_ordered=pd.MultiIndex.from_tuples(Case_format,names=['Technology','Distribution tariff', 'Energy tariff'])
B=B.set_index(index_ordered)
# Set the figure size

No_DER=425.450970874954

fig,ax= plt.subplots(2,3,layout='constrained',figsize=(8.3/2.54, 10/2.54))
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 8

# Define the width of each bar
bar_width = 0.2

# Define the x positions for the bars
x = np.arange(len(B[0:9]))

# Define the desired order of zones for the legend
zone_order = [2, 1,3, 4]

B_sorted = B[['case'] + zone_order][0:9]

# Define the colormap
colormap = plt.get_cmap('Set2', len(B.columns[1:]))
colormap = colormap(np.linspace(0, 1, len(B.columns[1:])))
colormap = ["#c6e6ff","#d9d9d9","#ffcccc","#f87c7c"]
# colormap[len(B.columns[1:])-1]=[0.122312, 0.633153, 0.530398, 1.]

# ax1.set_axisbelow(True)
# ax1.yaxis.grid(color='gray', linestyle='dashed')


# Iterate through each zone and plot bars separately
for t in range(0,3):
    x=np.arange(t*3, (t+1)*3)
    x_base=np.arange(3)
    for i, zone in enumerate(zone_order):
        ax[0,t].bar(x_base + i * bar_width, B_sorted[zone][t*3: (t+1)*3], width=bar_width, label=zone, color=colormap[i])
        ax[0,t].set_yticklabels(ax[0,t].get_yticklabels(), fontsize=8, family='Times New Roman')
        ax[0,t].set_xticks(x_base + 0.3, [f"{case[2]}" for case in B_sorted[t*3: (t+1)*3].index], rotation=0, ha='center',
                       family='Times New Roman', size=8)
        ax[0,t].axhline(y=No_DER, color='green', linestyle='--', linewidth=1)
        ax[0, t].set_axisbelow(True)
        # ax[0, t].yaxis.grid(color='gray', linestyle='dashed')
        ax[0, t].set_ylim(0, 480)
#ax[0, 0].set_ylabel('Average Total Cost \n[$-year]', family='Times New Roman', size=8)
# # Set labels and title
# ax1.set_ylabel('Average Total Cost \n[$-year]', family='Times New Roman', size=8)
# ax1.set_xticks(x+0.3, [f"{case[2]}" for case in B_sorted.index ], rotation=0, ha='center', family='Times New Roman', size= 8)
# ax1.set_yticklabels(ax1.get_yticklabels(),fontsize=8, family='Times New Roman')


# # Create secondary x-axis at the top
# ax_top = ax1.twiny()
#
# # Set the same limits and ticks as the bottom axis
# ax_top.set_xlim(ax1.get_xlim())
# ax_top.set_xticks([1.3, 4.3, 7.3], labels=['Vol 100', 'Solar PV \nVol 30 Peak 70', 'Vol 10 Peak 90'])
# ax1.axvline(x=2.8, color='black', label='_nolegend_')
# ax1.axvline(x=5.8, color='black', label='_nolegend_')
#
# ax1.axhline(y=No_DER, color='red', linestyle='--', linewidth=1)




B_sorted = B[['case'] + zone_order][9:18]




############# SECOND FIGURE.
# Iterate through each zone and plot bars separately
for t in range(0,3):
    x=np.arange(t*3, (t+1)*3)
    x_base=np.arange(3)
    for i, zone in enumerate(zone_order):
        ax[1,t].bar(x + i * bar_width, B_sorted[zone][t*3: (t+1)*3], width=bar_width, label=zone, color=colormap[i])
        ax[1,t].set_xticks(x + 0.3, [f"{case[2]}" for case in B_sorted[t*3: (t+1)*3].index], rotation=0, ha='center',
                       family='Times New Roman', size=8)
        ax[1, t].set_yticklabels(ax[0, t].get_yticklabels(), fontsize=8, family='Times New Roman')
        ax[1, t].axhline(y=No_DER, color='green', linestyle='--', linewidth=1)
        ax[1, t].set_axisbelow(True)
        # ax[1, t].yaxis.grid(color='gray', linestyle='dashed')
        ax[1, t].set_ylim(0,480)


fig,ax= plt.subplots(1,2,layout='constrained',figsize=(8.3/2.54, 5/2.54))

LABEL=['High','Middle','Low','No budget']

for t in range(0,2):
    x = np.arange(t * 3, (t + 1) * 3)
    x_base = np.arange(3)
    ax[t].axhline(y=No_DER, color='green', linestyle='--', linewidth=1, label='No DER installations')
    for i, zone in enumerate(zone_order):
        k=0
        if t==1:
            k=t+1
        ax[t].bar(x + i * bar_width, B_sorted[zone][k*3: (k+1)*3], width=bar_width, label=LABEL[i], color=colormap[i])
        ax[t].set_xticks(x + 0.3, [f"{case[2]}" for case in B_sorted[k*3: (k+1)*3].index], rotation=0, ha='center',
                       family='Times New Roman', size=8)
    ax[t].set_axisbelow(True)
    # ax[t].yaxis.grid(color='gray', linestyle='dashed')
    ax[t].set_ylim(200, 480)
ax[0].set_ylabel('Average total cost [$/year]', family='Times New Roman', size=8)

ax[0].set_title('Vol 100 Peak 0', family='Times New Roman', size=8)
ax[1].set_title('Vol 10 Peak 90', family='Times New Roman', size=8)

lines_labels = [axi.get_legend_handles_labels() for axi in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines[0:5], labels[0:5], loc='lower center', ncol=3, frameon=False)
plt.tight_layout(rect=[0, 0.15, 1, 1])



plt.savefig('output/Total_cost_'+ id +'_2.pdf')

# Set labels and title
#ax[1, 0].set_ylabel('Average Total Cost \n[$-year]', family='Times New Roman', size=8)
# ax2.set_xticks(x+0.3, [f"{case[2]}" for case in B_sorted.index ], rotation=0, ha='center', family='Times New Roman', size= 8)
# ax2.set_yticklabels(ax1.get_yticklabels(),fontsize=8, family='Times New Roman')

# Create secondary x-axis at the top
# ax_top = ax2.twiny()
#
# # Set the same limits and ticks as the bottom axis
# ax_top.set_xlim(ax2.get_xlim())
# ax_top.set_xticks([1.3, 4.3, 7.3], labels=['Vol 100', 'Solar PV and Storage \nVol 30 Peak 70', 'Vol 10 Peak 90'])
# ax2.axvline(x=2.8, color='black', label='_nolegend_')
# ax2.axvline(x=5.8, color='black', label='_nolegend_')
#
# ax2.axhline(y=No_DER, color='red', linestyle='--', linewidth=1)
# ax2.set_axisbelow(True)
# ax2.yaxis.grid(color='gray', linestyle='dashed')
#
#
# ax2.legend(['No DER','High', 'Medium', 'Low', 'None'], loc='lower center', bbox_to_anchor=(0.5, 0), ncol=3)
#plt.tight_layout()
#plt.savefig('output/Total_cost_'+str(As_selected)+'.svg')
#plt.show()


###################################Second chart


B = A.groupby(['case', 'Zone_c']).apply(lambda x: x[x['As'] == As_selected]['Grid_buys_distribution [$]_d'].mean()).unstack()

# If you want to reset the index and have 'case' as a regular column instead of index
B.reset_index(inplace=True)

B = B.set_index('case').reindex(Case_order).reset_index()
index_ordered=pd.MultiIndex.from_tuples(Case_format,names=['Technology','Distribution tariff', 'Energy tariff'])
B=B.set_index(index_ordered)
# Set the figure size

No_DER=204.930525853588

fig,ax= plt.subplots(2,3,layout='constrained',figsize=(8.3/2.54, 10/2.54))
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 8

# Define the width of each bar
bar_width = 0.2

# Define the x positions for the bars
x = np.arange(len(B[0:9]))

# Define the desired order of zones for the legend
zone_order = [2, 1,3, 4]

B_sorted = B[['case'] + zone_order][0:9]


# Iterate through each zone and plot bars separately
for t in range(0,3):
    x=np.arange(t*3, (t+1)*3)
    x_base=np.arange(3)
    for i, zone in enumerate(zone_order):
        ax[0,t].bar(x_base + i * bar_width, B_sorted[zone][t*3: (t+1)*3], width=bar_width, label=zone, color=colormap[i])
        ax[0,t].set_xticks(x_base + 0.3, [f"{case[2]}" for case in B_sorted[t*3: (t+1)*3].index], rotation=0, ha='center',
                       family='Times New Roman', size=8)
        ax[0,t].axhline(y=No_DER, color='green', linestyle='--', linewidth=1)
        ax[0, t].set_axisbelow(True)
        # ax[0, t].yaxis.grid(color='gray', linestyle='dashed')
        ax[0, t].set_ylim(0, 280)
        ax[0, t].set_yticklabels(ax[0, t].get_yticklabels(), fontsize=8, family='Times New Roman')

# # Set labels and title
# ax1.set_ylabel('Average Distribution Cost \n[$-year]', family='Times New Roman', size=8)
# ax1.set_xticks(x+0.3, [f"{case[2]}" for case in B_sorted.index ], rotation=0, ha='center', family='Times New Roman', size= 8)
# ax1.set_yticklabels(ax1.get_yticklabels(),fontsize=8, family='Times New Roman')
#
#
# sec_1= ax1.secondary_xaxis(location=-0.0)
# sec_1.set_xticks([1.3, 4.3, 7.3,10.3, 13.3, 16.3], labels=['\n\nVol 100', '\n\nVol 30 Peak 70', '\n\nVol 10 Peak 90','\n\nVol 100', '\n\nVol 30 Peak 70', '\n\nVol 10 Peak 90'])
# sec_1.tick_params('x', length=0)
#
# sec2_1 = ax1.secondary_xaxis(location=0)
# sec2_1.set_xticks([2.8, 5.8, 8.8, 11.8, 14.8, 17.8], labels=[])
# sec2_1.tick_params('x', length=25, width=1)
#
# sec3_1 = ax1.secondary_xaxis(location=0)
# sec3_1.set_xticks([4.3, 13.3], labels=['\n\n\nSolar PV', '\n\n\nSolar PV + Storage'])
# sec3_1.tick_params('x', length=0)
#
# sec4_1 = ax1.secondary_xaxis(location=0)
# sec4_1.set_xticks([-0.2,8.8,17.8], labels=[])
# sec4_1.tick_params('x', length=35, width=1)
# ax1.axhline(y=No_DER, color='red', linestyle='--', linewidth=1)
# ax1.grid(axis='y', linestyle='--')

B_sorted = B[['case'] + zone_order][9:18]


# Iterate through each zone and plot bars separately
# Iterate through each zone and plot bars separately
for t in range(0,3):
    x=np.arange(t*3, (t+1)*3)
    x_base=np.arange(3)
    for i, zone in enumerate(zone_order):
        ax[1,t].bar(x + i * bar_width, B_sorted[zone][t*3: (t+1)*3], width=bar_width, label=zone, color=colormap[i])
        ax[1,t].set_xticks(x + 0.3, [f"{case[2]}" for case in B_sorted[t*3: (t+1)*3].index], rotation=0, ha='center',
                       family='Times New Roman', size=8)
        ax[1, t].set_yticklabels(ax[0, t].get_yticklabels(), fontsize=8, family='Times New Roman')
        ax[1, t].axhline(y=No_DER, color='green', linestyle='--', linewidth=1)
        ax[1, t].set_axisbelow(True)
        # ax[1, t].yaxis.grid(color='gray', linestyle='dashed')
        ax[1, t].set_ylim(0,280)

# Set labels and title
# ax2.set_ylabel('Average Distribution Cost \n[$-year]', family='Times New Roman', size=8)
# ax2.set_xticks(x+0.3, [f"{case[2]}" for case in B_sorted.index ], rotation=0, ha='center', family='Times New Roman', size= 8)
# ax2.set_yticklabels(ax1.get_yticklabels(),fontsize=8, family='Times New Roman')
#
# sec_2= ax2.secondary_xaxis(location=-0.0)
# sec_2.set_xticks([1.3, 4.3, 7.3,10.3, 13.3, 16.3], labels=['\n\nVol 100', '\n\nVol 30 Peak 70', '\n\nVol 10 Peak 90','\n\nVol 100', '\n\nVol 30 Peak 70', '\n\nVol 10 Peak 90'])
# sec_2.tick_params('x', length=0)
#
# sec2_2 = ax2.secondary_xaxis(location=0)
# sec2_2.set_xticks([2.8, 5.8, 8.8, 11.8, 14.8, 17.8], labels=[])
# sec2_2.tick_params('x', length=25, width=1)
#
# sec3_2 = ax2.secondary_xaxis(location=0)
# sec3_2.set_xticks([4.3], labels=['\n\n\nSolar PV + Storage'])
# sec3_2.tick_params('x', length=0)
#
# sec4_2 = ax2.secondary_xaxis(location=0)
# sec4_2.set_xticks([-0.2,8.8,17.8], labels=[])
# sec4_2.tick_params('x', length=35, width=1)
# ax2.grid(axis='y', linestyle='--')
# ax2.axhline(y=No_DER, color='red', linestyle='--', linewidth=1,label='No DER cost')
#
#
# ax1.legend(['No DER cost','High', 'Medium', 'Low', 'None'], loc='lower right')
# plt.savefig('output/Total_dist_cost_'+str(As_selected)+'.svg')
# plt.show()

fig,ax= plt.subplots(1,2,layout='constrained',figsize=(8.3/2.54, 5/2.54))

LABEL=['High','Middle','Low','No budget']

for t in range(0,2):
    x = np.arange(t * 3, (t + 1) * 3)
    x_base = np.arange(3)
    ax[t].axhline(y=No_DER, color='green', linestyle='--', linewidth=1, label='No DER installations')
    for i, zone in enumerate(zone_order):
        k=0
        if t==1:
            k=t+1
        ax[t].bar(x + i * bar_width, B_sorted[zone][k*3: (k+1)*3], width=bar_width, label=LABEL[i], color=colormap[i])
        ax[t].set_xticks(x + 0.3, [f"{case[2]}" for case in B_sorted[k*3: (k+1)*3].index], rotation=0, ha='center',
                       family='Times New Roman', size=8)
    ax[t].set_axisbelow(True)
    # ax[t].yaxis.grid(color='gray', linestyle='dashed')
    ax[t].set_ylim(0, 260)
ax[0].set_ylabel('Average distribution cost [$/year]', family='Times New Roman', size=8)

ax[0].set_title('Vol 100 Peak 0', family='Times New Roman', size=8)
ax[1].set_title('Vol 10 Peak 90', family='Times New Roman', size=8)

lines_labels = [axi.get_legend_handles_labels() for axi in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines[0:5], labels[0:5], loc='lower center', ncol=3, frameon=False)
plt.tight_layout(rect=[0, 0.15, 1, 1])
plt.savefig('output/Total_dist_cost'+ id +'_2.pdf')


#plt.savefig('output/Total_cost_1.pdf')


Crest_demand=np.array([7.80853E-05,
0.000126839,
0.000123465,
0.000191253,
0.000196036,
0.000132388,
0.000178468,
0.000288236,
0.000250504,
0.000449117,
0.000359069,
0.000265768,
0.000398787,
0.000466953,
0.000435096,
0.000329395,
0.000317035,
0.000200711,
0.000510281,
0.000339968,
0.000800664,
0.000660489,
0.000279823,
0.000174675,
0.000112532,
0.000100797,
0.000107568,
0.000100715,
0.00011255,
0.000107907,
0.00014839,
0.000534498,
0.000664622,
0.000344013,
0.00044722,
0.000411487,
0.00071534,
0.000324473,
0.000136875,
0.000142833,
0.00034092,
0.000820292,
0.00092828,
0.000605142,
0.000743257,
0.000719082,
0.000493817,
0.000397857,
0.000168798,
0.000151195,
0.000161353,
0.000151073,
0.000168825,
0.00016186,
0.000222585,
0.000801748,
0.000996933,
0.00051602,
0.00067083,
0.00061723,
0.00107301,
0.00048671,
0.000205313,
0.00021425,
0.00051138,
0.001230438,
0.00139242,
0.000907713,
0.001114885,
0.001078623,
0.000740725,
0.000596785])

fig,ax= plt.subplots(1,3, layout='constrained',figsize=(8.3/2.54, 5/2.54))
Time=[i for j in range(3) for i in range(24)]

ax[0].plot(Crest_demand[0:24]*1000000)
ax[1].plot(Crest_demand[24:48]*1000000)
ax[2].plot(Crest_demand[48:72]*1000000)
ax[0].set_ylabel('Consumer active energy demand \n [Wh]', family='Times New Roman', size=8)
ax[0].set_ylim(0,1450)
ax[1].set_ylim(0,1450)
ax[2].set_ylim(0,1450)

ax[0].set_title('Summer day')
ax[1].set_title('Winter day')
ax[2].set_title('Peak day')


# ax.set_xticks([0,6,12,18,24,30,36,42,48,54,60,66,72],["",6,12,18,"",6,12,18,"",6,12,18,""])
#
# sec_2= ax.secondary_xaxis(location=-0.0)
# sec_2.set_xticks([12, 36, 60], labels=['\n\nSummer', '\n\nWinter', '\n\nPeak day'])
# sec_2.tick_params('x', length=0)
#
# sec2_2 = ax.secondary_xaxis(location=0)
# sec2_2.set_xticks([0,24,48,72], labels=[])
# sec2_2.tick_params('x', length=25, width=1)

#plt.savefig('output/CREST_demand.pdf')


Total_costs=[
10443.69016,
10442.95692,
10442.96426,
10459.25135,
10454.24774,
10454.24973,
10490.70012,
10485.18171,
10487.84716,
10442.91933,
10442.9318,
10443.71506,
10418.23242,
10451.30868,
10451.84347,
10394.37071,
10453.89431,
10465.41389,
]

Cent_PV= 10442.95692
NO_DER_costs=10728.49455

fig,(ax1, ax2)= plt.subplots(2,1,layout='constrained',figsize=(8.3/2.54, 10/2.54))

ax1.bar(range(len(Total_costs[0:9])),np.array(Total_costs[0:9])/1000)
ax1.set_ylim(10.2, 10.8)

# Set labels and title
ax1.set_ylabel('Total system costs [k$-year]', family='Times New Roman', size=8)
ax1.set_xticks(range(9), [f"{case[2]}" for case in B_sorted.index ], rotation=0, ha='center', family='Times New Roman', size= 8)
ax1.set_yticklabels(ax1.get_yticklabels(),fontsize=8, family='Times New Roman')


sec_1= ax1.secondary_xaxis(location=-0.0)
sec_1.set_xticks([1, 4, 7], labels=['\n\nVol 100', '\n\nVol 30 Peak 70', '\n\nVol 10 Peak 90'])
sec_1.tick_params('x', length=0)

sec2_1 = ax1.secondary_xaxis(location=0)
sec2_1.set_xticks([-0.5,2.5,5.5,8.5], labels=[])
sec2_1.tick_params('x', length=25, width=1)

sec3_1 = ax1.secondary_xaxis(location=0)
sec3_1.set_xticks([3.5], labels=['\n\n\nSolar PV'])
sec3_1.tick_params('x', length=0)

ax1.axhline(y=Cent_PV/1000, color='red', linestyle='--', linewidth=1)
ax1.axhline(y=NO_DER_costs/1000, color='green', linestyle='--', linewidth=1)
ax1.legend(['CPM cost', 'No DER installations'], loc='lower right', frameon=False)

#ax1.grid(axis='y', linestyle='--')

####################

Cent_PV_Sto= 10247.27553

ax2.bar(range(len(Total_costs[9:18])),np.array(Total_costs[9:18])/1000)
ax2.set_ylim(10.2, 10.8)

# Set labels and title
ax2.set_ylabel('Total system costs [k$-year]', family='Times New Roman', size=8)
ax2.set_xticks(range(9), [f"{case[2]}" for case in B_sorted.index ], rotation=0, ha='center', family='Times New Roman', size= 8)
ax2.set_yticklabels(ax1.get_yticklabels(),fontsize=8, family='Times New Roman')


sec_2= ax2.secondary_xaxis(location=-0.0)
sec_2.set_xticks([1, 4, 7], labels=['\n\nVol 100', '\n\nVol 30 Peak 70', '\n\nVol 10 Peak 90'])
sec_2.tick_params('x', length=0)

sec2_2 = ax2.secondary_xaxis(location=0)
sec2_2.set_xticks([-0.5,2.5,5.5,8.5], labels=[])
sec2_2.tick_params('x', length=25, width=1)

sec3_2 = ax2.secondary_xaxis(location=0)
sec3_2.set_xticks([3.5], labels=['\n\n\nSolar PV + Storage'])
sec3_2.tick_params('x', length=0)

ax2.axhline(y=Cent_PV_Sto/1000, color='red', linestyle='--', linewidth=1)
ax2.axhline(y=NO_DER_costs/1000, color='green', linestyle='--', linewidth=1)
#ax2.grid(axis='y', linestyle='--')

fig,ax = plt.subplots(1,2,layout='constrained',figsize=(8.3/2.54, 5/2.54))
colo=plt.get_cmap('tab20')

ax[0].bar(range(len(Total_costs[9:12])),np.array(Total_costs[9:12])/1000, label='DIM costs',color=colo(1))
ax[1].bar(range(len(Total_costs[15:18])),np.array(Total_costs[15:18])/1000,color=colo(1))

ax[0].set_ylim(10.2, 10.8)
ax[1].set_ylim(10.2, 10.8)



ax[0].set_xticks(range(3), [f"{case[2]}" for case in B_sorted.index[0:3] ], rotation=0, ha='center', family='Times New Roman', size= 8)
ax[1].set_xticks(range(3), [f"{case[2]}" for case in B_sorted.index[0:3]], rotation=0, ha='center', family='Times New Roman', size= 8)

ax[0].axhline(y=Cent_PV_Sto/1000, color='red', linestyle='--', linewidth=1, label='CPM costs')
ax[0].axhline(y=NO_DER_costs/1000, color='green', linestyle='--', linewidth=1, label='No DER installations')
ax[1].axhline(y=Cent_PV_Sto/1000, color='red', linestyle='--', linewidth=1)
ax[1].axhline(y=NO_DER_costs/1000, color='green', linestyle='--', linewidth=1)


ax[0].set_title('Vol 100 Peak 0', family='Times New Roman', size=8)
ax[1].set_title('Vol 10 Peak 90', family='Times New Roman', size=8)

ax[0].set_axisbelow(True)
#ax[0].yaxis.grid(color='gray', linestyle='dashed')
ax[1].set_axisbelow(True)
#ax[1].yaxis.grid(color='gray', linestyle='dashed')

ax[0].set_ylabel('Total system-level costs [k$-year]', family='Times New Roman', size=8)

lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels, loc='lower center', ncol=4, frameon=False)
plt.tight_layout(rect=[0, 0.1, 1, 1])

plt.savefig('output/Total_Sys_Cost_Term_03.pdf')

cmap = plt.get_cmap('tab20')

Storage=np.array([
606.0722652,
0,
0,
0,
221.4381488,
86.28821616,
87.85056034,
731.1994405,
123.4953118,
105.498654
])

Solar_PV=np.array([
143.9277348,
750,
750,
747.8717157,
528.5618512,
663.7117838,
662.1494397,
18.80055954,
626.5046882,
644.501346,
])
fig, ax = plt.subplots(1, 3,figsize=(8.3/2.54, 5/2.54), gridspec_kw={'width_ratios': [1, 3, 3]})

ax[0].bar(['CPM'],Solar_PV[0], label="Solar PV",color=cmap(1))
ax[0].bar(['CPM'],Storage[0],bottom=Solar_PV[0],label="Storage",color=cmap(3))

ax[1].bar(range(3),Solar_PV[1:4], label="Solar PV",color=cmap(1))
ax[1].bar(range(3),Storage[1:4],bottom=Solar_PV[1:4],label="Storage",color=cmap(3))

ax[2].bar(range(3),Solar_PV[7:10], label="Solar PV",color=cmap(1))
ax[2].bar(range(3),Storage[7:10],bottom=Solar_PV[7:10],label="Storage",color=cmap(3))

ax[0].set_ylim(0, 800)
ax[1].set_ylim(0, 800)
ax[2].set_ylim(0, 800)

ax[1].set_title('Vol 100 Peak 0', family='Times New Roman', size=8)
ax[2].set_title('Vol 10 Peak 90', family='Times New Roman', size=8)

ax[0].set_ylabel('Annual system DER investment \n [$-year]', family='Times New Roman', size=8)

ax[1].set_xticks(range(3), [f"{case[2]}" for case in B_sorted.index[0:3] ], rotation=0, ha='center', family='Times New Roman', size= 8,)
ax[2].set_xticks(range(3), [f"{case[2]}" for case in B_sorted.index[0:3]], rotation=0, ha='center', family='Times New Roman', size= 8)

lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines[0:2], labels[0:2], loc='lower center', ncol=2, frameon=False)
plt.tight_layout(rect=[0, 0.1, 1, 1])

plt.savefig('output/DER_investment_2.pdf')


MgC=[
60.5,
55.4,
55,
55,
52.1,
53.8,
65.3,
84.4,
82.9,
75,
65.9,
59.6,
56.5,
54.7,
46.3,
45,
51.3,
61.9,
78,
88.6,
89.7,
88.8,
77.5,
64
]


plt.figure(layout='constrained',figsize=(8.3/2.54, 5/2.54))
plt.plot(MgC)

plt.ylabel('Marginal cost of energy [$/MWh]', family='Times New Roman', size=8)
plt.ylim(0,170)
plt.tight_layout()
#plt.savefig('output/MgC_thermal.pdf')

MgC=[
72.2728323,
70.28576903,
67.41947422,
59.02281305,
58.68870506,
63.54206315,
72.2728323,
74.215934,
34.81756885,
0,
0,
0,
0,
0,
0,
0,
37.38492493,
130.504337,
144.9940727,
120.5602283,
116.2432014,
116.2432014,
112.6383521,
94.61410565
]


plt.figure(layout='constrained',figsize=(8.3/2.54, 5/2.54))
plt.plot(MgC)

plt.ylabel('Marginal cost of energy [$/MWh]', family='Times New Roman', size=8)
plt.ylim(0,170)
plt.tight_layout()
# plt.savefig('output/MgC_renewable.pdf')

Sol_Ava=[
0,
0,
0,
0,
0.035820896,
0.214925373,
0.417910448,
0.620895522,
0.8,
1,
1,
1,
0.919402985,
0.967164179,
0.857761194,
0.72641791,
0.545373134,
0.32238806,
0.151343284,
0.04,
0,
0,
0,
0
]


plt.figure(layout='constrained',figsize=(8.3/2.54, 5/2.54))
plt.plot(Sol_Ava)

plt.ylabel('Solar PV generation rate  [p.u.]', family='Times New Roman', size=8)
plt.ylim(0,1.1)
plt.tight_layout()
#plt.savefig('output/Sol_ava.pdf')



############

Total_costs=[
10920.98579,
10920.93621,
10920.93624,
9962.279715,
10788.68447,
10795.59257,
9489.484601,
10767.18232,
10775.98585,
10920.98579,
10920.93621,
10920.93624,
9962.279715,
10788.68447,
10795.59257,
9489.484601,
10767.18232,
10775.98585
]

Cent_PV= 9295.832646
NO_DER_costs=10581.75742

fig,(ax1, ax2)= plt.subplots(2,1,layout='constrained',figsize=(8.3/2.54, 10/2.54))

ax1.bar(range(len(Total_costs[0:9])),np.array(Total_costs[0:9])/1000)
ax1.set_ylim(10.2, 10.8)

# Set labels and title
ax1.set_ylabel('Total system costs [k$-year]', family='Times New Roman', size=8)
ax1.set_xticks(range(9), [f"{case[2]}" for case in B_sorted.index ], rotation=0, ha='center', family='Times New Roman', size= 8)
ax1.set_yticklabels(ax1.get_yticklabels(),fontsize=8, family='Times New Roman')


sec_1= ax1.secondary_xaxis(location=-0.0)
sec_1.set_xticks([1, 4, 7], labels=['\n\nVol 100', '\n\nVol 30 Peak 70', '\n\nVol 10 Peak 90'])
sec_1.tick_params('x', length=0)

sec2_1 = ax1.secondary_xaxis(location=0)
sec2_1.set_xticks([-0.5,2.5,5.5,8.5], labels=[])
sec2_1.tick_params('x', length=25, width=1)

sec3_1 = ax1.secondary_xaxis(location=0)
sec3_1.set_xticks([3.5], labels=['\n\n\nSolar PV'])
sec3_1.tick_params('x', length=0)

ax1.axhline(y=Cent_PV/1000, color='red', linestyle='--', linewidth=1)
ax1.axhline(y=NO_DER_costs/1000, color='green', linestyle='--', linewidth=1)
ax1.legend(['CPM cost', 'No DER installations'], loc='lower right', frameon=False)

#ax1.grid(axis='y', linestyle='--')

####################

Cent_PV_Sto= 9295.832646

ax2.bar(range(len(Total_costs[9:18])),np.array(Total_costs[9:18])/1000)
ax2.set_ylim(10.2, 10.8)

# Set labels and title
ax2.set_ylabel('Total system costs [k$-year]', family='Times New Roman', size=8)
ax2.set_xticks(range(9), [f"{case[2]}" for case in B_sorted.index ], rotation=0, ha='center', family='Times New Roman', size= 8)
ax2.set_yticklabels(ax1.get_yticklabels(),fontsize=8, family='Times New Roman')


sec_2= ax2.secondary_xaxis(location=-0.0)
sec_2.set_xticks([1, 4, 7], labels=['\n\nVol 100', '\n\nVol 30 Peak 70', '\n\nVol 10 Peak 90'])
sec_2.tick_params('x', length=0)

sec2_2 = ax2.secondary_xaxis(location=0)
sec2_2.set_xticks([-0.5,2.5,5.5,8.5], labels=[])
sec2_2.tick_params('x', length=25, width=1)

sec3_2 = ax2.secondary_xaxis(location=0)
sec3_2.set_xticks([3.5], labels=['\n\n\nSolar PV + Storage'])
sec3_2.tick_params('x', length=0)

ax2.axhline(y=Cent_PV_Sto/1000, color='red', linestyle='--', linewidth=1)
ax2.axhline(y=NO_DER_costs/1000, color='green', linestyle='--', linewidth=1)
#ax2.grid(axis='y', linestyle='--')

fig,ax = plt.subplots(1,2,layout='constrained',figsize=(8.3/2.54, 5/2.54))

ax[0].bar(range(len(Total_costs[9:12])),np.array(Total_costs[9:12])/1000, label='DIM costs')
ax[1].bar(range(len(Total_costs[15:18])),np.array(Total_costs[15:18])/1000)

ax[0].set_ylim(9, 11)
ax[1].set_ylim(9, 11)


ax[0].set_xticks(range(3), [f"{case[2]}" for case in B_sorted.index[0:3] ], rotation=0, ha='center', family='Times New Roman', size= 8,)
ax[1].set_xticks(range(3), [f"{case[2]}" for case in B_sorted.index[0:3]], rotation=0, ha='center', family='Times New Roman', size= 8)

ax[0].axhline(y=Cent_PV_Sto/1000, color='red', linestyle='--', linewidth=1, label='CPM costs')
ax[0].axhline(y=NO_DER_costs/1000, color='green', linestyle='--', linewidth=1, label='No DER installations')
ax[1].axhline(y=Cent_PV_Sto/1000, color='red', linestyle='--', linewidth=1)
ax[1].axhline(y=NO_DER_costs/1000, color='green', linestyle='--', linewidth=1)


ax[0].set_title('Vol 100 Peak 0', family='Times New Roman', size=8)
ax[1].set_title('Vol 10 Peak 90', family='Times New Roman', size=8)

ax[0].set_axisbelow(True)
#ax[0].yaxis.grid(color='gray', linestyle='dashed')
ax[1].set_axisbelow(True)
#ax[1].yaxis.grid(color='gray', linestyle='dashed')

ax[0].set_ylabel('Total system-level costs [k$-year]', family='Times New Roman', size=8)

lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels, loc='lower center', ncol=4, frameon=False)
plt.tight_layout(rect=[0, 0.1, 1, 1])

plt.savefig('output/Total_Sys_Cost_1007.pdf')

Storage=np.array([
750,
0,
0,
0,
749.9517395,
86.28821616,
86.31546391,
750,
105.7379098,
105.6360189
])

Solar_PV=np.array([
0,
750,
750,
750,
0.048260529,
663.7117838,
663.6845361,
0,
644.2620902,
644.3639811
])
fig, ax = plt.subplots(1, 3,figsize=(8.3/2.54, 5/2.54), gridspec_kw={'width_ratios': [1, 3, 3]})

ax[0].bar(['CPM'],Solar_PV[0], label="Solar PV")
ax[0].bar(['CPM'],Storage[0],bottom=Solar_PV[0],label="Storage")

ax[1].bar(range(3),Solar_PV[1:4], label="Solar PV")
ax[1].bar(range(3),Storage[1:4],bottom=Solar_PV[1:4],label="Storage")

ax[2].bar(range(3),Solar_PV[7:10], label="Solar PV")
ax[2].bar(range(3),Storage[7:10],bottom=Solar_PV[7:10],label="Storage")

ax[0].set_ylim(0, 800)
ax[1].set_ylim(0, 800)
ax[2].set_ylim(0, 800)

ax[1].set_title('Vol 100 Peak 0', family='Times New Roman', size=8)
ax[2].set_title('Vol 10 Peak 90', family='Times New Roman', size=8)

ax[0].set_ylabel('Annual system DER investment \n [$-year]', family='Times New Roman', size=8)

ax[1].set_xticks(range(3), [f"{case[2]}" for case in B_sorted.index[0:3] ], rotation=0, ha='center', family='Times New Roman', size= 8,)
ax[2].set_xticks(range(3), [f"{case[2]}" for case in B_sorted.index[0:3]], rotation=0, ha='center', family='Times New Roman', size= 8)

lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines[0:2], labels[0:2], loc='lower center', ncol=2, frameon=False)
plt.tight_layout(rect=[0, 0.1, 1, 1])

plt.savefig('output/DER_investment_1007.pdf')


