import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FuncFormatter
##Script to create images from the results obtained.

D1=pd.read_csv('output/Tariff.csv')
D2=pd.read_csv('output/Tariff_1207.csv')

D1[0:48]=D1[0:48]/182
D2[0:48]=D2[0:48]/182
LCOE=150000/(0.4*8760)
EN=[ 'Cmg', 'UKE7', 'FLAT']
ENE_Label=['MgC', '2-b', 'Flat']
DIS =["VOL100_PEAK0", "VOL30_PEAK70", "VOL10_PEAK90"]
DIS_label= [r"$\bf{Vol100\ Peak0}$",r"$\bf{Vol30\ Peak70}$", r"$\bf{Vol10\ Peak90}$"]
cmap = plt.get_cmap('tab20')
Col=[cmap(1), cmap(3), cmap(5)]


fig, axes = plt.subplots(2, 3,layout='constrained',figsize=((8.3*2)/2.54, 10/2.54))
b=0
for TD in DIS:
    j=0
    axes[0, b].axhline(y=LCOE, color='red', linestyle='--', linewidth=1, label='LCOE Solar PV')
    axes[1, b].axhline(y=LCOE, color='red', linestyle='--', linewidth=1, label='LCOE Solar PV')
    axes[0, b].set_xticks([0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72], ["", 6, 12, 18, "", 6, 12, 18, "", 6, 12, 18, ""])
    axes[1, b].set_xticks([0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72],["", 6, 12, 18, "", 6, 12, 18, "", 6, 12, 18, ""])
    sec_0 = axes[0, b].secondary_xaxis(location=-0.0)
    sec_0.set_xticks([12, 36, 60], labels=['\n\nSummer', '\n\nWinter', '\n\nPeak day'],size=8,family='Times New Roman')
    sec_0.tick_params('x', length=0)
    sec_1 = axes[1, b].secondary_xaxis(location=-0.0)
    sec_1.set_xticks([12, 36, 60], labels=['\n\nSummer', '\n\nWinter', '\n\nPeak day'],size=8, family='Times New Roman')
    sec_1.tick_params('x', length=0)
    sec2_0 = axes[0, b].secondary_xaxis(location=0)
    sec2_0.set_xticks([0, 24, 48, 72], labels=[])
    sec2_0.tick_params('x', length=25, width=1)
    sec2_1 = axes[1, b].secondary_xaxis(location=0)
    sec2_1.set_xticks([0, 24, 48, 72], labels=[])
    sec2_1.tick_params('x', length=25, width=1)
    axes[0,b].set_title(DIS_label[b], size=8)
    if b==0:
        axes[0, b].set_ylabel(r'$\bf{Solar\ PV\ case}$'+ '\n Tariff costs [$/MWh]', family='Times New Roman', size=8)
        axes[1, b].set_ylabel(r'$\bf{Solar\ PV\ and\ storage\ case}$'+'\n Tariff costs [$/MWh]', family='Times New Roman', size=8)
    for TENE in EN:
        axes[0,b].plot(D1['Ene_'+TENE+"_"+TD]+D1['Dist_'+TENE+"_"+TD],label=ENE_Label[j],color=Col[j])
        axes[0,b].set_ylim(0, 180)
        axes[1,b].plot(D2['Ene_'+TENE+"_"+TD]+D2['Dist_'+TENE+"_"+TD],label=ENE_Label[j],color=Col[j])
        axes[1, b].set_ylim(0, 180)
        j=j+1
        if b==2 and j==3:
            axes[0,b].legend(loc='upper left')
        axes[0, b].plot(D1['Ene_' + TENE + "_" + TD]*0.1 + D1['Dist_' + TENE + "_" + TD],color=Col[j-1],linestyle='--')
        axes[1, b].plot(D2['Ene_' + TENE + "_" + TD]*0.1 + D2['Dist_' + TENE + "_" + TD],color=Col[j-1],linestyle='--')
    b=b+1


for ax in axes.flatten():
    # Ajustar la fuente y el tamaño de fuente del título
    ax.title.set_fontsize(8)
    ax.title.set_fontname('Times New Roman')

    # Ajustar la fuente y el tamaño de fuente del eje x
    ax.xaxis.label.set_fontsize(8)
    ax.xaxis.label.set_fontname('Times New Roman')

    # Ajustar la fuente y el tamaño de fuente del eje y
    ax.yaxis.label.set_fontsize(8)
    ax.yaxis.label.set_fontname('Times New Roman')

    # Ajustar la fuente y el tamaño de fuente de los ticks en el eje x
    ax.tick_params(axis='x', labelsize=8, labelfontfamily='Times New Roman')

    # Ajustar la fuente y el tamaño de fuente de los ticks en el eje y
    ax.tick_params(axis='y', labelsize=8, labelfontfamily='Times New Roman')

    # Ajustar la fuente y el tamaño de fuente del texto de la leyenda
    if ax.get_legend():
        ax.legend(prop={'size': 8, 'family': 'Times New Roman'})

#plt.savefig('output/Tariff_chart.pdf')
#plt.show()

fig, ax = plt.subplots(2, 2,layout='constrained',figsize=((8.3)/2.54, (10*2/3)/2.54))

ax[0,0].axhline(y=LCOE, color=cmap(7), linestyle='-.', linewidth=1, label='LCOE Solar PV')
ax[0,1].axhline(y=LCOE, color=cmap(7), linestyle='-.', linewidth=1, label='LCOE Solar PV')
ax[1,0].axhline(y=LCOE, color=cmap(7), linestyle='-.', linewidth=1, label='LCOE Solar PV')
ax[1,1].axhline(y=LCOE, color=cmap(7), linestyle='-.', linewidth=1, label='LCOE Solar PV')
# ax[2,0].axhline(y=LCOE, color='red', linestyle='--', linewidth=1, label='LCOE Solar PV')
# ax[2,1].axhline(y=LCOE, color='red', linestyle='--', linewidth=1, label='LCOE Solar PV')

TD="VOL100_PEAK0"

j=0
for TENE in EN:
    # ax[0, 0].plot(range(24), D2['Ene_'+TENE+"_"+TD][0:24]+D2['Dist_'+TENE+"_"+TD][0:24],label=ENE_Label[j],color=Col[j])
    # ax[0, 0].plot(range(24), D2['Ene_' + TENE + "_" + TD][0:24] * 0.1 + D2['Dist_' + TENE + "_" + TD][0:24], color=Col[j], linestyle='--')
    ax[0, 0].plot(range(24), D2['Ene_' + TENE + "_" + TD][24:48] + D2['Dist_' + TENE + "_" + TD][24:48], label=ENE_Label[j],
                  color=Col[j])
    ax[0, 0].plot(range(24), D2['Ene_' + TENE + "_" + TD][24:48] * 0.1 + D2['Dist_' + TENE + "_" + TD][24:48], color=Col[j],
                  linestyle='--')
    ax[1, 0].plot(range(24),D2['Ene_' + TENE + "_" + TD][48:72] + D2['Dist_' + TENE + "_" + TD][48:72], label=ENE_Label[j],
                  color=Col[j])
    ax[1, 0].plot(range(24), D2['Ene_' + TENE + "_" + TD][48:72] * 0.1 + D2['Dist_' + TENE + "_" + TD][48:72], color=Col[j],
                  linestyle='--')
    ax[0,0].set_xticks([0, 12, 24],[0, 12, 24])
    ax[0, 1].set_xticks([0, 12, 24], [0, 12, 24])
    ax[1, 0].set_xticks([0, 12, 24], [0, 12, 24])
    ax[1, 1].set_xticks([0, 12, 24], [0, 12, 24])
    j=j+1


TD = "VOL10_PEAK90"

j = 0
for TENE in EN:
    # ax[0, 1].plot(range(24), D2['Ene_'+TENE+"_"+TD][0:24]+D2['Dist_'+TENE+"_"+TD][0:24],label=ENE_Label[j],color=Col[j])
    # ax[0, 1].plot(range(24), D2['Ene_' + TENE + "_" + TD][0:24] * 0.1 + D2['Dist_' + TENE + "_" + TD][0:24], color=Col[j], linestyle='--')
    ax[0, 1].plot(range(24), D2['Ene_' + TENE + "_" + TD][24:48] + D2['Dist_' + TENE + "_" + TD][24:48], label=ENE_Label[j],
                  color=Col[j])
    ax[0, 1].plot(range(24), D2['Ene_' + TENE + "_" + TD][24:48] * 0.1 + D2['Dist_' + TENE + "_" + TD][24:48], color=Col[j],
                  linestyle='--')
    ax[1, 1].plot(range(24),D2['Ene_' + TENE + "_" + TD][48:72] + D2['Dist_' + TENE + "_" + TD][48:72], label=ENE_Label[j],
                  color=Col[j])
    ax[1, 1].plot(range(24), D2['Ene_' + TENE + "_" + TD][48:72] * 0.1 + D2['Dist_' + TENE + "_" + TD][48:72], color=Col[j],
                  linestyle='--')
    j=j+1

ax[0,0].set_ylim(0, 250)
ax[0,1].set_ylim(0, 250)
# ax[1,0].set_ylim(0, 180)
# ax[1,1].set_ylim(0, 180)
ax[1,0].set_ylim(0, 250)
ax[0, 0].set_ylabel(r'$\bf{Summer/Winter}$'+ '\n Total tariff [$/MWh]', family='Times New Roman', size=8)
# ax[1, 0].set_ylabel(r'$\bf{Winter}$'+ '\n Tariff costs [$/MWh]', family='Times New Roman', size=8)
ax[1, 0].set_ylabel(r'$\bf{Peak\ day}$'+ '\n Total tariff [$/MWh]', family='Times New Roman', size=8)
ax[0, 0].set_title("Vol 100 Peak 0", size=8)
ax[0, 1].set_title("Vol 10 Peak 90", size=8)
#ax[1, 0].set_xlabel(r'hours', family='Times New Roman', size=8)
#ax[1, 1].set_xlabel(r'hours', family='Times New Roman', size=8)

def thousands(x, pos):
    return '%1.0f' % (x * 1e-3)

ax[1,1].yaxis.set_major_formatter(FuncFormatter(thousands))
ax[1,1].annotate(r'$10^3$', xy=(-0.15, 0.75), xycoords='axes fraction', fontsize=7,
             xytext=(-5, 5), textcoords='offset points',
             ha='left', va='bottom')

for axi in ax.flatten():
    # Ajustar la fuente y el tamaño de fuente del título
    axi.title.set_fontsize(8)
    axi.title.set_fontname('Times New Roman')

    # Ajustar la fuente y el tamaño de fuente del eje x
    axi.xaxis.label.set_fontsize(8)
    axi.xaxis.label.set_fontname('Times New Roman')

    # Ajustar la fuente y el tamaño de fuente del eje y
    axi.yaxis.label.set_fontsize(8)
    axi.yaxis.label.set_fontname('Times New Roman')

    # Ajustar la fuente y el tamaño de fuente de los ticks en el eje x
    axi.tick_params(axis='x', labelsize=8, labelfontfamily='Times New Roman')

    # Ajustar la fuente y el tamaño de fuente de los ticks en el eje y
    axi.tick_params(axis='y', labelsize=8, labelfontfamily='Times New Roman')

    # Ajustar la fuente y el tamaño de fuente del texto de la leyenda
    if axi.get_legend():
        axi.legend(prop={'size': 8, 'family': 'Times New Roman'})


lines_labels = [axi.get_legend_handles_labels() for axi in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines[0:4], labels[0:4], loc='lower center', ncol=4,prop={'size': 8, 'family': 'Times New Roman'}, frameon=False)
plt.tight_layout(rect=[0, 0.05, 1, 1])

plt.savefig('output/Tariff_chart_1207_2.pdf')
#plt.show()