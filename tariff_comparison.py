import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FuncFormatter
import numpy as np
import matplotlib as mpl

Rate_cost_thermal=[
0.339096609,
0.365346066,
0.363524824,
0.156300606,
0.139493794,
0.13965174
]

Over_cost_thermal=[
0.108779451,
0.111371871,
0.1087637,
0.01053175,
0.014292069,
0.011190196
]

Rate_cost_renewable=[
0.221370113,
0.364426277,
0.36199293,
0.433525624,
0.141425243,
0.134485547
]

Over_cost_renewable=[
0.095185979,
0.112175354,
0.095115964,
-0.03805546,
0.014306142,
-0.002749996
]

Rate_dist_thermal=[
0.654332035,
0.654339869,
0.65434298,
0.571166067,
0.25807322,
0.238558287
]

Over_dist_thermal=[
0.225834209,
0.225829915,
0.225833758,
0.021864694,
0.024285044,
0.023263927
]

Rate_dist_renewable=[
0.654331587,
0.654331588,
0.654331592,
0.632822966,
0.238411794,
0.238266732
]

Over_dist_renewable=[
0.225861163,
0.225863535,
0.225863431,
-0.050757947,
0.022679794,
0.022686438
]

Eff_thermal=[
0.019092274,
0.019093492,
0.019169928,
0.014354564,
0.020163289,
0.02128745
]

Eff_renewable=[
0.17482599,
0.174820656,
0.174820659,
0.020832126,
0.158280569,
0.159227609
]

fig, axes = plt.subplots(2, 2,layout='constrained',figsize=((8.3*2)/2.54, 12/2.54))
markers = ['o', 's', 'd']
tag=["MgC", '2-b','Flat']

for i in range(3):
    axes[0,0].scatter(Eff_thermal[i]*100,Rate_cost_thermal[i]*100,marker=markers[i], label='Vol 100 Peak 0 '+tag[i], color='b')
    axes[1, 0].scatter(Eff_thermal[i]*100, Over_cost_thermal[i]*100, marker=markers[i], label='Vol 100 Peak 0 ' + tag[i], color='b')
    axes[0, 1].scatter(Eff_renewable[i]*100, Rate_cost_renewable[i]*100, marker=markers[i], label='Vol 100 Peak 0 ' + tag[i], color='b')
    axes[1, 1].scatter(Eff_renewable[i]*100, Over_cost_renewable[i]*100, marker=markers[i], label='Vol 100 Peak 0 ' + tag[i], color='b')
    axes[0, 0].scatter(Eff_thermal[i+3] * 100, Rate_cost_thermal[i+3] * 100, marker=markers[i],
                    label='Vol 10 Peak 90 ' + tag[i], color='r')
    axes[1, 0].scatter(Eff_thermal[i+3] * 100, Over_cost_thermal[i+3] * 100, marker=markers[i],
                    label='Vol 10 Peak 90 ' + tag[i], color='r')
    axes[0, 1].scatter(Eff_renewable[i+3] * 100, Rate_cost_renewable[i+3] * 100, marker=markers[i],
                    label='Vol 10 Peak 90 ' + tag[i], color='r')
    axes[1, 1].scatter(Eff_renewable[i+3] * 100, Over_cost_renewable[i+3] * 100, marker=markers[i],
                    label='Vol 10 Peak 90 ' + tag[i], color='r')

axes[1,0].set_xlabel("Over cost of equilibrium compared\n with CPM cost [%]", family='Times New Roman', size=9)
axes[1,1].set_xlabel("Over cost of equilibrium compared\n with CPM cost [%]", family='Times New Roman', size=9)
axes[0,0].set_ylabel("Savings of high-budget \n compared with no-budget \n prosumers in terms of \n total costs [%]", family='Times New Roman', size=9)
axes[1,0].set_ylabel("Over cost paid by no-budget\n prosumers compared with the\n no DER costs [%]", family='Times New Roman', size=9)

axes[0,0].set_title("Marginal costs driven by thermal generation", size=10)
axes[0,1].set_title("Marginal costs driven by renewable generation", size=10)

for axi in axes.flatten():
    # Ajustar la fuente y el tamaño de fuente del título
    axi.title.set_fontsize(10)
    axi.title.set_fontname('Times New Roman')

    # Ajustar la fuente y el tamaño de fuente del eje x
    axi.xaxis.label.set_fontsize(10)
    axi.xaxis.label.set_fontname('Times New Roman')

    # Ajustar la fuente y el tamaño de fuente del eje y
    axi.yaxis.label.set_fontsize(10)
    axi.yaxis.label.set_fontname('Times New Roman')

    # Ajustar la fuente y el tamaño de fuente de los ticks en el eje x
    axi.tick_params(axis='x', labelsize=10, labelfontfamily='Times New Roman')

    # Ajustar la fuente y el tamaño de fuente de los ticks en el eje y
    axi.tick_params(axis='y', labelsize=10, labelfontfamily='Times New Roman')

    # Ajustar la fuente y el tamaño de fuente del texto de la leyenda
    if axi.get_legend():
        axi.legend(prop={'size': 10, 'family': 'Times New Roman'})

lines_labels = [axi.get_legend_handles_labels() for axi in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines[0:6], labels[0:6], loc='lower center', ncol=3,prop={'size': 10, 'family': 'Times New Roman'}, frameon=False)
plt.tight_layout(rect=[0, 0.1, 1, 1])

plt.savefig("output/tariff_cost_map.pdf")

fig, axes = plt.subplots(2, 2,layout='constrained',figsize=((8.3*2)/2.54, 12/2.54))
markers = ['o', 's', 'd']
tag=["MgC", '2-b','Flat']

for i in range(3):
    axes[0,0].scatter(Eff_thermal[i]*100,Rate_dist_thermal[i]*100,marker=markers[i], label='Vol 100 Peak 0 '+tag[i], color='b')
    axes[1, 0].scatter(Eff_thermal[i]*100, Over_dist_thermal[i]*100, marker=markers[i], label='Vol 100 Peak 0 ' + tag[i], color='b')
    axes[0, 1].scatter(Eff_renewable[i]*100, Rate_dist_renewable[i]*100, marker=markers[i], label='Vol 100 Peak 0 ' + tag[i], color='b')
    axes[1, 1].scatter(Eff_renewable[i]*100, Over_dist_renewable[i]*100, marker=markers[i], label='Vol 100 Peak 0 ' + tag[i], color='b')
    axes[0, 0].scatter(Eff_thermal[i+3] * 100, Rate_cost_thermal[i+3] * 100, marker=markers[i],
                    label='Vol 10 Peak 90 ' + tag[i], color='r')
    axes[1, 0].scatter(Eff_thermal[i+3] * 100, Over_dist_thermal[i+3] * 100, marker=markers[i],
                    label='Vol 10 Peak 90 ' + tag[i], color='r')
    axes[0, 1].scatter(Eff_renewable[i+3] * 100, Rate_dist_renewable[i+3] * 100, marker=markers[i],
                    label='Vol 10 Peak 90 ' + tag[i], color='r')
    axes[1, 1].scatter(Eff_renewable[i+3] * 100, Over_dist_renewable[i+3] * 100, marker=markers[i],
                    label='Vol 10 Peak 90 ' + tag[i], color='r')

axes[1,0].set_xlabel("Over cost of equilibrium compared\n with CPM cost [%]", family='Times New Roman', size=9)
axes[1,1].set_xlabel("Over cost of equilibrium compared\n with CPM cost [%]", family='Times New Roman', size=9)
axes[0,0].set_ylabel("Savings of high-budget \n compared with no budget \n prosumers  in terms of \n DN cost [%]", family='Times New Roman', size=9)
axes[1,0].set_ylabel("Over DN charges of no-budget\n prosumers compared  with the\n no DER DN costs [%]", family='Times New Roman', size=9)

axes[0,0].set_title("Marginal costs driven by thermal generation", size=10)
axes[0,1].set_title("Marginal costs driven by renewable generation", size=10)

for axi in axes.flatten():
    # Ajustar la fuente y el tamaño de fuente del título
    axi.title.set_fontsize(10)
    axi.title.set_fontname('Times New Roman')

    # Ajustar la fuente y el tamaño de fuente del eje x
    axi.xaxis.label.set_fontsize(10)
    axi.xaxis.label.set_fontname('Times New Roman')

    # Ajustar la fuente y el tamaño de fuente del eje y
    axi.yaxis.label.set_fontsize(10)
    axi.yaxis.label.set_fontname('Times New Roman')

    # Ajustar la fuente y el tamaño de fuente de los ticks en el eje x
    axi.tick_params(axis='x', labelsize=10, labelfontfamily='Times New Roman')

    # Ajustar la fuente y el tamaño de fuente de los ticks en el eje y
    axi.tick_params(axis='y', labelsize=10, labelfontfamily='Times New Roman')

    # Ajustar la fuente y el tamaño de fuente del texto de la leyenda
    if axi.get_legend():
        axi.legend(prop={'size': 10, 'family': 'Times New Roman'})

lines_labels = [axi.get_legend_handles_labels() for axi in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines[0:6], labels[0:6], loc='lower center', ncol=3,prop={'size': 10, 'family': 'Times New Roman'}, frameon=False)
plt.tight_layout(rect=[0, 0.1, 1, 1])

plt.savefig("output/tariff_dist_map.pdf")

