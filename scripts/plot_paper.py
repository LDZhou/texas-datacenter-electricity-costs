"""Original-paper figure entry point; run analysis and REP first."""
import argparse
from pathlib import Path
import subprocess
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
import plot_impacts as rev
import plot_siting as siting
import plot_capacity as plots

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/nc_paper_figures'

def framework(att,graded,metrics):
    folder=ROOT/'results/dc_experiments_full_tx/2023/all2030_full_tx'
    n=pypsa.Network(folder/'network.nc')
    mapping=pd.read_csv(folder/'dc_bus_mapping.csv').merge(n.buses[['x','y']],left_on='bus',right_index=True)
    fig,axes=plt.subplots(1,5,figsize=(15,3.2),layout='constrained')
    axes[0].scatter(n.buses.x,n.buses.y,s=2,c='lightgrey')
    for zone,g in mapping.groupby('zone'):
        axes[0].scatter(g.x,g.y,s=g.capacity_mw*0.03,label=f'{zone}: {g.capacity_mw.sum()/1000:.1f} GW',alpha=.7)
    axes[0].set_aspect(1/np.cos(np.deg2rad(31)));axes[0].set_xlabel('Longitude');axes[0].set_ylabel('Latitude');axes[0].legend(fontsize=5)
    ts=n.loads_t.p_set;dc=n.loads.index[n.loads.index.str.startswith('DC_')]
    original=ts.drop(columns=dc,errors='ignore').sum(axis=1)/1000
    added=ts.reindex(columns=dc).sum(axis=1)/1000
    idx=pd.DatetimeIndex(ts.index.get_level_values(-1) if isinstance(ts.index,pd.MultiIndex) else ts.index)
    peak=idx[int(np.argmax(original))]
    selected=(idx>=peak-pd.Timedelta(days=7))&(idx<peak+pd.Timedelta(days=7))
    axes[1].stackplot(idx[selected],original[selected],added[selected],colors=['lightgrey',plots.C['blue']],labels=['Existing','Data centers'])
    axes[1].set_ylabel('Load (GW)');axes[1].tick_params(axis='x',rotation=60,labelsize=5);axes[1].legend(fontsize=5)
    labels=['Gen +\nstorage','Coordinated'];modes=['generation_storage','full_tx']
    vals=[np.array([metrics[m,y]['total_annual_investment']/1e9 for y in range(2019,2024)]) for m in modes]
    means=np.array([v.mean() for v in vals])
    axes[2].bar(labels,means,yerr=np.array([means-np.array([v.min() for v in vals]),np.array([v.max() for v in vals])-means]),capsize=3)
    axes[2].set_ylabel('Investment (billion $/year)')
    hours=[graded[graded.case==m].snaps_any_bus_over_cap.mean()*3 for m in ['gen_storage','full_tx']]
    axes[3].bar(labels,hours);axes[3].set_ylabel('Hours above $5,000/MWh')
    if min(hours)>0:axes[3].set_yscale('log')
    bottom=0
    for value,label in [(att.loc[2023,'energy'],'Energy'),(att.loc[2023,'congestion'],'Congestion'),(rev.TCOS_RESIDENTIAL,'Transmission')]:
        axes[4].bar(['Residential'],value/10,bottom=bottom,label=label);bottom+=value/10
    axes[4].set_ylabel('Increment (cents/kWh)');axes[4].legend(fontsize=5)
    for ax,letter,title in zip(axes,'abcde',['Locations','Peak load','Investment','Cap hours','Bill channels']):ax.set_title(f'{letter}  {title}',loc='left',fontsize=9)
    plots.save(fig,'fig1_framework')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--skip-validation',action='store_true')
    a=p.parse_args()
    metrics={(m,y):pd.read_csv(ROOT/f'results/dc_experiments_{m}/{y}/all2030_{m}/metrics.csv').iloc[0].to_dict()
             for m in ['generation_storage','full_tx'] for y in range(2019,2024)}
    att=rev.load_attribution();graded=pd.read_csv(OUT/'graded_unserved_all2030.csv')
    framework(att,graded,metrics)
    plots.make_fig5(metrics);plots.make_fig6_fig7(metrics)
    siting.PAPER=OUT;siting.main();rev.main()
    if not a.skip_validation:
        subprocess.run([sys.executable,str(ROOT/'scripts/validate_2021_full_year_network.py')],cwd=ROOT,check=True)

if __name__=='__main__':main()
