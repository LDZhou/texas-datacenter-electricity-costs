"""Small synthetic integration test, independent of private datasets and EIA."""
import json
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pypsa

from dc_common import DC_LOCATIONS, create_dc_load_profile
from prepare_starting_network import prepare_starting_network
from compute_adder_factor_decomposition import decompose
from compute_graded_unserved import summarise

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'smoke'

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'verification.json').unlink(missing_ok=True)
    n = pypsa.Network()
    snapshots = pd.DatetimeIndex([pd.Timestamp(2023,m,1)+pd.Timedelta(hours=h)
                                  for m in range(5,11) for h in range(0,24,3)])
    n.set_snapshots(snapshots)
    n.snapshot_weightings.loc[:,:] = 3.0
    for carrier in ['AC','OCGT','CCGT','solar','onwind','load','datacenter','4hr_battery_storage']:
        n.add('Carrier',carrier)
    buses=[]
    for loc,info in DC_LOCATIONS.items():
        bus=info['bus']; buses.append(bus)
        n.add('Bus',bus,x=info['x'],y=info['y'],v_nom=230,carrier='AC',nerc_reg='ERCOT')
        n.add('Load',f'load_{loc}',bus=bus,p_set=2.5)
        for carrier,mc in [('OCGT',40),('CCGT',25),('solar',0),('onwind',0)]:
            n.add('Generator',f'{loc}_{carrier}',bus=bus,carrier=carrier,p_nom=2 if carrier=='CCGT' else 1,
                  marginal_cost=mc,capital_cost=1000,p_max_pu=0.3 if carrier in ['solar','onwind'] else 1.0)
        n.add('StorageUnit',f'{loc}_battery',bus=bus,carrier='4hr_battery_storage',p_nom=1,
              max_hours=4,capital_cost=1000,cyclic_state_of_charge=True)
        n.add('Generator',f'{loc}_shed',bus=bus,carrier='load',p_nom=1e9,sign=1e-3,marginal_cost=5000)
    for i in range(len(buses)-1):
        n.add('Line',f'line{i}',bus0=buses[i],bus1=buses[i+1],x=0.1,r=0.01,s_nom=50,capital_cost=100)
    n.buses['nerc_reg'] = 'ERCOT'
    from run_dc_experiment import solve
    solve(n, threads=1)
    n.export_to_netcdf(OUT/'input.nc')
    prepare_starting_network(OUT/'input.nc',OUT/'start.nc')
    pd.DataFrame([{'full_address':'Houston, TX','State':'TX','current_mw':10,
                   'construction_mw':0,'planned_mw':10}]).to_excel(OUT/'synthetic_projects.xlsx',index=False)
    # Profile generator must not change the caller's random state.
    np.random.seed(12); expected=np.random.random()
    np.random.seed(12); create_dc_load_profile(snapshots,2); assert np.random.random()==expected
    env=dict(os.environ,PYTHONHASHSEED='0',MPLBACKEND='Agg')
    records=[]
    for scenario in ['none','all2030','multiloc']:
        modes=['dispatch'] if scenario=='none' else ['dispatch','generation','generation_storage','generation_tx','transmission','full_tx']
        for mode in modes:
            args=[sys.executable,'scripts/run_dc_experiment.py','--year','2023','--dc-scenario',scenario,
                  '--mode',mode,'--starting-network',str(OUT/'start.nc'),'--results-root',str(OUT/'runs'),
                  '--threads','1']
            if scenario=='all2030':args+=['--dc-data',str(OUT/'synthetic_projects.xlsx')]
            if scenario=='multiloc':args+=['--location','HOUSTON','--scale','20']
            subprocess.run(args,cwd=ROOT,env=env,check=True)
            name=f'{scenario}_{mode}'+('_HOUSTON_20MW' if scenario=='multiloc' else '')
            folder=OUT/'runs'/'2023'/name
            solved=pypsa.Network(folder/'network.nc')
            assert np.isfinite(solved.objective)
            if mode in ['generation','generation_storage','generation_tx','full_tx']:
                assert solved.generators.loc[solved.generators.carrier=='CCGT','p_nom_extendable'].any()
            if mode in ['generation','generation_storage','generation_tx','full_tx']:
                assert (solved.generators.p_nom_opt-solved.generators.p_nom).clip(lower=0).sum()>0.1
            assert (folder/'metrics.csv').is_file()
            d=decompose(str(folder/'network.nc'),5000)
            assert abs(d['lmp']-d['ref']-d['congestion']-d['scarcity'])<1e-6
            records.append({'case':name,**d,**summarise(str(folder/'network.nc'),5000)})
    pd.DataFrame(records).to_csv(OUT/'summary.csv',index=False)
    subprocess.run([sys.executable,'scripts/analyze_rep_risk.py','--base-dirs',str(OUT/'runs'),
        '--years','2023','--scenarios','none','all2030','--modes','dispatch','full_tx','generation_storage',
        '--output-dir',str(OUT/'rep'),'--cache-path',str(OUT/'rep/cache.pkl'),'--zones','system',
        '--n-sims','10','--seed','123','--lmp-clip-upper','5000','--full-tx-tcos-adder-mwh','2.751319804231755',
        '--refresh-cache','--summary-only','--no-plots'],cwd=ROOT,env=env,check=True)
    assert list((OUT/'rep').glob('rep_default_sim_summary_*.csv'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pd.DataFrame(records).plot.bar(x='case',y='lmp_capped',legend=False)
    plt.tight_layout();plt.savefig(OUT/'smoke_prices.png');plt.close()
    report={'status':'passed','solves':len(records)+1,'pypsa':pypsa.__version__,
            'gurobipy':importlib.metadata.version('gurobipy'),
            'scope':'synthetic preparation, 6 modes, all2030/multiloc, metrics, decomposition, REP, plot; no source-data rebuild'}
    (OUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))

if __name__=='__main__':main()
