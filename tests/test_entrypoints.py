"""Check the public experiment matrix without scheduling or solving jobs."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import dc_common
import run_paper


class EntrypointTests(unittest.TestCase):
    def test_matrix_has_755_unique_cases(self):
        calls=[]
        with tempfile.NamedTemporaryFile(suffix='.xlsx') as data:
            with patch.object(run_paper,'run',lambda *args:calls.append(tuple(map(str,args)))):
                for stage in ['baseline','all2030','multiloc']:
                    for mode in (['dispatch'] if stage=='baseline' else run_paper.MODES):
                        for index in range(120 if stage=='multiloc' else 5):
                            with patch.object(sys,'argv',['run_paper.py',stage,'--mode',mode,'--index',str(index),'--dc-data',data.name]):
                                run_paper.main()
        self.assertEqual(len(calls),755)
        self.assertEqual(len(set(calls)),755)
        for call in calls:
            self.assertIn(call[call.index('--year')+1],['2019','2020','2021','2022','2023'])

    def test_original_carriers(self):
        self.assertEqual(dc_common.EXPANDABLE_CARRIERS['generators'],['CCGT','OCGT','solar','onwind'])
        self.assertIn('CCGT',dc_common.FULL_EXPANDABLE_CARRIERS['generators'])

    def test_rep_uses_paper_parameters(self):
        with patch.object(run_paper,'run') as run, patch.object(sys,'argv',['run_paper.py','rep']):
            run_paper.main()
        args=run.call_args.args
        self.assertEqual(args[args.index('--lmp-clip-upper')+1],5000)
        self.assertEqual(args[args.index('--full-tx-tcos-adder-mwh')+1],2.751319804231755)

    def test_capacity_export_includes_bus_coordinates(self):
        class Network:
            buses = pd.DataFrame({'x':[-97.0],'y':[31.0]},index=['bus-a'])
            generators = pd.DataFrame({
                'p_nom_extendable':[True], 'p_nom_opt':[15.0], 'p_nom':[10.0],
                'capital_cost':[100.0], 'bus':['bus-a'], 'carrier':['OCGT'],
            }, index=['generator-a'])
            storage_units = pd.DataFrame()
            lines = pd.DataFrame()

        result = dc_common.extract_new_capacity(
            Network(),
            {'generators': pd.Series({'generator-a': 10.0}), 'storage_units': pd.Series(dtype=float), 'lines': pd.Series(dtype=float)},
        )

        self.assertEqual(result.loc[0, 'bus_x'], -97.0)
        self.assertEqual(result.loc[0, 'bus_y'], 31.0)

    def test_datacenter_input_accepts_public_csv(self):
        with tempfile.NamedTemporaryFile(suffix='.csv', mode='w') as data:
            data.write('full_address,State,current_mw,construction_mw,planned_mw\n')
            data.write('Synthetic Site,TX,1,2,3\n')
            data.flush()

            result = dc_common.read_datacenter_table(Path(data.name))

        self.assertEqual(result.loc[0, 'planned_mw'], 3)

if __name__=='__main__':unittest.main()
