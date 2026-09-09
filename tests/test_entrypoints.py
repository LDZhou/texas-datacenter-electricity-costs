"""Check the public experiment matrix without scheduling or solving jobs."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_paper
import dc_common

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
        self.assertEqual(dc_common.EXPANDABLE_CARRIERS['generators'],['OCGT','solar','onwind'])
        self.assertNotIn('CCGT',dc_common.FULL_EXPANDABLE_CARRIERS['generators'])

    def test_rep_uses_paper_parameters(self):
        with patch.object(run_paper,'run') as run, patch.object(sys,'argv',['run_paper.py','rep']):
            run_paper.main()
        args=run.call_args.args
        self.assertEqual(args[args.index('--lmp-clip-upper')+1],5000)
        self.assertEqual(args[args.index('--full-tx-tcos-adder-mwh')+1],2.751319804231755)

if __name__=='__main__':unittest.main()
