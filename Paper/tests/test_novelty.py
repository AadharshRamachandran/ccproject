import sys
from pathlib import Path
import unittest
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Novelty.src.burst import AdaptiveBurstDetector, BurstParameters, tune_burst_parameters
from Novelty.src.monitoring import AdaptiveMonitoringInterval
from Novelty.ablation import cost_eq_2, cost_eq_3, cost_eq_4
from src.data import worldcup98_to_workload


class NoveltyTests(unittest.TestCase):
    def test_adaptive_burst_detects_large_jump(self):
        detector=AdaptiveBurstDetector(parameters=BurstParameters(threshold=2,volatility_adaptive=True,consecutive_exceedances=1))
        detector.warm([10]*10); self.assertTrue(detector.update(100))

    def test_tuning_uses_lowest_scorer_result(self):
        params,score=tune_burst_parameters([10,10,100],[10,10,100],lambda segment,flags,actual: sum(not x for x in flags),thresholds=(1,),influences=(.5,),adaptive=(False,),folds=1)
        self.assertEqual(params.threshold,1); self.assertEqual(score,2)

    def test_monitor_interval_shortens_for_burst_and_lengthens_when_flat(self):
        policy=AdaptiveMonitoringInterval(minimum_seconds=15,maximum_seconds=120)
        for value in (100,100,100): interval=policy.update(value)
        self.assertEqual(interval,120); self.assertEqual(policy.update(300,burst=True),15)

    def test_inplace_removes_only_overlap_component(self):
        base=cost_eq_2(2,600,1e-6,60); update=cost_eq_3(1,600,2,700,1e-6,60,12)
        self.assertGreater(update,base); self.assertEqual(cost_eq_4(1,600,2,700,1e-6,0),0)

    def test_worldcup_count_format_is_minute_aggregated_and_zero_filled(self):
        raw=pd.DataFrame({'period':['1998-06-30 08:00:01','1998-06-30 08:02:01'],'count':[3,5]})
        result=worldcup98_to_workload(raw)
        self.assertEqual(result.request_rate.tolist(),[3,0,5])

if __name__=='__main__': unittest.main()
