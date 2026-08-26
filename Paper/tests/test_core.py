import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Paper.src.burst import OnlineBurstDetector
from Paper.src.scaler import HybridProvisioner

class Model:
 def predict(self,w,c,s): return (max(1,round(w/c*1000)), c/1000)

class CoreTests(unittest.TestCase):
 def test_burst(self):
  detector=OnlineBurstDetector(window=4,threshold=2,influence=.5); detector.warm([10,10,10,10]); self.assertTrue(detector.update(100))
 def test_burst_uses_max_cpu(self):
  d=HybridProvisioner(Model()).decide(1000,True); self.assertEqual(d.cpu_millicores,950)
 def test_nonburst_selects_highest_utilization(self):
  d=HybridProvisioner(Model()).decide(200,False); self.assertEqual(d.cpu_millicores,950)
if __name__=='__main__': unittest.main()
