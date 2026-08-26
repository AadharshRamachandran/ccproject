"""Paper fixed detector and novelty validated/adaptive burst detector."""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import statistics
from typing import Iterable
from Paper.src.burst import OnlineBurstDetector

@dataclass(frozen=True)
class BurstParameters:
    threshold: float=5.0
    influence: float=.5
    volatility_adaptive: bool=False
    volatility_gain: float=1.0
    min_threshold: float=.25
    consecutive_exceedances: int=2
    magnitude_floor: float=0.0

class AdaptiveBurstDetector:
    def __init__(self,window:int=10,parameters:BurstParameters=BurstParameters()):
        self.values=deque(maxlen=window); self.parameters=parameters; self.min_threshold=parameters.min_threshold; self._exceedances=0
    def warm(self,values:Iterable[float])->None: self.values.extend(float(value) for value in values)
    def _threshold(self,mean:float,std:float)->float:
        if not self.parameters.volatility_adaptive: return self.parameters.threshold
        cv=std/max(abs(mean),1e-9)
        return max(self.min_threshold,self.parameters.threshold*(1+self.parameters.volatility_gain*cv))
    def update(self,predicted:float)->bool:
        predicted=float(predicted)
        if len(self.values)<2: self.values.append(predicted); return False
        mean=statistics.fmean(self.values); std=statistics.stdev(self.values) or 1e-9
        exceeds=predicted-mean>self._threshold(mean,std)*std
        magnitude_ok=predicted>=self.parameters.magnitude_floor if self.parameters.magnitude_floor>0 else True
        self._exceedances=self._exceedances+1 if (exceeds and magnitude_ok) else 0
        burst=(exceeds and magnitude_ok) and self._exceedances>=self.parameters.consecutive_exceedances
        filtered=self.parameters.influence*predicted+(1-self.parameters.influence)*self.values[-1] if burst else predicted
        self.values.append(filtered); return burst

def tune_burst_parameters(predictions,actuals,scorer,thresholds=(1,2,3,4,5,6),influences=(.2,.5,.8),adaptive=(False,True),folds=3,consecutive_exceedances=(1,2,3),magnitude_floors=(0.0,)):
    """Rolling-origin validation search; scorer(predictions, flags, actuals) is lower-is-better.
      
    Uses worst-fold (max) scoring to penalize configs that fail SLO on any fold, rather than
    rewarding those that are OK on average but bad on hard folds.
     
    magnitude_floor optional parameter targets volatile-but-harmless stretches: predicted
    demand must exceed this absolute threshold before triggering a burst flag. Useful to
    reduce false positives during noisy periods that don't actually threaten SLO.
    """
    best=None
    for threshold in thresholds:
        for influence in influences:
            for is_adaptive in adaptive:
                for gain in ((0.5, 1.0, 1.5) if is_adaptive else (1.0,)):
                    for min_threshold in (.25, .5, 1.0):
                        for consec_exceed in consecutive_exceedances:
                            for mag_floor in magnitude_floors:
                                params=BurstParameters(threshold,influence,is_adaptive,gain,min_threshold,consec_exceed,mag_floor)
                                boundaries=range(0,len(predictions),max(1,len(predictions)//folds))
                                scores=[]
                                for start in boundaries:
                                    stop=min(len(predictions),start+max(1,len(predictions)//folds))
                                    detector=AdaptiveBurstDetector(parameters=params)
                                    segment=predictions[start:stop]
                                    flags=[detector.update(value) for value in segment]
                                    scores.append(float(scorer(segment,flags,actuals[start:stop])))
                                score=max(scores)
                                if best is None or score<best[0]: best=(score,params)
    return best[1],best[0]

__all__=['OnlineBurstDetector','AdaptiveBurstDetector','BurstParameters','tune_burst_parameters']
