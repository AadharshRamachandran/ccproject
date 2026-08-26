"""Adaptive interval policy; monitor.py remains Prometheus query code."""
from collections import deque
import statistics
class AdaptiveMonitoringInterval:
 def __init__(self,minimum_seconds=15,maximum_seconds=240,base_seconds=60,window=10,high_cv=.30,low_cv=.05,high_cv_percentile=0.75): 
  self.minimum,self.maximum,self.base=minimum_seconds,maximum_seconds,base_seconds
  self.high_cv,self.low_cv=high_cv,low_cv
  self.high_cv_percentile=high_cv_percentile
  self.high_forecast_width,self.low_forecast_width=.6,.2
  self.values=deque(maxlen=window)
  self.forecast_widths=deque(maxlen=window)
 def calibrate(self,values,forecast_widths=None,high_cv_percentile=None):
  """Calibrate calm/volatile thresholds from rolling validation CVs and forecast widths.
  
  high_cv_percentile controls how aggressively to classify ticks as 'volatile'.
  Lower values (e.g., 0.60) make the detector tighter/more sensitive to volatility.
  """
  if high_cv_percentile is not None: self.high_cv_percentile=high_cv_percentile
  values=list(map(float,values)); cvs=[]
  for end in range(3,len(values)+1):
   window=values[max(0,end-self.values.maxlen):end]; cvs.append(statistics.stdev(window)/max(abs(statistics.fmean(window)),1e-9))
  if cvs:
   cvs.sort(); self.low_cv=cvs[len(cvs)//4]; self.high_cv=max(self.low_cv+1e-6,cvs[max(0,int((len(cvs)-1)*self.high_cv_percentile))])
  if forecast_widths:
   widths=sorted(float(w) for w in forecast_widths if w>=0)
   if widths:
    self.low_forecast_width=widths[len(widths)//4]; self.high_forecast_width=max(self.low_forecast_width+1e-6,widths[max(0,int((len(widths)-1)*self.high_cv_percentile))])
 def _urgency_percentile(self,cv,forecast_width):
  """Convert both signals to 0-1 urgency scale, blending them for stacked variants."""
  load_urgency=max(0.,min(1.,(cv-self.low_cv)/(self.high_cv-self.low_cv))) if self.high_cv>self.low_cv else 0.
  width_urgency=max(0.,min(1.,(forecast_width-self.low_forecast_width)/(self.high_forecast_width-self.low_forecast_width))) if self.high_forecast_width>self.low_forecast_width else 0.
  return max(load_urgency,width_urgency)
 def update(self,value,burst=False,forecast_relative_width=0.):
  self.values.append(float(value))
  if burst or len(self.values)<3: return self.minimum if burst else self.base
  cv=statistics.stdev(self.values)/max(abs(statistics.fmean(self.values)),1e-9)
  urgency=self._urgency_percentile(cv,float(forecast_relative_width))
  if urgency>=0.75: return self.minimum
  if urgency<=0.25: return self.maximum
  return int(round(self.maximum-(self.maximum-self.minimum)*urgency))


