"""Trace-driven evaluation of isolated novelty effects against the paper baseline.

This emulator is for reproducible policy comparison. Confirm the finalist
configurations with the accompanying Kubernetes/Locust run before reporting
empirical latency or cost claims.
"""
from __future__ import annotations
import argparse
import json
from itertools import combinations
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'Paper'))
from src.burst import OnlineBurstDetector
from src.data import load_workload_csv
from Novelty.src.performance import PerformanceModel
from src.scaler import HybridProvisioner, ScalingDecision
from Novelty.src.burst import AdaptiveBurstDetector, BurstParameters, tune_burst_parameters
from Novelty.src.hpa import ReactiveHPA
from Novelty.src.monitoring import AdaptiveMonitoringInterval
from Novelty.src.scaler import UncertaintyAwareProvisioner

PAPER_NAME='paper_hybrid_bilstm_dtr_rolling_fixed_monitor'

@dataclass(frozen=True)
class ExperimentConfig:
    system: str
    forecaster: str='bilstm'
    burst: str='paper_fixed'
    performance: str='dtr'
    execution: str='rolling'
    monitoring: str='fixed'
    label: str|None=None
    def name(self):
        if self.system=='reactive_hpa': return 'reactive_hpa_fixed_cpu'
        if self.system=='proactive_hpa': return 'proactive_hpa_bilstm_fixed_cpu'
        if self.system=='burst_aware_hpa': return 'burst_aware_hpa_bilstm_fixed_cpu'
        if self==paper_config(): return PAPER_NAME
        return self.label or 'novelty_'+'_'.join((self.forecaster,self.burst,self.performance,self.execution,self.monitoring))

def paper_config(): return ExperimentConfig('hybrid')
def cost_eq_2(replicas,cpu,price,seconds): return replicas*cpu*price*seconds
def cost_eq_4(old_replicas,old_cpu,replicas,cpu,price,overlap_seconds): return (old_replicas*old_cpu+replicas*cpu)*price*overlap_seconds
def cost_eq_3(old_replicas,old_cpu,replicas,cpu,price,interval_seconds,overlap_seconds): return cost_eq_4(old_replicas,old_cpu,replicas,cpu,price,overlap_seconds)+cost_eq_2(replicas,cpu,price,interval_seconds-overlap_seconds)

def synthetic_performance():
    rows=[]
    for cpu in (600,700,800,900,950):
        for rate in range(50,2001,50):
            replicas=max(1,int(np.ceil(rate/(cpu*.42))))
            rows.append((rate,cpu,350,replicas,min(.99,rate/(replicas*cpu*.42))))
    return pd.DataFrame(rows,columns=['request_rate','cpu_millicores','slo_ms','replicas','utilization'])

def _factory(name):
    # Keeps policy-only utilities importable without requiring the optional
    # PyTorch runtime; training paths import models only when actually used.
    from Novelty.src.forecast import WorkloadForecaster, QuantileBiLSTMForecaster
    return {'bilstm':WorkloadForecaster,'quantile_bilstm':QuantileBiLSTMForecaster}[name]

def _predictions(model,values,history):
    if hasattr(model,'predict_intervals_many'):
        intervals=model.predict_intervals_many(history,values)
        return intervals[:,1].astype(float),intervals.astype(float)
    if hasattr(model,'predict_many'):
        points=model.predict_many(history,values)
        return points.astype(float),np.column_stack((points,points,points)).astype(float)
    points=[]; intervals=[]; running=list(history)
    for actual in values:
        if hasattr(model,'predict_interval'): low,mid,high=model.predict_interval(running)
        else: mid=model.predict(running); low=high=mid
        points.append(float(mid)); intervals.append((float(low),float(mid),float(high))); running.append(float(actual))
    return np.asarray(points),np.asarray(intervals)

def _seed_model(seed):
    import random
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

def forecast_cache(train,validation,test,epochs,batch_size,names=('bilstm','quantile_bilstm'),models_dir=None,window_stride=5,quantile_window_stride=5,progress_every=5,patchtst_epochs=120,seed=42):
    """Each model trains once, then all factorial variants reuse its predictions."""
    results={}
    for name in names:
        model=None
        loaded=False
        if models_dir:
            import os
            model_path=os.path.join(models_dir,f'{name}.pt')
            if os.path.exists(model_path):
                print(f"Loading pre-trained {name} from {model_path} for ablation cache...", flush=True)
                model=_factory(name).load(model_path)
                loaded=True
        if not loaded:
            _seed_model(seed)
            model=_factory(name)()
            fit_epochs = patchtst_epochs if name == 'patchtst' else epochs
            print(f"Training {name} from scratch for {fit_epochs} epochs...", flush=True)
            model.fit(train,epochs=fit_epochs,batch_size=batch_size,
                      window_stride=quantile_window_stride if name=='quantile_bilstm' else window_stride,
                      progress_every=progress_every,label=name,seed=seed)
        val,val_intervals=_predictions(model,validation,train)
        point,intervals=_predictions(model,test,list(train)+list(validation))
        results[name]={'validation':val,'validation_actual':np.asarray(validation),'validation_intervals':val_intervals,'test':point,'intervals':intervals}
    return results

def _make_model(frame,name):
    model=PerformanceModel()
    model.fit(frame); return model

def _tuned_parameters(predicted,actual,performance_model,slo_target=.0368,price=1e-6,interval_seconds=60):
    """Minimize cost subject to matching the paper SLO target on validation.
     
    Uses soft SLO constraint: cost is penalized proportionally by SLO miss amount,
    rather than hitting a hard cliff at 1e6. This prevents one hard fold from
    dominating the worst-fold scoring and allows the search to find policies that
    are "good enough" (low SLO violation with reasonable cost) even if not perfect.
    """
    # The adaptive-burst ablation changes the detector only; retain the Paper
    # Hybrid provisioning rule so burst results do not include quantile logic.
    provisioner=HybridProvisioner(performance_model)
    # Provisioning depends only on a forecast and the burst flag, not on a
    # candidate detector. Compute both outcomes once, then tune the detector
    # with inexpensive lookups instead of repeated DTR predictions.
    normal_decisions,burst_decisions=_batch_paper_decisions(predicted,performance_model)
    decisions={float(pred):(normal_decisions[index],burst_decisions[index])
               for index,pred in enumerate(predicted)}
    def replay(segment_predictions,flags,values):
        slo_hits=[]; costs=[]; transitions=0
        replicas,cpu=1,600; pending=None
        for tick,(pred,actual_load,burst) in enumerate(zip(segment_predictions,values,flags)):
            transition=None
            if pending is not None and pending[0]<=tick:
                decision=pending[1]; old_replicas,old_cpu=replicas,cpu
                replicas,cpu=decision.replicas,decision.cpu_millicores
                transition=(old_replicas,old_cpu); pending=None
            decision=decisions[float(pred)]
            selected=decision[int(bool(burst))]
            delay=int(selected.cpu_millicores!=cpu)
            if delay:
                pending=(tick+delay,selected)
            else:
                old_replicas,old_cpu=replicas,cpu
                replicas,cpu=selected.replicas,selected.cpu_millicores
                if (old_replicas,old_cpu)!=(replicas,cpu):
                    transition=(old_replicas,old_cpu)
            _util,response_ms=_latency(float(actual_load),replicas,cpu)
            slo_hits.append(response_ms>350)
            if transition and transition[1]!=cpu:
                costs.append(cost_eq_3(transition[0],transition[1],replicas,cpu,price,interval_seconds,12))
            else:
                costs.append(cost_eq_2(replicas,cpu,price,interval_seconds))
            transitions+=int(bool(transition))
        return np.mean(slo_hits),np.sum(costs),transitions
    reference_cache={}
    def objective(segment_predictions,flags,values):
        slo_rate,total_cost,transitions=replay(segment_predictions,flags,values)
        key=np.asarray(segment_predictions,dtype=float).tobytes()
        if key not in reference_cache:
            reference=OnlineBurstDetector(window=10,threshold=5.,influence=.5)
            reference_flags=[reference.update(float(value)) for value in segment_predictions]
            reference_cache[key]=replay(segment_predictions,reference_flags,values)[:2]
        baseline_slo,baseline_cost=reference_cache[key]
        # Prefer a strict SLO and cost win over the same validation fold.
        # If the search cannot find one, penalize SLO regression before small
        # cost differences so it does not buy savings by accepting more misses.
        beats=slo_rate<baseline_slo and total_cost<baseline_cost
        if beats: return total_cost+.5*transitions
        slo_excess=max(0.0,slo_rate-(baseline_slo-1/max(len(flags),1)))
        cost_excess=max(0.0,total_cost-baseline_cost)
        return total_cost+.5*transitions+1000.0*slo_excess+cost_excess
    # Tune Algorithm 1 on the same predicted-workload signal the paper uses.
    # Searching thresholds below 5 lets N1 fire at the start of a ramp instead
    # of waiting for a 5-sigma jump, which is the usual source of hybrid SLO
    # misses. consecutive_exceedances=1 matches the paper's immediate trigger.
    floors=(0.0,)
    return tune_burst_parameters(predicted,actual,objective,
                                 thresholds=(.5,1.,1.5,2.,3.,5.),
                                 influences=(.2,.5,.8),
                                 adaptive=(True,),
                                 consecutive_exceedances=(1,),
                                 magnitude_floors=floors,
                                 forecast_error_margins=(0.0,),
                                 volatility_gains=(.5,1.0),min_thresholds=(.5,),
                                 signals=predicted)

def _provisioner(config,performance_model):
    # Quantile provisioning is an isolated novelty. Other arms, including
    # adaptive burst detection and monitoring, keep the Paper Hybrid scaler.
    if config.forecaster=='quantile_bilstm':
        return UncertaintyAwareProvisioner(performance_model)
    return HybridProvisioner(performance_model)

def _quantile_policy_parameters(policy):
    """Return (uncertainty slope, risk threshold) for the gated quantile arm."""
    if isinstance(policy,(tuple,list)):
        return float(policy[0]),float(policy[1])
    return float(policy),.5

def _batch_paper_decisions(workloads,performance_model,resources=(600,700,800,900,950),slo_ms=350):
    """Run the canonical Paper Hybrid policy for normal and burst forecasts."""
    provisioner=HybridProvisioner(performance_model,resources=resources,slo_ms=slo_ms)
    normal=[provisioner.decide(float(workload),burst=False) for workload in workloads]
    burst=[provisioner.decide(float(workload),burst=True) for workload in workloads]
    return normal,burst

def _batch_decisions_for_resources(workloads,cpus,performance_model,slo_ms=350,burst=False):
    workloads=np.asarray(workloads,dtype=float); cpus=np.asarray(cpus,dtype=int)
    frame=pd.DataFrame({'request_rate':workloads,'cpu_millicores':cpus,'slo_ms':slo_ms})
    values=performance_model.predict_frame(frame)
    return [ScalingDecision(max(1,int(np.ceil(row[0]))),int(cpu),
                            float(np.clip(row[1],0,1)),burst)
            for row,cpu in zip(values,cpus)]

def _latency(demand,replicas,cpu):
    utilization=demand/max(replicas*cpu*.42,1e-9)
    return min(utilization,1.5),80+180*min(utilization,1.5)**3

def _bootstrap(values,statistic,samples,rng,block=10):
    values=np.asarray(values); n=len(values)
    if samples<=0 or n<2: return np.nan,np.nan
    starts=np.arange(0,n,block); positions=np.arange(n); estimates=[]
    # Batch the resampling indices to avoid Python-level loops over every
    # sampled block (important for full-trace studies with thousands of draws).
    batch_size=64; initial_blocks=int(np.ceil(n/block))+block
    for batch_start in range(0,samples,batch_size):
        count=min(batch_size,samples-batch_start)
        chosen=rng.choice(starts,size=(count,initial_blocks))
        lengths=np.minimum(block,n-chosen)
        cumulative=np.cumsum(lengths,axis=1)
        while np.any(cumulative[:,-1]<n):
            short=cumulative[:,-1]<n
            chosen[short,-block:]=rng.choice(starts,size=(int(short.sum()),block))
            lengths=np.minimum(block,n-chosen[short])
            cumulative[short]=np.cumsum(lengths,axis=1)
        sampled_indices=np.empty((count,n),dtype=np.int32)
        for row in range(count):
            blocks=np.searchsorted(cumulative[row],positions,side='right')
            previous=np.where(blocks==0,0,cumulative[row,np.maximum(blocks-1,0)])
            sampled_indices[row]=chosen[row,blocks]+positions-previous
        sampled=values[sampled_indices]
        estimates.extend(statistic(sample) for sample in sampled)
    return tuple(np.quantile(estimates,[.025,.975]))

def evaluate(config,actual_raw,demand,forecast,models,interval_seconds=60,overlap_seconds=12,price=1e-6,quantile_safety_gain=0.5,burst_slo_target=.0368,monitoring_options=None):
    """Policy execution with reactive delay and paper Eq. (2)-(4) accounting."""
    model=models[config.performance]; params=BurstParameters(); tune_score=np.nan
    if config.burst=='tuned_adaptive': params,tune_score=_tuned_parameters(forecast['validation'],forecast['validation_actual'],model,slo_target=burst_slo_target)
    detector=AdaptiveBurstDetector(parameters=params) if config.burst=='tuned_adaptive' else OnlineBurstDetector(window=10,threshold=5.,influence=.5)
    # Algorithm 1 is initialized from the final k measured workload points
    # available before the test segment. Use identical warm state for Paper
    # Hybrid and N1 so novelty comparisons change the detector policy only.
    detector.warm(np.asarray(forecast['validation_actual'],dtype=float)[-10:])
    hpa_burst_detector=OnlineBurstDetector(window=10,threshold=5.,influence=.5) if config.system=='burst_aware_hpa' else None
    if hpa_burst_detector is not None:
        hpa_burst_detector.warm(np.asarray(forecast['validation_actual'],dtype=float)[-10:])
    monitoring_options=monitoring_options or {}
    interval_policy=AdaptiveMonitoringInterval(base_seconds=interval_seconds,
        minimum_seconds=monitoring_options.get('minimum_seconds',15),
        maximum_seconds=monitoring_options.get('maximum_seconds',240),
        high_cv_percentile=monitoring_options.get('high_cv_percentile',.60))
    if config.monitoring=='adaptive':
        interval_policy.calibrate(forecast['validation_actual'],high_cv_percentile=interval_policy.high_cv_percentile)
        print(f'{config.name()} monitoring calibration: low_cv={interval_policy.low_cv:.3f}, high_cv={interval_policy.high_cv:.3f}, max={interval_policy.maximum}s')
    provisioner=_provisioner(config,model)
    if config.forecaster=='quantile_bilstm':
        slope,_threshold=_quantile_policy_parameters(quantile_safety_gain)
        provisioner.interval_safety_gain=slope
        widths=[provisioner._relative_width(interval) for interval in forecast['validation_intervals']]
        provisioner.set_interval_calibration(widths)
        print(f'{config.name()} interval-width calibration: p99 burst saturation={provisioner.max_relative_width:.3f}')
        if config.monitoring=='adaptive':
            interval_policy.calibrate(forecast['validation_actual'],forecast_widths=widths,high_cv_percentile=interval_policy.high_cv_percentile)
            print(f'{config.name()} forecast-width calibration: low={interval_policy.low_forecast_width:.3f}, high={interval_policy.high_forecast_width:.3f}')
    decisions_by_tick=None
    if config.system=='hybrid':
        if config.forecaster=='quantile_bilstm':
            bands=np.asarray(forecast['intervals'],dtype=float)
            lows,medians,highs=bands[:,0],bands[:,1],bands[:,2]
            widths=(highs-lows)/np.maximum(np.abs(medians),1e-9)
            gain,threshold=_quantile_policy_parameters(quantile_safety_gain)
            # Risk-gated quantile provisioning: use a lower-tail estimate in
            # tightly calibrated periods, and progressively use P90 as
            # uncertainty grows. This avoids paying a permanent safety margin.
            relative=np.asarray([provisioner._relative_width(row) for row in bands])
            risk=np.clip(relative/max(provisioner.max_relative_width,1e-9),0.,1.)
            signed_gain=np.clip(gain*(risk-threshold),-1.,1.)
            adjusted=medians+signed_gain*np.maximum(0.,highs-medians)
            normal,_=_batch_paper_decisions(adjusted,model)
            relative=np.clip(widths/max(provisioner.max_relative_width,1e-9),0.,1.)
            indexes=np.rint(relative*(len(provisioner.resources)-1)).astype(int)
            burst_cpus=np.asarray(provisioner.resources,dtype=int)[indexes]
            burst_adjusted=medians+relative*np.maximum(0.,highs-medians)
            burst=_batch_decisions_for_resources(burst_adjusted,burst_cpus,model,burst=True)
            decisions_by_tick=list(zip(normal,burst))
        else:
            normal,burst=_batch_paper_decisions(forecast['test'],model)
            decisions_by_tick=list(zip(normal,burst))
    hpa=ReactiveHPA() if config.system in ('reactive_hpa','proactive_hpa','burst_aware_hpa') else None
    replicas,cpu=1,(600 if config.system=='hybrid' else 950); pending=[]; next_monitor=0; previous_monitor_tick=None; last_prediction=float(demand[0]); last_interval=(last_prediction,)*3; last_monitor_interval=interval_seconds; rows=[]
    for tick,(raw,load) in enumerate(zip(actual_raw,demand)):
        transition=None; due=[item for item in pending if item[0]<=tick]; pending=[item for item in pending if item[0]>tick]
        if due:
            _,decision,kind=due[-1]; old_replicas,old_cpu=replicas,cpu; replicas,cpu=decision.replicas,decision.cpu_millicores; transition=(old_replicas,old_cpu,kind)
        monitored=tick>=next_monitor; burst=False; prediction=last_prediction; low,mid,high=last_interval; chosen_interval=max(1,next_monitor-tick); after_stretched_interval=last_monitor_interval>interval_seconds
        # Decide from completed observations only; including this tick leaks
        # the value that has not yet been observed by the monitor.
        monitor_cv=interval_policy.history_cv() if config.monitoring=='adaptive' else 0.0
        if monitored:
            prediction=float(forecast['test'][tick]); low,mid,high=forecast['intervals'][tick]; last_prediction=prediction; last_interval=(low,mid,high)
            if config.system=='reactive_hpa': decision=hpa.decide(demand[max(0,tick-1)]); delay=1
            elif config.system=='proactive_hpa': decision=hpa.decide(prediction); delay=0
            elif config.system=='burst_aware_hpa':
                burst=hpa_burst_detector.update(prediction)
                decision=hpa.decide(prediction*(1.2 if burst else 1.0)); delay=0
            else:
                burst=detector.update(prediction)
                if decisions_by_tick is not None:
                    decision=decisions_by_tick[tick][int(bool(burst))]
                else:
                    decision=provisioner.decide(prediction,burst=burst)
                hold_cv=monitoring_options.get('hold_scale_down_cv')
                if hold_cv is not None and monitor_cv>=float(hold_cv) and decision.replicas*decision.cpu_millicores < replicas*cpu:
                    # During volatile periods, keep the current allocation
                    # until the next adaptive sample instead of risking a
                    # rolling downsize immediately before a rebound.
                    decision=ScalingDecision(replicas,cpu,
                        load/max(replicas*cpu*.42,1e-9),burst)
                delay=int(decision.cpu_millicores!=cpu and config.execution=='rolling')
            if delay: pending.append((tick+delay,decision,'rolling'))
            else:
                old_replicas,old_cpu=replicas,cpu; replicas,cpu=decision.replicas,decision.cpu_millicores
                if (old_replicas,old_cpu)!=(replicas,cpu): transition=(old_replicas,old_cpu,config.execution)
            forecast_width=provisioner._relative_width((low,mid,high)) if config.forecaster=='quantile_bilstm' else 0.
            chosen_interval=interval_policy.update(load,burst,forecast_width) if config.monitoring=='adaptive' else interval_seconds
            last_monitor_interval=chosen_interval
            next_monitor=tick+max(1,int(np.ceil(chosen_interval/interval_seconds)))
        utilization,response_ms=_latency(load,replicas,cpu)
        if transition and transition[2]=='rolling' and transition[1]!=cpu:
            resource_cost=cost_eq_3(transition[0],transition[1],replicas,cpu,price,interval_seconds,overlap_seconds); overlap=cost_eq_4(transition[0],transition[1],replicas,cpu,price,overlap_seconds)
        else: resource_cost=cost_eq_2(replicas,cpu,price,interval_seconds); overlap=0.
        rows.append({'variant':config.name(),'system':config.system,'tick':tick,'actual_request_rate':raw,'scaled_demand':load,'predicted_demand':prediction,'p10':low,'p50':mid,'p90':high,'monitored':monitored,'next_monitor_tick':next_monitor,'realized_monitor_interval_seconds':(tick-previous_monitor_tick)*interval_seconds if monitored and previous_monitor_tick is not None else np.nan,'burst':burst,'next_interval_seconds':chosen_interval,'after_stretched_interval':after_stretched_interval,'replicas':replicas,'cpu_millicores':cpu,'utilization':utilization,'response_ms':response_ms,'slo_violation':response_ms>350,'resource_cost':resource_cost,'rolling_overlap_cost':overlap,'execution_backend':config.execution})
        if monitored: previous_monitor_tick=tick
    return pd.DataFrame(rows),tune_score

def _validation_replay(forecast):
    """Expose a chronological validation forecast as a replay, without test data."""
    return {
        'validation': forecast['validation'],
        'validation_actual': forecast['validation_actual'],
        'validation_intervals': forecast['validation_intervals'],
        'test': forecast['validation'],
        'intervals': forecast['validation_intervals'],
    }

def _select_quantile_gains(selected,cache,models,raw,demand,cut,validation_size):
    quantile_configs=[config for config in selected if config.forecaster=='quantile_bilstm']
    if not quantile_configs: return {}
    start=cut-validation_size
    actual_raw=raw[start:cut]; validation_demand=demand[start:cut]
    policies={}
    # Select a distinct risk gate for rolling and in-place execution. This
    # accounts for the execution mode's cost model on validation, then freezes
    # the selected rule across the corresponding factorial arms.
    for execution in ('rolling','inplace'):
        group=[item for item in quantile_configs if item.execution==execution]
        if not group: continue
        baseline_config=(paper_config() if execution=='rolling' else
                         ExperimentConfig('hybrid',execution='inplace',label='_inplace_reference'))
        baseline_forecaster='bilstm'
        baseline,_=evaluate(baseline_config,actual_raw,validation_demand,
                            _validation_replay(cache[baseline_forecaster]),models)
        baseline_slo=float(baseline.slo_violation.mean())
        baseline_cost=float(baseline.resource_cost.sum())
        config=ExperimentConfig('hybrid',forecaster='quantile_bilstm',
                                execution=execution,label=f'_quantile_{execution}_tuning')
        candidates=[]
        for gain in ((slope,threshold) for slope in (0.,.25,.5,1.,1.5,2.)
                     for threshold in (.35,.5,.65,.8,1.0)):
            detail,_=evaluate(config,actual_raw,validation_demand,
                              _validation_replay(cache['quantile_bilstm']),models,
                              quantile_safety_gain=gain)
            slo=float(detail.slo_violation.mean()); cost=float(detail.resource_cost.sum())
            beats=(slo<baseline_slo and cost<baseline_cost)
            score=(0.0,cost,slo) if beats else (
                1.0 + max(0.0,slo-baseline_slo)*100.0 +
                max(0.0,cost-baseline_cost)/max(baseline_cost,1e-9),cost,slo)
            candidates.append((score,gain,slo,cost,beats))
        _score,gain,slo,cost,beats=min(candidates,key=lambda item:item[0])
        slope,threshold=gain
        print(f'quantile/{execution} validation tuning: {"beats both" if beats else "best available"} '
              f'risk-gated slope={slope:.2f}, threshold={threshold:.2f}; '
              f'SLO={100*slo:.3f}% vs reference={100*baseline_slo:.3f}%, '
              f'cost={cost:.3f} vs reference={baseline_cost:.3f}',flush=True)
        policies.update({item.name():gain for item in group})
    return policies

def _select_validation_policy_options(selected,cache,models,raw,demand,cut,validation_size,quantile_gains):
    """Tune novelty-specific controls on validation only, against Paper Hybrid."""
    start=cut-validation_size
    actual_raw=raw[start:cut]; validation_demand=demand[start:cut]
    baseline,_=evaluate(paper_config(),actual_raw,validation_demand,
                        _validation_replay(cache['bilstm']),models)
    baseline_slo=float(baseline.slo_violation.mean())
    baseline_cost=float(baseline.resource_cost.sum())
    strict_target=max(0.0,baseline_slo-1.0/max(len(baseline),1))
    burst_targets={config.name():strict_target for config in selected if config.burst=='tuned_adaptive'}
    monitoring_options={}
    adaptive_configs=[item for item in selected if item.monitoring=='adaptive']
    if adaptive_configs:
        # Select a single cadence on the isolated monitoring arm, then freeze
        # it across combinations so interactions are measured on held-out data.
        config=ExperimentConfig('hybrid',monitoring='adaptive',label='_monitor_tuning')
        candidates=[]
        for maximum in (60,75,90,120,180,240):
            for percentile in (.20,.40,.60,.80):
             for hold_cv in (None,.18):
                options={'minimum_seconds':15,'maximum_seconds':maximum,
                         'high_cv_percentile':percentile,'hold_scale_down_cv':hold_cv}
                detail,_=evaluate(config,actual_raw,validation_demand,
                    _validation_replay(cache[config.forecaster]),models,
                    quantile_safety_gain=.5,
                    monitoring_options=options)
                slo=float(detail.slo_violation.mean()); cost=float(detail.resource_cost.sum())
                feasible=slo<baseline_slo and cost<baseline_cost
                score=(0,cost,slo) if feasible else (
                    max(0.0,slo-baseline_slo)/max(baseline_slo,1/len(baseline))+
                    max(0.0,cost-baseline_cost)/max(baseline_cost,1e-9),
                    slo,cost)
                candidates.append((score,options,slo,cost,feasible))
        _score,options,slo,cost,feasible=min(candidates,key=lambda item:item[0])
        print(f'monitoring validation tuning: {"beats both" if feasible else "best available"}; '
              f'monitor max={options["maximum_seconds"]}s, percentile={options["high_cv_percentile"]:.2f}; '
              f'downscale hold CV={options["hold_scale_down_cv"]}; '
              f'SLO={100*slo:.3f}% vs paper={100*baseline_slo:.3f}%, cost={cost:.3f} vs paper={baseline_cost:.3f}',flush=True)
        monitoring_options={item.name():options for item in adaptive_configs}
    for name,target in burst_targets.items():
        print(f'{name} validation SLO target: strictly below Paper Hybrid ({100*target:.3f}% vs '
              f'{100*baseline_slo:.3f}%)',flush=True)
    return burst_targets,monitoring_options

def summarize(detail,config,tune_score,samples,rng,forecaster_eligible=None,performance_eligible=None,forecast=None):
    violation=detail.slo_violation.to_numpy(dtype=float); cost=detail.resource_cost.to_numpy(); util=detail.utilization.clip(upper=1).to_numpy()
    slo_ci=_bootstrap(violation,np.mean,samples,rng); cost_ci=_bootstrap(cost,np.sum,samples,rng); util_ci=_bootstrap(util,np.mean,samples,rng)
    stretched=detail.loc[detail.after_stretched_interval,'slo_violation']
    coverage=np.nan
    if config.forecaster=='quantile_bilstm' and forecast is not None:
        actual=detail.scaled_demand.to_numpy()
        intervals=np.asarray(forecast['intervals'])
        coverage=100*np.mean((actual>=intervals[:,0])&(actual<=intervals[:,2]))
    forecast_mae=(np.mean(np.abs(detail.scaled_demand.to_numpy()-np.asarray(forecast['test'])))
                  if forecast is not None else np.nan)
    realized=detail.realized_monitor_interval_seconds.dropna()
    return {'variant':config.name(),**asdict(config),'forecaster_policy_eligible':forecaster_eligible,'performance_policy_eligible':performance_eligible,'intervals':len(detail),'slo_violation_pct':100*violation.mean(),'slo_ci95_low_pct':100*slo_ci[0],'slo_ci95_high_pct':100*slo_ci[1],'mean_utilization_pct':100*util.mean(),'utilization_ci95_low_pct':100*util_ci[0],'utilization_ci95_high_pct':100*util_ci[1],'resource_cost':cost.sum(),'cost_ci95_low':cost_ci[0],'cost_ci95_high':cost_ci[1],'rolling_overlap_cost':detail.rolling_overlap_cost.sum(),'monitor_cycles':int(detail.monitored.sum()),'mean_monitor_interval_seconds':realized.mean() if len(realized) else np.nan,'slo_after_stretched_interval_pct':np.nan if stretched.empty else 100*stretched.mean(),'quantile_p10_p90_coverage_pct':coverage,'forecast_mae_scaled':forecast_mae,'tuning_validation_objective':tune_score}

def configurations():
    """Return controls plus one experiment for each independent novelty.

    Every novelty differs from the paper hybrid in exactly one dimension, so
    its delta can be interpreted directly rather than as an interaction term.
    """
    yield ExperimentConfig('reactive_hpa')
    yield ExperimentConfig('proactive_hpa')
    yield ExperimentConfig('burst_aware_hpa')
    yield paper_config()
    yield ExperimentConfig('hybrid', burst='tuned_adaptive', label='novelty_1_tuned_adaptive_burst')
    yield ExperimentConfig('hybrid', forecaster='quantile_bilstm', label='novelty_2_quantile_provisioning')
    yield ExperimentConfig('hybrid', monitoring='adaptive', label='novelty_3_adaptive_monitoring')
    yield ExperimentConfig('hybrid', execution='inplace', label='novelty_4_inplace_resize')
    # Enumerate every multi-novelty combination (the isolated effects above
    # complete the 2^4 factorial design). Use validation-selected component
    # controls consistently across arms; held-out results capture interactions.
    factors=(('n1_burst',{'burst':'tuned_adaptive'}),
             ('n2_quantile',{'forecaster':'quantile_bilstm'}),
             ('n3_monitoring',{'monitoring':'adaptive'}),
             ('n4_inplace',{'execution':'inplace'}))
    for size in range(2,len(factors)+1):
        for subset in combinations(factors,size):
            settings={}
            for _name,options in subset: settings.update(options)
            label='optimized_novelty_'+'_'.join(name for name,_options in subset)
            yield ExperimentConfig('hybrid',label=label,**settings)

def write_report(summary,out,capacity_peak,performance_source='unspecified',epochs=None,seed=None,pairwise=None,bootstrap_samples=None):
    paper=summary.loc[summary.variant==PAPER_NAME].iloc[0]; top=summary.sort_values(['slo_violation_pct','resource_cost']).head(20)
    columns=['variant','forecaster_policy_eligible','performance_policy_eligible','slo_violation_pct','mean_utilization_pct','resource_cost','rolling_overlap_cost','monitor_cycles','mean_monitor_interval_seconds','quantile_p10_p90_coverage_pct','forecast_mae_scaled','quantile_interval_safety_gain']
    table=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    for _,row in top[columns].iterrows(): table.append('| '+' | '.join(f'{row[column]:.3f}' if isinstance(row[column],(float,np.floating)) else str(row[column]) for column in columns)+' |')
    composites=summary.loc[summary.variant.str.startswith('optimized_novelty_')]
    composite_lines=[]
    if not composites.empty:
        composite_lines=['','## Multi-novelty combinations versus Paper Hybrid','','Each row names the novelty factors included in that configuration. These are combined arms, not isolated one-factor effects. Every comparison uses the same held-out trace and emulator assumptions.','','| Combination | SLO violations | SLO delta (pp) | Mean utilization | Resource cost | Cost delta (%) | Rolling overlap cost |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for _,row in composites.iterrows():
            cost_delta=100*(row.resource_cost-paper.resource_cost)/max(paper.resource_cost,1e-9)
            slo_delta=row.slo_violation_pct-paper.slo_violation_pct
            composite_lines.append(f'| {row.variant} | {row.slo_violation_pct:.3f}% | {slo_delta:+.3f} | {row.mean_utilization_pct:.3f}% | {row.resource_cost:.3f} | {cost_delta:+.2f}% | {row.rolling_overlap_cost:.3f} |')
    run_info=f'Performance data: `{performance_source}`.'
    if 'synthetic' in performance_source.lower(): run_info+=' Synthetic development fixture; not live cluster measurements.'
    else: run_info+=' DTR fitted from the supplied performance table; the trace simulator still estimates response latency and cost from its assumed curves.'
    if epochs is not None: run_info+=f' Forecaster epochs: `{epochs}`.'
    if seed is not None: run_info+=f' Seed: `{seed}`.'
    if bootstrap_samples is not None: run_info+=f' Paired block-bootstrap draws: `{bootstrap_samples}`.'
    paired_lines=[]
    novelty_rows=summary.loc[summary.variant.str.startswith('novelty_')]
    if paper.slo_violation_pct <= 0:
        slo_interpretation='Paper Hybrid is already at 0% SLO violations, so no novelty can show a lower point estimate in this run.'
    else:
        all_novelties=summary.loc[summary.variant.str.startswith(('novelty_','optimized_novelty_'))]
        lower=float(all_novelties.slo_violation_pct.min()); upper=float(all_novelties.slo_violation_pct.max())
        joint=(pairwise.loc[pairwise.variant.str.startswith(('novelty_','optimized_novelty_')),
               ['paired_slo_delta_ci95_high_pp','paired_cost_delta_ci95_high']].lt(0).all(axis=1).any()
               if pairwise is not None and {'paired_slo_delta_ci95_high_pp','paired_cost_delta_ci95_high'}.issubset(pairwise.columns)
               else False)
        slo_interpretation=(f'Novelty SLO point estimates range from {lower:.3f}% to {upper:.3f}% (Paper Hybrid: {paper.slo_violation_pct:.3f}%). '
            + ('At least one arm has paired intervals below zero on both outcomes.' if joint else
               'No arm has paired 95% intervals below zero for both SLO and cost; the data do not establish a joint improvement.'))
    if pairwise is not None:
        paired=pairwise.loc[pairwise.variant.str.startswith(('novelty_','optimized_novelty_'))]
        paired_lines=['','## Paired block-bootstrap differences vs Paper Hybrid','','Negative deltas favor the novelty. Intervals resample the same chronological blocks in both arms.','',
                      '| Novelty | SLO delta (pp), 95% CI | Cost delta, 95% CI |','| --- | ---: | ---: |']
        for _,row in paired.iterrows():
            paired_lines.append(f"| {row.variant} | {row.delta_slo_violation_pct:+.3f} [{row.paired_slo_delta_ci95_low_pp:+.3f}, {row.paired_slo_delta_ci95_high_pp:+.3f}] | {row.delta_resource_cost:+.3f} [{row.paired_cost_delta_ci95_low:+.3f}, {row.paired_cost_delta_ci95_high:+.3f}] |")
    performance_limit=('The performance table is a synthetic development fixture; it is not measured cluster data.'
                       if 'synthetic' in performance_source.lower()
                       else 'The DTR uses the supplied measured calibration data, but the trace simulator still assumes response-latency and cost curves; reported trace SLO and cost results are emulator estimates, not live Kubernetes measurements.')
    lines=['# Paper-method baseline and novelty ablation (offline simulator)','', 'All latency/cost values are simulator estimates. The baseline controller logic is shared from `Paper/src/`; these are not live Kubernetes measurements or the paper\'s originally reported experimental results.','',run_info,'',f'Capacity calibration peak: `{capacity_peak:g}` requests per evaluation interval.','', '## All configurations (SLO first, cost second)','',*table,'',slo_interpretation,'Small cost differences are not evidence of a reliable gain unless the paired confidence interval excludes zero.','', '## Paper-hybrid reference','',f'- SLO violation: {paper.slo_violation_pct:.3f}%\n- Resource cost: {paper.resource_cost:.3f}\n- Mean utilization: {paper.mean_utilization_pct:.3f}%',*paired_lines,*composite_lines,'', '## Interpretation limits','',performance_limit+' The in-place arm removes rolling overlap cost by construction and does not model PodResize latency, readiness delay, or transient CPU mismatch. Treat its small cost delta as a simulator artifact until validated against Kubernetes/Locust measurements. Component eligibility was not benchmarked in this run.','','See `pairwise_vs_paper.csv` for point effects and paired confidence intervals.']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')

def main():
    parser=argparse.ArgumentParser(description='Run a full factorial novelty ablation against Paper Hybrid.')
    parser.add_argument('--workload',required=True); parser.add_argument('--performance'); parser.add_argument('--allow-synthetic',action='store_true'); parser.add_argument('--out',default='Novelty/results')
    parser.add_argument('--max-points',type=int,default=0,help='Most recent chronological points; 0 (default) uses the full trace')
    parser.add_argument('--capacity-peak',type=float,default=1800,help='Map training-trace peak to this app benchmark load')
    parser.add_argument('--epochs',type=int,default=80); parser.add_argument('--batch-size',type=int,default=512); parser.add_argument('--bootstrap-samples',type=int,default=500); parser.add_argument('--only',help='comma-separated config-name fragments'); parser.add_argument('--skip-component-benchmark',action='store_true')
    parser.add_argument('--models',default='Novelty/models')
    parser.add_argument('--skip-forecasters',action='store_true',help='Skip training and load pre-trained forecaster models.')
    parser.add_argument('--window-stride',type=int,default=5)
    parser.add_argument('--quantile-window-stride',type=int,default=5,
                        help='Training-window stride for the quantile forecaster (default: same as Paper Hybrid).')
    parser.add_argument('--progress-every',type=int,default=5)
    parser.add_argument('--patchtst-epochs',type=int,default=120)
    parser.add_argument('--torch-threads',type=int,default=4)
    parser.add_argument('--seed',type=int,default=42,help='Random seed for reproducible forecaster initialization and training.')
    args=parser.parse_args()
    if args.torch_threads:
        import torch
        torch.set_num_threads(args.torch_threads)
    raw=np.asarray(load_workload_csv(args.workload),dtype=float)
    if args.max_points>0: raw=raw[-args.max_points:]
    if len(raw)<160: raise ValueError('Need at least 160 chronological observations.')
    cut=int(.8*len(raw)); scale=args.capacity_peak/max(raw[:cut].max(),1e-9); demand=raw*scale; train,test=demand[:cut],demand[cut:]
    selected=list(configurations())
    if args.only: selected=[config for config in selected if any(part in config.name() for part in args.only.split(','))]
    # A focused novelty run is still an ablation: always retain the paper
    # reference so pairwise deltas and the report remain meaningful.
    if paper_config() not in selected: selected.insert(0,paper_config())
    validation_size=max(20,len(train)//5); forecast_train,validation=train[:-validation_size],train[-validation_size:]
    models_dir = args.models if args.skip_forecasters else None
    cache=forecast_cache(forecast_train,validation,test,args.epochs,args.batch_size,
                         names=tuple(sorted({config.forecaster for config in selected})),
                         models_dir=models_dir, window_stride=args.window_stride,
                         quantile_window_stride=args.quantile_window_stride,
                         progress_every=args.progress_every, patchtst_epochs=args.patchtst_epochs,seed=args.seed)
    if args.performance: frame=pd.read_csv(args.performance)
    elif args.allow_synthetic: frame=synthetic_performance()
    else: raise ValueError('Pass measured --performance data; --allow-synthetic is only for a development dry run.')
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    (out/'run_metadata.json').write_text(json.dumps({
        'workload':str(Path(args.workload).resolve()),'workload_points':int(len(raw)),
        'training_points':int(cut-validation_size),'validation_points':int(validation_size),
        'heldout_points':int(len(test)),'chronological_split':'80% development / 20% held out',
        'seed':int(args.seed),'epochs':int(args.epochs),'batch_size':int(args.batch_size),
        'capacity_peak':float(args.capacity_peak),'performance_data':args.performance or 'synthetic development fixture',
        'performance_model':'normalized multi-output DTR from Paper/src/performance.py',
        'baseline_policy':'canonical Paper/src HybridProvisioner; fixed paper burst detector; fixed monitoring; rolling update',
        'result_type':'offline simulator estimates; not live Kubernetes results',
        'novelty_design':('2^4 factorial: four isolated novelties plus all 11 multi-novelty combinations'
                          if not args.only else 'focused ablation selected by --only'),
        'evaluated_configurations':[config.name() for config in selected],
        'bootstrap_samples':int(args.bootstrap_samples),
    },indent=2),encoding='utf-8')
    forecaster_eligibility={}; performance_eligibility={}
    if not args.skip_component_benchmark:
        from Novelty.src.benchmark import benchmark_forecasters, benchmark_performance_models
        forecast_table=benchmark_forecasters(train,test,args.epochs,args.batch_size,
                                             window_stride=args.window_stride,
                                             models_dir=models_dir,progress_every=args.progress_every,seed=args.seed)
        performance_table=benchmark_performance_models(frame)
        pd.concat([forecast_table,performance_table],ignore_index=True).to_csv(out/'component_benchmark.csv',index=False)
        forecaster_eligibility=dict(zip(forecast_table['name'],forecast_table['policy_eligible']))
        performance_eligibility=dict(zip(performance_table['name'],performance_table['policy_eligible']))
    models={name:_make_model(frame,name) for name in {config.performance for config in selected}}
    quantile_policies=_select_quantile_gains(selected,cache,models,raw,demand,cut,validation_size)
    burst_targets,monitoring_options=_select_validation_policy_options(
        selected,cache,models,raw,demand,cut,validation_size,
        quantile_policies)
    rng=np.random.default_rng(args.seed); summaries=[]; detail_by_variant={}
    for config in selected:
        gain=quantile_policies.get(config.name(),(1.0,.5))
        detail,tune_score=evaluate(config,raw[cut:],test,cache[config.forecaster],models,
            quantile_safety_gain=gain,
            burst_slo_target=burst_targets.get(config.name(),.0368),
            monitoring_options=monitoring_options.get(config.name()))
        detail.to_csv(out/f'{config.name()}.csv',index=False); detail_by_variant[config.name()]=detail
        row=summarize(detail,config,tune_score,args.bootstrap_samples,rng,forecaster_eligibility.get(config.forecaster),performance_eligibility.get(config.performance),cache[config.forecaster])
        slope,threshold=_quantile_policy_parameters(gain)
        row['quantile_interval_safety_gain']=slope if config.forecaster=='quantile_bilstm' else np.nan
        row['quantile_risk_policy']=f'risk_gated_slope={slope:.2f}; threshold={threshold:.2f}; low-risk=P10-leaning, high-risk=P90-leaning' if config.forecaster=='quantile_bilstm' else ''
        if config.monitoring=='adaptive':
            row['monitoring_validation_options']=str(monitoring_options.get(config.name(),{}))
        summaries.append(row); print(config.name())
    summary=pd.DataFrame(summaries); summary.to_csv(out/'summary.csv',index=False); paper=summary.loc[summary.variant==PAPER_NAME].iloc[0]
    pairwise=summary.copy(); pairwise['delta_slo_violation_pct']=pairwise.slo_violation_pct-paper.slo_violation_pct; pairwise['delta_resource_cost']=pairwise.resource_cost-paper.resource_cost; pairwise['delta_utilization_pct']=pairwise.mean_utilization_pct-paper.mean_utilization_pct
    reference=detail_by_variant[PAPER_NAME]
    for index,row in pairwise.iterrows():
        detail=detail_by_variant[row.variant]
        slo_delta=detail.slo_violation.to_numpy(dtype=float)-reference.slo_violation.to_numpy(dtype=float)
        cost_delta=detail.resource_cost.to_numpy(dtype=float)-reference.resource_cost.to_numpy(dtype=float)
        slo_ci=_bootstrap(slo_delta,np.mean,args.bootstrap_samples,rng)
        cost_ci=_bootstrap(cost_delta,np.sum,args.bootstrap_samples,rng)
        pairwise.loc[index,'paired_slo_delta_ci95_low_pp']=100*slo_ci[0]
        pairwise.loc[index,'paired_slo_delta_ci95_high_pp']=100*slo_ci[1]
        pairwise.loc[index,'paired_cost_delta_ci95_low']=cost_ci[0]
        pairwise.loc[index,'paired_cost_delta_ci95_high']=cost_ci[1]
    pairwise.to_csv(out/'pairwise_vs_paper.csv',index=False)
    effects=pairwise.loc[pairwise.variant.str.startswith(('novelty_','optimized_novelty_'))].copy()
    effects.to_csv(out/'novelty_effects_vs_paper.csv',index=False)
    write_report(summary,out,args.capacity_peak,performance_source=args.performance or 'synthetic development fixture',epochs=args.epochs,seed=args.seed,pairwise=pairwise,bootstrap_samples=args.bootstrap_samples)

if __name__=='__main__': main()
