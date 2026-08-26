"""Trace-driven evaluation of isolated novelty effects against the paper baseline.

This emulator is for reproducible policy comparison. Confirm the finalist
configurations with the accompanying Kubernetes/Locust run before reporting
empirical latency or cost claims.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'Paper'))
from src.burst import OnlineBurstDetector
from src.data import load_workload_csv
from src.performance import PerformanceModel
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
    points=[]; intervals=[]; running=list(history)
    for actual in values:
        if hasattr(model,'predict_interval'): low,mid,high=model.predict_interval(running)
        else: mid=model.predict(running); low=high=mid
        points.append(float(mid)); intervals.append((float(low),float(mid),float(high))); running.append(float(actual))
    return np.asarray(points),np.asarray(intervals)

def forecast_cache(train,validation,test,epochs,batch_size,names=('bilstm','quantile_bilstm'),models_dir=None,window_stride=5,progress_every=5,patchtst_epochs=120):
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
            model=_factory(name)()
            fit_epochs = patchtst_epochs if name == 'patchtst' else epochs
            print(f"Training {name} from scratch for {fit_epochs} epochs...", flush=True)
            model.fit(train,epochs=fit_epochs,batch_size=batch_size,window_stride=window_stride,progress_every=progress_every,label=name)
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
    provisioner=UncertaintyAwareProvisioner(performance_model)
    # Provisioning depends only on a forecast and the burst flag, not on a
    # candidate detector. Compute both outcomes once, then tune the detector
    # with inexpensive lookups instead of repeated DTR predictions.
    decisions={float(pred):
        (provisioner.decide(float(pred), burst=False), provisioner.decide(float(pred), burst=True))
        for pred in predicted
    }
    def objective(segment_predictions,flags,values):
        slo_hits=[]; costs=[]
        transitions=sum(left != right for left,right in zip(flags,flags[1:]))
        for pred,actual_load,burst in zip(segment_predictions,values,flags):
            decision=decisions[float(pred)]
            selected=decision[int(bool(burst))]
            replicas,cpu=selected.replicas,selected.cpu_millicores
            _util,response_ms=_latency(float(actual_load),replicas,cpu)
            slo_hits.append(response_ms>350)
            costs.append(cost_eq_2(replicas,cpu,price,interval_seconds))
        slo_rate=np.mean(slo_hits)
        transition_penalty=.5*transitions
        base_cost=np.sum(costs)+transition_penalty
        if slo_rate <= slo_target:
            return base_cost
        else:
            slo_penalty=1000.0*(slo_rate-slo_target)
            return base_cost+slo_penalty
    return tune_burst_parameters(predicted,actual,objective)

def _provisioner(config,performance_model):
    if config==paper_config(): return HybridProvisioner(performance_model)
    return UncertaintyAwareProvisioner(performance_model)

def _latency(demand,replicas,cpu):
    utilization=demand/max(replicas*cpu*.42,1e-9)
    return min(utilization,1.5),80+180*min(utilization,1.5)**3

def _bootstrap(values,statistic,samples,rng,block=10):
    values=np.asarray(values); n=len(values)
    if samples<=0 or n<2: return np.nan,np.nan
    starts=np.arange(0,n,block); estimates=[]
    for _ in range(samples):
        picked=[]
        while len(picked)<n:
            start=int(rng.choice(starts)); picked.extend(values[start:min(start+block,n)])
        estimates.append(statistic(np.asarray(picked[:n])))
    return tuple(np.quantile(estimates,[.025,.975]))

def evaluate(config,actual_raw,demand,forecast,models,interval_seconds=60,overlap_seconds=12,price=1e-6):
    """Policy execution with reactive delay and paper Eq. (2)-(4) accounting."""
    model=models[config.performance]; params=BurstParameters(); tune_score=np.nan
    if config.burst=='tuned_adaptive': params,tune_score=_tuned_parameters(forecast['validation'],forecast['validation_actual'],model)
    detector=AdaptiveBurstDetector(parameters=params) if config.burst=='tuned_adaptive' else OnlineBurstDetector(window=10,threshold=5.,influence=.5)
    interval_policy=AdaptiveMonitoringInterval(base_seconds=interval_seconds)
    if config.monitoring=='adaptive':
        interval_policy.calibrate(forecast['validation_actual'],high_cv_percentile=0.60)
        print(f'{config.name()} monitoring calibration: low_cv={interval_policy.low_cv:.3f}, high_cv={interval_policy.high_cv:.3f}, max={interval_policy.maximum}s')
    provisioner=_provisioner(config,model)
    if config.forecaster=='quantile_bilstm':
        widths=[provisioner._relative_width(interval) for interval in forecast['validation_intervals']]
        provisioner.set_interval_calibration(widths)
        print(f'{config.name()} interval-width calibration: p99 burst saturation={provisioner.max_relative_width:.3f}')
        if config.monitoring=='adaptive':
            interval_policy.calibrate(forecast['validation_actual'],forecast_widths=widths,high_cv_percentile=0.60)
            print(f'{config.name()} forecast-width calibration: low={interval_policy.low_forecast_width:.3f}, high={interval_policy.high_forecast_width:.3f}')
    hpa=ReactiveHPA() if config.system in ('reactive_hpa','proactive_hpa') else None
    replicas,cpu=1,(600 if config.system=='hybrid' else 950); pending=[]; next_monitor=0; last_prediction=float(demand[0]); last_interval=(last_prediction,)*3; last_monitor_interval=interval_seconds; rows=[]
    for tick,(raw,load) in enumerate(zip(actual_raw,demand)):
        transition=None; due=[item for item in pending if item[0]<=tick]; pending=[item for item in pending if item[0]>tick]
        if due:
            _,decision,kind=due[-1]; old_replicas,old_cpu=replicas,cpu; replicas,cpu=decision.replicas,decision.cpu_millicores; transition=(old_replicas,old_cpu,kind)
        monitored=tick>=next_monitor; burst=False; prediction=last_prediction; low,mid,high=last_interval; chosen_interval=max(1,next_monitor-tick); after_stretched_interval=last_monitor_interval>interval_seconds
        if monitored:
            prediction=float(forecast['test'][tick]); low,mid,high=forecast['intervals'][tick]; last_prediction=prediction; last_interval=(low,mid,high)
            if config.system=='reactive_hpa': decision=hpa.decide(demand[max(0,tick-1)]); delay=1
            elif config.system=='proactive_hpa': decision=hpa.decide(prediction); delay=0
            else:
                burst=detector.update(prediction)
                interval=(low,mid,high) if config.forecaster=='quantile_bilstm' else None
                if isinstance(provisioner, UncertaintyAwareProvisioner):
                    decision=provisioner.decide(prediction,burst=burst,interval=interval)
                else:
                    decision=provisioner.decide(prediction,burst=burst)
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
        rows.append({'variant':config.name(),'system':config.system,'tick':tick,'actual_request_rate':raw,'scaled_demand':load,'predicted_demand':prediction,'p10':low,'p50':mid,'p90':high,'monitored':monitored,'burst':burst,'next_interval_seconds':chosen_interval,'after_stretched_interval':after_stretched_interval,'replicas':replicas,'cpu_millicores':cpu,'utilization':utilization,'response_ms':response_ms,'slo_violation':response_ms>350,'resource_cost':resource_cost,'rolling_overlap_cost':overlap,'execution_backend':config.execution})
    return pd.DataFrame(rows),tune_score

def summarize(detail,config,tune_score,samples,rng,forecaster_eligible=None,performance_eligible=None):
    violation=detail.slo_violation.to_numpy(dtype=float); cost=detail.resource_cost.to_numpy(); util=detail.utilization.clip(upper=1).to_numpy()
    slo_ci=_bootstrap(violation,np.mean,samples,rng); cost_ci=_bootstrap(cost,np.sum,samples,rng); util_ci=_bootstrap(util,np.mean,samples,rng)
    stretched=detail.loc[detail.after_stretched_interval,'slo_violation']
    coverage=np.nan if config.forecaster!='quantile_bilstm' else 100*np.mean((detail.actual_request_rate>=detail.p10)&(detail.actual_request_rate<=detail.p90))
    return {'variant':config.name(),**asdict(config),'forecaster_policy_eligible':forecaster_eligible,'performance_policy_eligible':performance_eligible,'intervals':len(detail),'slo_violation_pct':100*violation.mean(),'slo_ci95_low_pct':100*slo_ci[0],'slo_ci95_high_pct':100*slo_ci[1],'mean_utilization_pct':100*util.mean(),'utilization_ci95_low_pct':100*util_ci[0],'utilization_ci95_high_pct':100*util_ci[1],'resource_cost':cost.sum(),'cost_ci95_low':cost_ci[0],'cost_ci95_high':cost_ci[1],'rolling_overlap_cost':detail.rolling_overlap_cost.sum(),'monitor_cycles':int(detail.monitored.sum()),'mean_monitor_interval_seconds':detail.loc[detail.monitored,'next_interval_seconds'].mean(),'slo_after_stretched_interval_pct':np.nan if stretched.empty else 100*stretched.mean(),'quantile_p10_p90_coverage_pct':coverage,'forecast_mae_scaled':np.mean(np.abs(detail.scaled_demand-detail.predicted_demand)),'tuning_validation_objective':tune_score}

def configurations():
    """Return controls plus one experiment for each independent novelty.

    Every novelty differs from the paper hybrid in exactly one dimension, so
    its delta can be interpreted directly rather than as an interaction term.
    """
    yield ExperimentConfig('reactive_hpa')
    yield ExperimentConfig('proactive_hpa')
    yield paper_config()
    yield ExperimentConfig('hybrid', burst='tuned_adaptive', label='novelty_1_tuned_adaptive_burst')
    yield ExperimentConfig('hybrid', forecaster='quantile_bilstm', label='novelty_2_quantile_provisioning')
    yield ExperimentConfig('hybrid', execution='inplace', label='novelty_3_inplace_resize')
    yield ExperimentConfig('hybrid', monitoring='adaptive', label='novelty_4_adaptive_monitoring')
    yield ExperimentConfig('hybrid', forecaster='quantile_bilstm', monitoring='adaptive', label='novelty_4_stacked_quantile_with_adaptive_cadence')

def write_report(summary,out,capacity_peak):
    paper=summary.loc[summary.variant==PAPER_NAME].iloc[0]; top=summary.sort_values(['slo_violation_pct','resource_cost']).head(20)
    columns=['variant','forecaster_policy_eligible','performance_policy_eligible','slo_violation_pct','mean_utilization_pct','resource_cost','rolling_overlap_cost','monitor_cycles']
    table=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    for _,row in top[columns].iterrows(): table.append('| '+' | '.join(f'{row[column]:.3f}' if isinstance(row[column],(float,np.floating)) else str(row[column]) for column in columns)+' |')
    lines=['# Paper reproduction and novelty ablation','', 'All latency/cost values are emulator estimates; validate finalists with the Kubernetes/Locust experiment.','',f'Capacity calibration peak: `{capacity_peak:g}` requests per evaluation interval.','', '## Top configurations (SLO first, cost second)','',*table,'', 'A `False` eligibility flag means the corresponding component did not meet the held-out quality gate in `component_benchmark.csv`; treat its end-to-end row as diagnostic rather than a validated policy result.','', '## Paper-hybrid reference','',f'- SLO violation: {paper.slo_violation_pct:.3f}\n- Resource cost: {paper.resource_cost:.3f}\n- Mean utilization: {paper.mean_utilization_pct:.3f}%','', 'See `pairwise_vs_paper.csv` for absolute deltas against the paper hybrid.']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')

def main():
    parser=argparse.ArgumentParser(description='Compare each isolated novelty with the paper-hybrid baseline.')
    parser.add_argument('--workload',required=True); parser.add_argument('--performance'); parser.add_argument('--allow-synthetic',action='store_true'); parser.add_argument('--out',default='Novelty/results')
    parser.add_argument('--max-points',type=int,default=7200,help='Most recent chronological points; 0 uses the full trace')
    parser.add_argument('--capacity-peak',type=float,default=1800,help='Map training-trace peak to this app benchmark load')
    parser.add_argument('--epochs',type=int,default=80); parser.add_argument('--batch-size',type=int,default=512); parser.add_argument('--bootstrap-samples',type=int,default=500); parser.add_argument('--only',help='comma-separated config-name fragments'); parser.add_argument('--skip-component-benchmark',action='store_true')
    parser.add_argument('--models',default='Novelty/models')
    parser.add_argument('--skip-forecasters',action='store_true',help='Skip training and load pre-trained forecaster models.')
    parser.add_argument('--window-stride',type=int,default=5)
    parser.add_argument('--progress-every',type=int,default=5)
    parser.add_argument('--patchtst-epochs',type=int,default=120)
    parser.add_argument('--torch-threads',type=int,default=4)
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
                         progress_every=args.progress_every, patchtst_epochs=args.patchtst_epochs)
    if args.performance: frame=pd.read_csv(args.performance)
    elif args.allow_synthetic: frame=synthetic_performance()
    else: raise ValueError('Pass measured --performance data; --allow-synthetic is only for a development dry run.')
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    forecaster_eligibility={}; performance_eligibility={}
    if not args.skip_component_benchmark:
        from Novelty.src.benchmark import benchmark_forecasters, benchmark_performance_models
        forecast_table=benchmark_forecasters(train,test,args.epochs,args.batch_size,
                                             window_stride=args.window_stride,
                                             models_dir=models_dir,progress_every=args.progress_every)
        performance_table=benchmark_performance_models(frame)
        pd.concat([forecast_table,performance_table],ignore_index=True).to_csv(out/'component_benchmark.csv',index=False)
        forecaster_eligibility=dict(zip(forecast_table['name'],forecast_table['policy_eligible']))
        performance_eligibility=dict(zip(performance_table['name'],performance_table['policy_eligible']))
    models={name:_make_model(frame,name) for name in {config.performance for config in selected}}; rng=np.random.default_rng(42); summaries=[]
    for config in selected:
        detail,tune_score=evaluate(config,raw[cut:],test,cache[config.forecaster],models); detail.to_csv(out/f'{config.name()}.csv',index=False); summaries.append(summarize(detail,config,tune_score,args.bootstrap_samples,rng,forecaster_eligibility.get(config.forecaster),performance_eligibility.get(config.performance))); print(config.name())
    summary=pd.DataFrame(summaries); summary.to_csv(out/'summary.csv',index=False); paper=summary.loc[summary.variant==PAPER_NAME].iloc[0]
    pairwise=summary.copy(); pairwise['delta_slo_violation_pct']=pairwise.slo_violation_pct-paper.slo_violation_pct; pairwise['delta_resource_cost']=pairwise.resource_cost-paper.resource_cost; pairwise['delta_utilization_pct']=pairwise.mean_utilization_pct-paper.mean_utilization_pct; pairwise.to_csv(out/'pairwise_vs_paper.csv',index=False)
    effects=pairwise.loc[pairwise.variant.str.startswith('novelty_')].copy()
    effects.to_csv(out/'novelty_effects_vs_paper.csv',index=False)
    write_report(summary,out,args.capacity_peak)

if __name__=='__main__': main()
