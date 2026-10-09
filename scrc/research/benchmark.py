"""Isolated fair model benchmark; run: python run.py scrc.research.benchmark."""

import contextlib
import importlib.metadata
import json
import platform
import sys
import time
import traceback
from datetime import UTC, datetime

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from scrc.common import ROOT, digest, save_json
from scrc.data.official import CROPS, SourceRateLimit, fetch, rows
from scrc.production.features import (
    FEATURES,
    build_features,
    eligible,
    encode,
    vocabulary,
)
from scrc.production.train import metrics
from scrc.production.train_forward import SPECS
from scrc.research.calibration import calibrate, interval_metrics, interval_score

OUT = ROOT / "research/supplyguard"
CODES = ["state_code", "market_code", "commodity_code"]


class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()
    def flush(self):
        for stream in self.streams:
            stream.flush()


def model_specs(cfg):
    result = []
    for name, objective, target, leaves, trees in SPECS:
        result.append({"name": name, "family": "lightgbm", "objective": objective,
                       "target": target, "num_leaves": leaves, "n_estimators": trees})
    result.append({"name": "log_l2_31", "family": "lightgbm", "objective": "regression",
                   "target": "log", "num_leaves": 31, "n_estimators": 400})
    result += [{**r, "family": "random_forest"} for r in cfg["rf_specs"]]
    return result


def fit_predictor(train, spec, features, seed=42):
    """All vocabularies and imputers fit only on this training partition."""
    vocab = vocabulary(train)
    x = encode(train, vocab)[features]
    target = train.arrivals.to_numpy()
    if spec["target"] == "log":
        target = np.log1p(target)
    elif spec["target"] == "ratio":
        target = target / train.rolling_mean.clip(lower=1).to_numpy()
    started = time.perf_counter()
    if spec["family"] == "random_forest":
        numeric = [f for f in features if f not in CODES]
        categories = [f for f in features if f in CODES]
        transform = ColumnTransformer([
            ("numeric", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True), numeric),
            ("categories", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categories),
        ])
        model = Pipeline([("transform", transform), ("model", RandomForestRegressor(
            n_estimators=spec["n_estimators"], max_depth=spec["max_depth"],
            min_samples_leaf=spec["min_samples_leaf"], max_features=spec["max_features"],
            random_state=seed, n_jobs=2))])
        model.fit(x, target)
    else:
        model = LGBMRegressor(objective=spec["objective"], n_estimators=spec["n_estimators"],
            learning_rate=0.035, num_leaves=spec["num_leaves"], min_child_samples=50,
            reg_lambda=5, deterministic=True, force_col_wise=True, random_state=seed,
            n_jobs=2, verbosity=-1)
        model.fit(x, target, categorical_feature=[f for f in CODES if f in features])
    return {"model": model, "vocabulary": vocab, "spec": spec,
            "features": features, "training_seconds": time.perf_counter() - started}


def predict(fitted, frame):
    started = time.perf_counter()
    answer = fitted["model"].predict(encode(frame, fitted["vocabulary"])[fitted["features"]])
    if fitted["spec"]["target"] == "log":
        answer = np.expm1(answer)
    elif fitted["spec"]["target"] == "ratio":
        answer *= frame.rolling_mean.clip(lower=1).to_numpy()
    return np.maximum(0, answer), time.perf_counter() - started


def record(frame, prediction):
    answer = frame[["date", "state", "market", "commodity"]].copy()
    answer["actual"] = frame.arrivals.to_numpy()
    answer["scale"] = frame.rolling_mean.clip(lower=1).to_numpy()
    answer["prediction"] = prediction
    return answer.reset_index(drop=True)


def bundle_predictions(predictions, frame, routing):
    out = np.zeros(len(frame))
    for crop, name in routing.items():
        mask = frame.commodity.to_numpy() == crop
        out[mask] = predictions[name][mask]
    return out


def choose_point_models(validation, specs):
    all_scores = validation.groupby(["commodity", "name"]).agg(mean_wape=("wape", "mean"),
        wape_std=("wape", "std"), mean_rmse=("rmse", "mean")).reset_index()
    all_scores["selection_score"] = all_scores.mean_wape + 0.25 * all_scores.wape_std
    originals = {r[0] for r in SPECS}
    routing = {}
    for crop, group in all_scores.loc[all_scores.name.isin(originals)].groupby("commodity"):
        routing[crop] = group.sort_values(["selection_score", "mean_rmse", "name"]).iloc[0]["name"]
    globals_ = validation.groupby(["fold", "name"]).agg(
        absolute_error=("absolute_error", "sum"), actual_total=("actual_total", "sum"))
    globals_["wape"] = globals_.absolute_error / globals_.actual_total
    ranked = globals_.reset_index().groupby("name").wape.agg(["mean", "std"])
    ranked["selection_score"] = ranked["mean"] + 0.25 * ranked["std"]
    rf_names = [s["name"] for s in specs if s["family"] == "random_forest"]
    rf = ranked.loc[rf_names].sort_values(["selection_score"]).index[0]
    log = ranked.loc[["log_l2", "log_l2_31"]].sort_values(["selection_score"]).index[0]
    all_scores.to_csv(OUT / "validation_point_ranking.csv", index=False)
    ranked.to_csv(OUT / "validation_global_ranking.csv")
    return routing, rf, log


def select_adaptive(oof, selected, routing, cfg):
    candidates = []
    for parameters in cfg["adaptive_grid"]:
        scored = []
        for fold, cached in oof.items():
            cal, val = cached["cal"], cached["val"]
            pc = cached["cal_predictions"][selected]
            pv = cached["val_predictions"][selected]
            if selected == "bundle":
                pc = bundle_predictions(cached["cal_predictions"], cal, routing)
                pv = bundle_predictions(cached["val_predictions"], val, routing)
            result, audit = calibrate(record(cal, pc), record(val, pv), "commodity_adaptive",
                **parameters, minimum_count=cfg["minimum_group_count"])
            scored.append({"fold": fold, **interval_metrics(result), **audit})
        candidates.append({**parameters, "coverage": float(np.mean([r["coverage"] for r in scored])),
            "mean_interval_score": float(np.mean([r["mean_interval_score"] for r in scored])),
            "folds": scored})
    pool = [r for r in candidates if r["coverage"] >= 0.94]
    if pool:
        winner = min(pool, key=lambda r:r["mean_interval_score"])
    else:
        winner = min(candidates, key=lambda r:(abs(r["coverage"]-0.95), r["mean_interval_score"]))
    return {"selected": {k:winner[k] for k in ["window", "gamma"]}, "candidates": candidates}


def acquire_later(cfg, observed):
    sources, future = [], []
    chosen = set(map(tuple, observed[["state", "market"]].drop_duplicates().to_numpy()))
    preexisting = []
    report = {"requested_period": [cfg["later_test_start"], cfg["later_test_end"]],
              "status": "INCOMPLETE", "errors": [], "sources": sources,
              "preexisting_files": preexisting, "settings_frozen_before_attempt": True}
    for crop in CROPS:
        for month in [4,5,6]:
            path = ROOT / f"data/raw/official/34_{crop}_2024_{month:02d}.json.gz"
            if path.exists():
                preexisting.append(str(path.relative_to(ROOT)))
            try:
                print(f"Later source attempt: crop={crop} month={month}", flush=True)
                source = fetch((34,crop,2024,month))
                sources.append(source)
                for row in rows(ROOT / source["file"]):
                    if (row["state"],row["market"]) in chosen:
                        future.append(row)
            except SourceRateLimit as error:
                report["errors"].append({"type":"rate_limit", "retry_after_seconds":error.seconds,
                    "action":"Stopped acquisition immediately; no cooldown bypass or indefinite retry"})
                save_json(OUT / "later_source_manifest.json",report)
                return None
            except Exception as error:  # noqa: BLE001 - bounded acquisition failure is recorded, not hidden
                report["errors"].append({"type":type(error).__name__, "message":str(error)})
                save_json(OUT / "later_source_manifest.json",report)
                return None
    extra = pd.DataFrame(future)
    if extra.empty or extra.duplicated(["state","market","commodity","date"]).any():
        report["errors"].append({"type":"invalid_or_empty_source"})
        save_json(OUT / "later_source_manifest.json",report)
        return None
    report.update({"status":"COMPLETE", "observations":len(extra),
        "newly_acquired_after_freeze":not bool(preexisting),
        "missingness":extra.isna().sum().to_dict(),
        "date_min":str(extra.date.min()), "date_max":str(extra.date.max())})
    extra.to_parquet(OUT / "later_observations.parquet",index=False)
    save_json(OUT / "later_source_manifest.json",report)
    return extra


def summarize_intervals(frame, cfg, period):
    output = []
    for (model, method), group in frame.groupby(["model", "method"]):
        for level, partitions in [("overall", [("all",group)]),
            ("commodity",group.groupby("commodity")), ("market",group.groupby("market")),
            ("month",group.groupby(group.date.dt.strftime("%Y-%m")))]:
            for key, sample in partitions:
                if len(sample) >= cfg["minimum_subgroup_n"]:
                    output.append({"period":period,"model":model,"method":method,
                        "level":level,"group":key,**interval_metrics(sample)})
    return output


def paired_bootstrap(intervals, cfg, period):
    rng=np.random.default_rng(cfg["seed"])
    output=[]
    for model, group in intervals.groupby("model"):
        parts={method:g.sort_values(["date","market","commodity"]).reset_index(drop=True)
               for method,g in group.groupby("method")}
        adaptive=parts["commodity_adaptive"]
        for reference in ["pooled_fixed","commodity_fixed"]:
            fixed=parts[reference]
            assert adaptive[["date","market","commodity"]].equals(fixed[["date","market","commodity"]])
            deltas=pd.DataFrame({"date":adaptive.date,
                "coverage":((adaptive.actual>=adaptive.lower)&(adaptive.actual<=adaptive.upper)).astype(float)
                    -((fixed.actual>=fixed.lower)&(fixed.actual<=fixed.upper)).astype(float),
                "width":(adaptive.upper-adaptive.lower)-(fixed.upper-fixed.lower),
                "interval_score":interval_score(adaptive.actual,adaptive.lower,adaptive.upper)
                    -interval_score(fixed.actual,fixed.lower,fixed.upper)})
            daily=deltas.groupby("date").agg({"coverage":["sum","count"],"width":"sum","interval_score":"sum"})
            n=len(daily);idx=rng.integers(0,n,size=(cfg["bootstrap_repeats"],n))
            denominator=daily[("coverage","count")].to_numpy()[idx].sum(axis=1)
            for metric in ["coverage","width","interval_score"]:
                draws=daily[(metric,"sum")].to_numpy()[idx].sum(axis=1)/denominator
                output.append({"period":period,"model":model,"comparison":"adaptive-minus-"+reference,
                    "metric":metric,"difference":float(deltas[metric].mean()),
                    "ci_low":float(np.quantile(draws,.025)),"ci_high":float(np.quantile(draws,.975)),
                    "date_blocks":n,"repeats":cfg["bootstrap_repeats"],
                    "limitation":"Whole dates preserve within-date market dependence; independent date-block resampling does not preserve serial dependence across dates; descriptive uncertainty only"})
    return output


def evaluate_period(panel, final, routing, rf_name, log_name, adaptive, cfg, label, start, end):
    test=panel.loc[eligible(panel)&(panel.date>=start)&(panel.date<=end)].copy()
    cal=panel.loc[eligible(panel)&(panel.date>=cfg["final_calibration_start"])&(panel.date<=cfg["final_calibration_end"])].copy()
    pc,pv={},{};timings={};point_rows=[]
    for name,fitted in final.items():
        pc[name],_=predict(fitted,cal);pv[name],elapsed=predict(fitted,test);timings[name]=elapsed
    baseline_times = {}
    started = time.perf_counter()
    rolling_prediction = test.rolling_mean.to_numpy().copy()
    baseline_times["rolling_mean"] = time.perf_counter() - started
    started = time.perf_counter()
    seasonal_prediction = test.weekly_seasonal.fillna(test.rolling_mean).to_numpy().copy()
    baseline_times["weekly_seasonal"] = time.perf_counter() - started
    forecasts={"rolling_mean":rolling_prediction,
        "weekly_seasonal":seasonal_prediction,
        "random_forest":pv[rf_name],"log_lightgbm":pv[log_name],
        "selected_lightgbm_bundle":bundle_predictions(pv,test,routing),
        "ablation_without_price":pv["ablation_without_price"],
        "ablation_without_seasonality":pv["ablation_without_seasonality"],
        "prespecified_log_full":pv["log_l2"]}
    cal_forecasts={"rolling_mean":cal.rolling_mean.to_numpy(),
        "weekly_seasonal":cal.weekly_seasonal.fillna(cal.rolling_mean).to_numpy(),
        "random_forest":pc[rf_name],"log_lightgbm":pc[log_name],
        "selected_lightgbm_bundle":bundle_predictions(pc,cal,routing),
        "ablation_without_price":pc["ablation_without_price"],
        "ablation_without_seasonality":pc["ablation_without_seasonality"],
        "prespecified_log_full":pc["log_l2"]}
    original_card = json.loads((ROOT / "artifacts/production/model_card.json").read_text(encoding="utf8"))
    original_routing = {crop: route["name"] for crop, route in original_card["routing"].items()}
    if set(original_routing.values()) <= set(final):
        forecasts["existing_lightgbm_bundle"] = bundle_predictions(pv, test, original_routing)
        cal_forecasts["existing_lightgbm_bundle"] = bundle_predictions(pc, cal, original_routing)
    model_names={"random_forest":[rf_name],"log_lightgbm":[log_name],
        "selected_lightgbm_bundle":sorted(set(routing.values())),
        "ablation_without_price":["ablation_without_price"],
        "ablation_without_seasonality":["ablation_without_seasonality"],
        "prespecified_log_full":["log_l2"]}
    model_names["existing_lightgbm_bundle"] = sorted(set(original_routing.values()))
    all_intervals=[];audits={}
    for name,prediction in forecasts.items():
        point=record(test,prediction)
        elapsed=sum(timings[n] for n in model_names.get(name,[])) if name in model_names else baseline_times[name]
        training=sum(final[n]["training_seconds"] for n in model_names.get(name,[]))
        for level,parts in [("overall",[("all",point)]),("commodity",point.groupby("commodity")),
            ("market",point.groupby("market")),("month",point.groupby(point.date.dt.strftime("%Y-%m")))]:
            for key,g in parts:
                if len(g)>=cfg["minimum_subgroup_n"]:
                    point_rows.append({"period":label,"model":name,"level":level,"group":key,
                        **metrics(g.actual,g.prediction),"training_seconds":training if level=="overall" else None,
                        "inference_seconds":elapsed if level=="overall" else None})
        point.to_parquet(OUT/f"{label}_{name}_point.parquet",index=False)
        for method in ["pooled_fixed","commodity_fixed","commodity_adaptive"]:
            parameters=adaptive.get(name,adaptive["log_lightgbm"])["selected"]
            result,audit=calibrate(record(cal,cal_forecasts[name]),point,method,
                **parameters,minimum_count=cfg["minimum_group_count"])
            result["method"]=method;result["model"]=name;all_intervals.append(result)
            audits[name+"/"+method]=audit
    intervals=pd.concat(all_intervals,ignore_index=True)
    intervals.to_parquet(OUT/f"{label}_intervals.parquet",index=False)
    save_json(OUT/f"{label}_calibration_audit.json",audits)
    return point_rows,summarize_intervals(intervals,cfg,label),paired_bootstrap(intervals,cfg,label),{
        "n":len(test),"date_min":str(test.date.min()),"date_max":str(test.date.max()),
        "missing_calendar_rows":int((~panel.loc[(panel.date>=start)&(panel.date<=end)].observed).sum())}


def plots(points, intervals):
    for period in intervals.period.unique():
        g=intervals.loc[(intervals.period==period)&(intervals.level=="overall")&
            (intervals.model=="log_lightgbm")]
        fig,axes=plt.subplots(1,2,figsize=(9,3.5))
        labels=g.method.str.replace("commodity_","crop_",regex=False)
        axes[0].bar(labels,g.coverage);axes[0].axhline(.95,color='red',ls='--',label='95% target')
        axes[0].set_ylabel('Empirical coverage');axes[0].set_ylim(0,1);axes[0].legend()
        axes[1].bar(labels,g.mean_width);axes[1].set_ylabel('Mean interval width (tonnes)')
        for ax in axes:ax.tick_params(axis='x',rotation=20)
        fig.suptitle(period+' | same log-LightGBM point predictions');fig.tight_layout()
        fig.savefig(OUT/'plots'/f'{period}_calibration.png',dpi=180);plt.close(fig)
    g=points.loc[(points.period=='inspected_2024Q1')&(points.level=='overall')]
    fig,ax=plt.subplots(figsize=(9,4));ax.barh(g.model,g.wape*100);ax.set_xlabel('WAPE (%)');fig.tight_layout()
    fig.savefig(OUT/'plots'/'point_comparison.png',dpi=180);plt.close(fig)


def main():
    cfg=json.loads((OUT/'config.json').read_text(encoding='utf8'))
    OUT.mkdir(exist_ok=True,parents=True)
    env={"python":sys.version,"platform":platform.platform(),"seed":cfg['seed'],
        "dependencies":{name:importlib.metadata.version(name) for name in
            ['numpy','pandas','scikit-learn','lightgbm','scipy','matplotlib','joblib']},
        "command":"python run.py scrc.research.benchmark",
        "started_utc":datetime.now(UTC).isoformat()}
    save_json(OUT/'environment.json',env)
    observations=pd.read_csv(ROOT/'data/processed/deployment_observations.csv',parse_dates=['date'])
    observations=observations.loc[observations.date<='2024-03-31'].copy()
    panel=build_features(observations);data=panel.loc[eligible(panel)].copy()
    specs=model_specs(cfg);validation=[];oof={};partitions=[]
    for fold,(start,end) in enumerate(cfg['folds']):
        cal_start=pd.Timestamp(start)-pd.Timedelta(days=cfg['fold_calibration_days'])
        train=data.loc[data.date<cal_start];cal=data.loc[(data.date>=cal_start)&(data.date<start)]
        val=data.loc[(data.date>=start)&(data.date<end)]
        partitions.append({'fold':fold,'train_n':len(train),'calibration_n':len(cal),'validation_n':len(val),
            'train_end':str(train.date.max()),'calibration_start':str(cal.date.min()),
            'calibration_end':str(cal.date.max()),'validation_start':start,'validation_end_exclusive':end})
        cache={'cal':cal,'val':val,'cal_predictions':{},'val_predictions':{}}
        for spec in specs:
            print(f"Validation fold {fold+1}/3: {spec['name']} ({len(train)} train rows)",flush=True)
            fitted=fit_predictor(train,spec,FEATURES,cfg['seed'])
            pc,_=predict(fitted,cal);pv,elapsed=predict(fitted,val)
            cache['cal_predictions'][spec['name']]=pc;cache['val_predictions'][spec['name']]=pv
            for crop in sorted(val.commodity.unique()):
                mask=val.commodity.to_numpy()==crop;y=val.loc[mask,'arrivals'].to_numpy()
                validation.append({'fold':fold,'name':spec['name'],'commodity':crop,
                    **metrics(y,pv[mask]),'absolute_error':float(np.abs(y-pv[mask]).sum()),
                    'actual_total':float(np.abs(y).sum()),'training_seconds':fitted['training_seconds'],
                    'inference_seconds':elapsed})
        cache['cal_predictions']['rolling_mean']=cal.rolling_mean.to_numpy()
        cache['val_predictions']['rolling_mean']=val.rolling_mean.to_numpy()
        cache['cal_predictions']['weekly_seasonal']=cal.weekly_seasonal.fillna(cal.rolling_mean).to_numpy()
        cache['val_predictions']['weekly_seasonal']=val.weekly_seasonal.fillna(val.rolling_mean).to_numpy()
        # Bundle is assembled after validation-only routing is frozen.
        cache['cal_predictions']['bundle']=None;cache['val_predictions']['bundle']=None
        oof[fold]=cache
    validation=pd.DataFrame(validation);validation.to_csv(OUT/'validation_search.csv',index=False)
    routing,rf_name,log_name=choose_point_models(validation,specs)
    adaptive={}
    for display,selected in [('rolling_mean','rolling_mean'),('weekly_seasonal','weekly_seasonal'),
        ('random_forest',rf_name),('log_lightgbm',log_name),('selected_lightgbm_bundle','bundle')]:
        print('Adaptive validation selection: '+display,flush=True)
        adaptive[display]=select_adaptive(oof,selected,routing,cfg)
    train=data.loc[data.date<=cfg['final_train_end']]
    cal=data.loc[(data.date>=cfg['final_calibration_start'])&(data.date<=cfg['final_calibration_end'])]
    original_card = json.loads((ROOT / "artifacts/production/model_card.json").read_text(encoding="utf8"))
    original_routing = {crop: route["name"] for crop, route in original_card["routing"].items()}
    needed=set(routing.values())|set(original_routing.values())|{rf_name,log_name,'log_l2'}
    final={}
    for spec in specs:
        if spec['name'] in needed:
            print('Final pre-acquisition fit: '+spec['name'],flush=True)
            fitted=fit_predictor(train,spec,FEATURES,cfg['seed']);final[spec['name']]=fitted
            joblib.dump(fitted,OUT/'models'/f"{spec['name']}.joblib")
    prespecified=next(s for s in specs if s['name']=='log_l2')
    for ablation,removed in cfg['ablations'].items():
        if ablation=='full':continue
        name='ablation_'+ablation
        fitted=fit_predictor(train,prespecified,[f for f in FEATURES if f not in removed],cfg['seed'])
        final[name]=fitted;joblib.dump(fitted,OUT/'models'/f'{name}.joblib')
    freeze={'existing_reference_routing':original_routing,'existing_reference_card_sha256':digest(ROOT/'artifacts/production/model_card.json'),'frozen_utc':datetime.now(UTC).isoformat(),'point_selection':{'routing':routing,
        'random_forest':rf_name,'log_lightgbm':log_name},'adaptive':adaptive,
        'model_specs':specs,'features':cfg['features'],'ablations':cfg['ablations'],
        'partitions':partitions,'final_training_n':len(train),'final_calibration_n':len(cal),
        'model_hashes':{p.name:digest(p) for p in (OUT/'models').glob('*.joblib')},
        'data_sha256':digest(ROOT/'data/processed/deployment_observations.csv'),
        'config_sha256':digest(OUT/'config.json'),'nominal_coverage':.95,
        'primary_calibration_comparison':'log_lightgbm',
        'ablation_calibration':'Fixed commodity calibration is primary; adaptive ablation intervals use the full log baseline settings without retuning',
        'new_period_rule':'Freeze before acquisition. New later test adaptive histories initialise from Oct-Dec calibration; no Jan-Mar residuals are consumed, though prior actual arrival/price features are available one day at a time.'}
    save_json(OUT/'frozen_selection.json',freeze)
    print('SELECTION AND CALIBRATION FROZEN; attempting later sources.',flush=True)
    later=acquire_later(cfg,observations)
    if later is not None:
        panel=build_features(pd.concat([observations,later],ignore_index=True))
    point_rows,interval_rows,paired,counts=evaluate_period(panel,final,routing,rf_name,log_name,
        adaptive,cfg,'inspected_2024Q1','2024-01-01','2024-03-31')
    provenance={'existing_data':{'file':'data/processed/deployment_observations.csv',
        'sha256':freeze['data_sha256'],'rows':len(observations),'source':'Government AGMARKNET',
        'period':'2021-01-01 to 2024-03-31','previously_inspected_test':True},
        'partitions':partitions,'inspected_2024Q1':counts}
    if later is not None:
        p,i,b,c=evaluate_period(panel,final,routing,rf_name,log_name,adaptive,cfg,
            'later_2024Q2','2024-04-01','2024-06-30')
        point_rows+=p;interval_rows+=i;paired+=b;provenance['later_2024Q2']=c
    points=pd.DataFrame(point_rows);intervals=pd.DataFrame(interval_rows)
    points.to_csv(OUT/'point_metrics.csv',index=False);intervals.to_csv(OUT/'interval_metrics.csv',index=False)
    pd.DataFrame(paired).to_csv(OUT/'paired_differences.csv',index=False)
    save_json(OUT/'dataset_provenance.json',provenance)
    plots(points,intervals)
    preserved=json.loads((OUT/'preserved_before.json').read_text(encoding='utf8'))
    changed=[name for name,sha in preserved.items() if digest(ROOT/name)!=sha]
    save_json(OUT/'preservation_check.json',{'unchanged':not changed,'changed':changed,'checked_files':len(preserved)})
    if changed:raise ValueError('Existing artifacts changed unexpectedly')
    save_json(OUT/'completed.json',{'status':'completed','finished_utc':datetime.now(UTC).isoformat(),
        'new_period_acquired':later is not None,'unexecuted_experiments':[],
        'files':{p.name:digest(p) for p in OUT.glob('*.csv')}})
    print(points.loc[points.level=='overall',['period','model','mae','rmse','wape','r2']].to_string(index=False),flush=True)
    print(intervals.loc[(intervals.level=='overall')&(intervals.model=='log_lightgbm')].to_string(index=False),flush=True)


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'logs/run.log').open('w',encoding='utf8') as logfile:  # noqa: SIM117
        with contextlib.redirect_stdout(Tee(sys.stdout,logfile)),contextlib.redirect_stderr(Tee(sys.stderr,logfile)):
            try:
                main()
            except BaseException as error:
                save_json(OUT/'failure.json',{'error':type(error).__name__,'message':str(error),
                    'traceback':traceback.format_exc(),'utc':datetime.now(UTC).isoformat()})
                raise
