"""Build honest rough paper and review guide from completed benchmark CSVs only."""
import json

import pandas as pd

from scrc.common import ROOT, save_json

OUT=ROOT/'research/supplyguard'
NAMES={'rolling_mean':'Rolling mean','weekly_seasonal':'Weekly seasonal','random_forest':'Random Forest',
       'log_lightgbm':'Log-LightGBM','selected_lightgbm_bundle':'Reselected LGBM bundle','existing_lightgbm_bundle':'Existing LGBM bundle',
       'prespecified_log_full':'Original log-LGBM / full','ablation_without_price':'Without price',
       'ablation_without_seasonality':'Without calendar'}
METHODS={'pooled_fixed':'Pooled fixed','commodity_fixed':'Commodity fixed','commodity_adaptive':'Commodity adaptive'}


def esc(text):
    return str(text).replace('\\','\\textbackslash{}').replace('&','\\&').replace('%','\\%').replace('_','\\_').replace('#','\\#')


def tabular(headers,rows,columns):
    return '\\begin{tabular}{'+columns+'}\n\\toprule\n'+' & '.join(headers)+' \\\\\n\\midrule\n'+'\n'.join(' & '.join(map(str,row))+' \\\\' for row in rows)+'\n\\bottomrule\n\\end{tabular}\n'


def markdown(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in rows)+'\n'


def main():
    done=json.loads((OUT/'completed.json').read_text(encoding='utf8'))
    assert done['status']=='completed'
    point=pd.read_csv(OUT/'point_metrics.csv');interval=pd.read_csv(OUT/'interval_metrics.csv')
    paired=pd.read_csv(OUT/'paired_differences.csv')
    freeze=json.loads((OUT/'frozen_selection.json').read_text(encoding='utf8'))
    source=json.loads((OUT/'later_source_manifest.json').read_text(encoding='utf8'))
    prov=json.loads((OUT/'dataset_provenance.json').read_text(encoding='utf8'))
    tests=json.loads((OUT/'TEST_RESULTS.json').read_text(encoding='utf8'))
    research_models=['rolling_mean','weekly_seasonal','random_forest','log_lightgbm','selected_lightgbm_bundle','existing_lightgbm_bundle','prespecified_log_full']
    periods=list(point.period.unique())
    fresh='later_2024Q2' in periods and source.get('newly_acquired_after_freeze',False)
    acq=('A later contiguous April--June 2024 cohort was newly acquired only after model and calibration settings were frozen.' if fresh else
         'No newly acquired untouched later test was established; the historical benchmark remains previously inspected.' if 'later_2024Q2' not in periods else
         'April--June 2024 was available locally before acquisition; it is not claimed as a newly acquired untouched test.')
    primary_period = 'later_2024Q2' if 'later_2024Q2' in periods else 'inspected_2024Q1'
    period_name = 'later April-June benchmark' if primary_period == 'later_2024Q2' else 'previously inspected first-quarter benchmark'
    q1=point.loc[(point.period==primary_period)&(point.level=='overall')&point.model.isin(research_models)]
    best=q1.sort_values('wape').iloc[0]
    primary=interval.loc[(interval.period==primary_period)&(interval.level=='overall')&(interval.model=='log_lightgbm')]
    fixed=primary.loc[primary.method=='commodity_fixed'].iloc[0]
    adaptive=primary.loc[primary.method=='commodity_adaptive'].iloc[0]
    narrative=(f"On the {period_name}, {NAMES[best['model']]} has the lowest WAPE among the recorded main and preserved-reference forecasts ({best.wape:.2%}). "
        f"For identical log-LightGBM point predictions, commodity fixed coverage is {fixed.coverage:.2%} with mean width {fixed.mean_width:.3f} tonnes; "
        f"commodity adaptive coverage is {adaptive.coverage:.2%} with width {adaptive.mean_width:.3f} tonnes. "
        f"Their mean interval scores are {fixed.mean_interval_score:.3f} and {adaptive.mean_interval_score:.3f}, respectively. "
        "These are empirical tradeoffs, not a universal superiority or causal claim. The later-period interval-score difference has a descriptive paired interval that includes zero.")
    report=['# SupplyGuard - executed research benchmark','',narrative,'',acq,'',
        '## Protocol','Rolling one-day-ahead wholesale market-arrival forecasts. Identical eligible rows and source information across learned models. Training-only preprocessing. Three expanding validation folds, each with a preceding 90-day calibration segment held out of proper training. Final fitting through September 2023; October-December calibration. Main nominal coverage 95%.','',
        '## Actual results']
    sections=[]
    for period in periods:
        label='Previously inspected January-March 2024' if period=='inspected_2024Q1' else 'Later April-June 2024'
        rows=[]
        for _,r in point.loc[(point.period==period)&(point.level=='overall')&point.model.isin(research_models)].iterrows():
            rows.append([NAMES[r['model']],f'{r.mae:.3f}',f'{r.rmse:.3f}',f'{r.wape:.2%}',f'{r.r2:.3f}',f'{r.training_seconds:.3f}',f'{r.inference_seconds:.4f}'])
        report+=['',f'### {label}',markdown(['Model','MAE (t)','RMSE (t)','WAPE','R2','Fit seconds','Inference seconds'],rows)]
        latex_rows=[[esc(r[0]),*r[1:3],r[3].replace('%','\\%'),r[4]] for r in rows]
        sections += ['\\subsection{'+label+'}',f'The evaluation contains {prov[period]["n"]:,} eligible observed targets. Missing calendar rows are excluded rather than imputed as zero.',
            '\\begin{table}[!ht]\\centering\\small\\caption{Point forecasts: '+label+'.}'+tabular(['Model','MAE','RMSE','WAPE','R$^2$'],latex_rows,'lrrrr')+'\\end{table}']
        ir=[]
        for _,r in interval.loc[(interval.period==period)&(interval.level=='overall')&(interval.model=='log_lightgbm')].iterrows():
            ir.append([METHODS[r['method']],f'{r.coverage:.2%}',f'{100*r.coverage_gap:+.2f}',f'{r.mean_width:.3f}',f'{r.mean_interval_score:.3f}'])
        report += ['#### Calibration around the same log-LightGBM predictions',markdown(['Calibration','Coverage','Gap (percentage points)','Width (t)','Interval score'],ir)]
        sections += ['\\begin{table}[!ht]\\centering\\small\\caption{Same point forecasts, 95\\% nominal intervals. Gap is in percentage points.}'+
            tabular(['Calibration','Cov.','Gap','Width','IS'],[[esc(r[0]),r[1].replace('%','\\%'),*r[2:]] for r in ir],'lrrrr')+'\\end{table}']
        group_rows=[]
        for _,r in interval.loc[(interval.period==period)&(interval.level=='commodity')&(interval.model=='log_lightgbm')].iterrows():
            group_rows.append([r['group'],METHODS[r['method']],int(r.n),f'{r.coverage:.2%}',f'{r.mean_width:.3f}',f'{r.mean_interval_score:.3f}'])
        report += ['#### Commodity calibration breakdown',markdown(['Commodity','Calibration','n','Coverage','Width (t)','Interval score'],group_rows)]
        sections += ['\\begin{table}[!ht]\\centering\\scriptsize\\caption{Commodity breakdown: '+label+'.}'+tabular(['Crop','Method','n','Cov.','Width'],
            [[esc(r[0]),esc(r[1].replace('Commodity ','Crop ')),r[2],r[3].replace('%','\\%'),r[4]] for r in group_rows],'llrrr')+'\\end{table}']
        ar=[]
        ablation_names=['prespecified_log_full','ablation_without_price','ablation_without_seasonality']
        for _,r in point.loc[(point.period==period)&(point.level=='overall')&point.model.isin(ablation_names)].iterrows():
            ir_=interval.loc[(interval.period==period)&(interval.level=='overall')&(interval.model==r['model'])&(interval.method=='commodity_fixed')].iloc[0]
            ar.append([NAMES[r['model']],f'{r.mae:.3f}',f'{r.rmse:.3f}',f'{r.wape:.2%}',f'{ir_.coverage:.2%}',f'{ir_.mean_width:.3f}'])
        report += ['#### Feature ablations: prespecified log-LightGBM; fixed commodity calibration',markdown(['Variant','MAE (t)','RMSE (t)','WAPE','Coverage','Width (t)'],ar)]
        sections += ['\\begin{table}[!ht]\\centering\\small\\caption{Feature ablations: same rows and fixed log-LightGBM settings.}'+tabular(['Variant','MAE','WAPE','Cov.','Width'],
            [[esc(r[0].replace('Prespecified full LGBM','Full')),r[1],r[3].replace('%','\\%'),r[4].replace('%','\\%'),r[5]] for r in ar],'lrrrr')+'\\end{table}']
        bs=paired.loc[(paired.period==period)&(paired.model=='log_lightgbm')]
        br=[[r.comparison,r.metric,f'{r.difference:.4f}',f'[{r.ci_low:.4f}, {r.ci_high:.4f}]'] for _,r in bs.iterrows()]
        report += ['#### Paired descriptive differences',markdown(['Comparison','Metric','Adaptive minus reference','Date-block 95% interval'],br)]
        report += ['Date blocks preserve markets within a date, but independent date resampling ignores serial dependence. These are descriptive uncertainty estimates, not rigorous dependent-time-series significance claims.']
        for reference in ['adaptive-minus-pooled_fixed','adaptive-minus-commodity_fixed']:
            rows_=bs.loc[bs.comparison==reference]
            parts=[]
            for _,r in rows_.iterrows():parts.append(f"{r.metric} difference {r.difference:.4f} [{r.ci_low:.4f}, {r.ci_high:.4f}]")
            sections += [esc(reference)+': '+esc('; '.join(parts))+'.']
    report += ['','## Calibration choice and audit','Adaptive choices are selected on validation interval scores subject to a 94% coverage screening rule; if none qualify, nearest nominal coverage then interval score. Final settings and full search appear in frozen_selection.json. Sparse groups fall back to initial pooled fixed calibration. No method sees its own target before interval formation.','',
        '## Feature removals',json.dumps(freeze['ablations'],indent=2),'',
        '## Limitations','One state and four crops; reporting latency is not verified; market arrivals are not demand; all operational crises are synthetic; logistics effects remain exploratory; no stock-out reduction, savings or causal improvement is inferred. Price and calendar removal do not remove arrival lags that indirectly retain seasonal information. The existing production routing is also independently refitted as a fixed reference. The reselected bundle reuses the existing selection approach/configurations but is refitted under the common new fold protocol, so it need not reproduce the old bundle routing. Existing 100%-acceptance gate ablations remain unchanged and do not prove gate effectiveness. No EnbPI comparison or new theoretical guarantee is implemented. Novelty remains uncertain.','',
        '## Execution evidence',f"{tests['tests']} tests passed. Existing production artifacts are hash-checked in preservation_check.json. Numerical model runs are completed; paper is a rough draft. Native compiler and rendering checks are reported separately."]
    (OUT/'RESULTS.md').write_text('\n'.join(report),encoding='utf8')
    # Standalone LaTeX: embedded references, no external bibliography required.
    tex=r'''\documentclass[conference]{IEEEtran}
\usepackage{amsmath,amssymb,booktabs,url}
\setlength{\tabcolsep}{3pt}
\title{SupplyGuard: Evaluating Commodity-Specific Adaptive Conformal Calibration for Agricultural Market-Arrival Forecasting}
\author{\IEEEauthorblockN{M.Tech Research Project}\IEEEauthorblockA{Rough draft for staff review; author and affiliation to be completed}}
\begin{document}\maketitle
\begin{abstract}
'''+esc(narrative+' '+acq)+r''' We compare pooled fixed, commodity fixed and rolling commodity adaptive calibration under a common scaled residual score and 95\% nominal target. The benchmark includes rolling and weekly baselines, Random Forest, log-LightGBM and a validation-selected LightGBM bundle, with controlled price and calendar feature ablations. Findings are empirical and limited to the recorded cohort. No new conformal theorem, verified novelty or operational causal benefit is claimed.
\end{abstract}
\begin{IEEEkeywords}Conformal calibration, agricultural market arrivals, time-series forecasting, uncertainty evaluation\end{IEEEkeywords}
\section{Introduction and Research Question}
Reliable planning requires uncertainty information as conditions change. A single point forecast does not express the range of plausible future arrivals. This study asks whether commodity-specific rolling adaptive calibration improves the coverage--width tradeoff relative to pooled and commodity-specific fixed calibration. We treat this as an empirical question and candidate domain contribution, not established methodological novelty. Market arrivals represent supply inflows, not retail demand or warehouse inventory.
\section{Related Work}
Gibbs and Candes introduce adaptive conformal inference for changing distributions, evaluated on financial volatility and election-night prediction \cite{gibbs2021}. We reuse its level-adjustment idea but add clipped, date-batched updates and residual windows; original guarantees do not automatically transfer. Romano et al. study group-conditional coverage \cite{romano2020}; commodity grouping here is an application of existing group calibration, not an invention. Xu and Xie's EnbPI addresses dynamic time-series intervals \cite{xu2021}; EnbPI is related work and is not an implemented comparator in this draft. The M5 retail forecasting competition highlights boosted-tree and combination methods \cite{makridakis2022}. Its sales targets and 28-day horizon differ from wholesale one-day market inflows. Our contribution is reproducible engineering and empirical evaluation; novelty beyond this remains uncertain.
\section{Dataset Provenance}
The completed government AGMARKNET cohort comprises 59,648 actual market-day observations in 16 Uttar Pradesh markets during 2021--2023 for rice, wheat, onion and potato. There are 144 cached monthly source responses with request parameters and SHA-256 hashes. Published daily total arrivals are counted once, avoiding repeated totals across varieties. Modal prices are arrivals-weighted across valid varieties, with median fallback for zero usable weights. Prices are INR per quintal and arrivals are tonnes. Market selection uses January 2021--June 2022 completeness only. Daily reindexing creates 10,374 unobserved calendar rows; targets and prices remain missing and are not replaced by zero. Another 4,965 January--March 2024 observations were previously evaluated. '''+esc(acq)+r'''
Source provenance and subgroup missingness are retained in machine-readable manifests. The larger four-state plan was not completed because of acquisition restrictions; conclusions cover one state. Historical acquisition in 2026 does not make the records current 2026 market observations.
\section{Methodology}
\subsection{Point Forecasting}
We compare rolling mean, weekly seasonal mean, RandomForestRegressor, log-target LightGBM and a commodity-routed LightGBM bundle. Eight LightGBM specifications are configurations of one model family, not eight algorithms. Random Forest compares raw and log targets with 120 trees, depth 18, minimum leaf size 3 and maximum feature fraction 0.7. Its numeric median imputer and one-hot category encoder fit only on proper training data. LightGBM uses native missing-value and category handling. All receive the same source information and eligible target rows; representation differs. Log targets are inverted with expm1, ratio targets are rescaled with the past mean, and predictions are clipped below at zero.
\subsection{Feature Information}
The 34 inputs contain calendar-day arrival lags at 1, 2, 3, 7, 14, 21 and 28 days; past rolling mean and standard deviation; prior price; 7/14/28-day means, medians and standard deviations; weekly seasonal lag average; 7/28-day price means; trend and recent ratios; weekday, month and day-of-year; sine/cosine weekday and annual seasonality; and training-frozen state, market and commodity codes. All rolling features shift outcomes by one day before calculation. Unknown vocabulary values encode as -1; Random Forest ignores unknown one-hot categories. Fuel, warehouse stock, strikes and floods are not model inputs.
\subsection{Common Calibration Score}
For actual arrival $y_i$, point prediction $\hat y_i$ and past-only scale $s_i=\max(\overline y_{i,28},1)$, define
\begin{equation}r_i=|y_i-\hat y_i|/s_i.\end{equation}
The fixed split quantile uses order rank $\lceil(n+1)(1-\alpha)\rceil$ with $\alpha=0.05$. Intervals are $[\max(0,\hat y_i-q_i s_i),\hat y_i+q_i s_i]$. Pooled fixed uses all calibration residuals; commodity fixed uses crop-specific residuals. The existing state/commodity grouping reduces to commodity grouping in this one-state cohort and is reused transparently.
\subsection{Date-Batched Adaptive Calibration}
Commodity adaptive calibration retains the last $W$ residual observations per crop. The validation grid uses $W\in\{250,500\}$ and $\gamma\in\{0.005,0.02\}$. All observations in date batch $t$ receive intervals before any outcomes in that batch update the histories. After revelation,
\begin{equation}\alpha_{c,t+1}=\operatorname{clip}\big(\alpha_{c,t}+\gamma(0.05-\overline m_{c,t}),0.005,0.2\big),\end{equation}
where $\overline m$ is the crop's batch miss rate. Fewer than 100 group residuals or an unsupported order rank triggers fallback to the original pooled 95\% quantile. Clipping and fallback counts are recorded. The rolling, clipped and batched implementation is ACI-inspired; no general coverage guarantee is claimed.
\section{Experimental Protocol}
The task is rolling one-day-ahead prediction using observations strictly before each target date. It assumes earlier reporting is available and does not evaluate multi-day forecasts from one frozen origin. Three expanding validation quarters in 2023 each use a preceding 90-day calibration segment withheld from that fold's proper training data. All learned configurations use identical partitions. Model selection minimizes mean validation WAPE plus 0.25 times its across-fold standard deviation; crop routing is selected per commodity. Random Forest and log-LightGBM choices use global validation aggregates.
Adaptive settings minimize validation mean interval score among configurations with mean coverage at least 94\%; otherwise closest nominal coverage is chosen, then interval score. Final proper training ends September 2023, with October--December calibration. Settings and model hashes freeze before later acquisition. January--March 2024 is explicitly previously inspected. Later evaluation starts adaptive histories from the same October--December pool, not January--March residuals; prior observed arrivals remain available as forecasting features.
\subsection{Metrics and Ablations}
Point metrics are MAE and RMSE in tonnes, WAPE and $R^2$. WAPE is an error measure, not accuracy. Interval metrics are empirical coverage, signed gap from 95\%, mean width and mean interval score:
\begin{align}IS_\alpha(l,u;y)={}&(u-l)+\frac{2}{\alpha}(l-y)\mathbf1\{y<l\}\nonumber\\&+\frac{2}{\alpha}(y-u)\mathbf1\{y>u\}.\end{align}
A prespecified log-LightGBM (15 leaves, 400 trees) is retrained with full features, without price\_lag/price\_mean\_7/price\_mean\_28, and without weekday/month/year\_day plus four cyclic features. Arrival lags still retain indirect seasonality. Rows and remaining settings stay identical. Fixed commodity calibration is the primary ablation comparison; inherited adaptive settings are secondary. Results are saved overall and by crop, market and month, with at least 30 subgroup observations.
Paired differences use 300 resamples of entire dates, retaining all markets within each date. This preserves within-date dependence but treats date blocks as independent and ignores serial dependence. Bootstrap intervals are descriptive; an interval-score difference whose interval includes zero does not establish clear superiority. Training and inference times are measured on one machine, excluding common feature construction, and are not speed guarantees.
\section{Executed Results}
'''+esc(narrative)+'\n'+'\n'.join(sections)+r'''
\section{Discussion and Limitations}
Coverage and width must be interpreted together. On the later period, adaptive calibration improves coverage by widening intervals substantially; its mean interval-score reduction has a date-block interval crossing zero. This result supports a tradeoff, not clear dominance. A narrower interval can arise from undercoverage, and lower aggregate error can hide commodity or market weaknesses. A validation winner need not remain a test winner. The inspected quarter cannot provide a new independent generalization claim. Even a newly acquired later period is limited to the same region and reporting process. Windows and model searches are small and budget-limited. Reporting latency, calendar dependence and group sparsity remain important practical issues. Independent date-block resampling understates some temporal dependence uncertainty.
\subsection{Supporting Components}
The webpage's main and regional warehouse inventory, consumption, fuel prices, strikes and floods are deterministic synthetic scenarios. Its coloured response scores are heuristic feasibility assessments, not calibrated success probabilities. Delhivery logistics records come from an educational mirror and support separate exploratory observational analysis; there is no verified shipment join to market data. Causal identification remains unsupported, and real actions stay withheld. Existing predictive gate ablations accepted 100\% of historical predictions and demonstrate no selective benefit. No demand forecasting, stock-out reduction, savings, disaster prediction or causal improvement is inferred from market-arrival metrics.
\subsection{Unfinished Work}
This is a rough staff-review draft, not a publication-ready paper. Author/affiliation, broader literature coverage, external-region validation, reporting-delay audit, serial-dependence-aware inference, additional time-series calibration comparators and prospective business evaluation remain unfinished. We do not claim first use, novelty, a new theorem or universal superiority.
\section{Conclusion}
SupplyGuard provides an executed, reproducible comparison of fixed pooled, fixed commodity and rolling adaptive commodity calibration around common one-day-ahead agricultural market forecasts. The findings should be read as measured coverage--width--score tradeoffs within the documented cohort. Negative and inconclusive outcomes remain part of the record. The working application and earlier model artifacts are preserved separately.
\begin{thebibliography}{9}
\bibitem{gibbs2021} I. Gibbs and E. Candes, ``Adaptive Conformal Inference Under Distribution Shift,'' Advances in Neural Information Processing Systems, vol. 34, 2021.
\bibitem{romano2020} Y. Romano, R. F. Barber, C. Sabatti, and E. Candes, ``With Malice Toward None: Assessing Uncertainty via Equalized Coverage,'' Harvard Data Science Review, vol. 2, no. 2, 2020.
\bibitem{xu2021} C. Xu and Y. Xie, ``Conformal prediction interval for dynamic time-series,'' Proceedings of ICML, PMLR, vol. 139, pp. 11559--11569, 2021.
\bibitem{makridakis2022} S. Makridakis, E. Spiliotis, and V. Assimakopoulos, ``M5 accuracy competition: Results, findings, and conclusions,'' International Journal of Forecasting, vol. 38, no. 4, pp. 1346--1364, 2022, doi:10.1016/j.ijforecast.2021.11.013.
\end{thebibliography}\end{document}
'''
    (OUT/'paper/supplyguard_rough_draft.tex').write_text(tex,encoding='utf8')
    refs=json.loads((OUT/'related_work.json').read_text(encoding='utf8'))
    bib=[]
    for r in refs:
        bib.append('@misc{'+r['key']+',\n  author = {'+r['authors']+'},\n  title = {'+r['title']+'},\n  year = {'+str(r['year'])+'},\n  howpublished = {'+r['venue']+'},\n  url = {'+r['url']+'}\n}')
    (OUT/'paper/references.bib').write_text('\n\n'.join(bib),encoding='utf8')
    review=['# Beginner staff-review guide - updated executed benchmark','',
        '## A short explanation','We predict tomorrow\'s wholesale market arrivals from historical arrivals, prices and calendar patterns. We then compare three ways of forming prediction intervals, which express uncertainty. Our research question is whether crop-specific adaptive calibration improves the balance between coverage and width. Warehouse crisis displays remain a synthetic supporting demonstration.','',
        '## Real and synthetic data','AGMARKNET is real historical government market data. Delhivery comes from a historical educational mirror and is analysed separately. Main/regional stock, demand, fuel, strikes and floods are synthetic. Arrivals are not customer demand.','',
        '## What each model does','Rolling mean averages recent arrivals. Weekly seasonal uses observations from the same weekday in earlier weeks. Random Forest averages many trees; the new implementation trains median imputation and categorical encoding only on training data. LightGBM builds trees sequentially to reduce prediction error. The routed bundle chooses a validated global LightGBM configuration for each crop; its eight candidates are configurations of one algorithm.','',
        '## Why LightGBM, and what the benchmark says',
        'LightGBM was originally a practical choice for nonlinear tabular histories, missing values and categories. Random Forest was not previously implemented; it is now an executed benchmark, not an invented past result.',narrative,'',
        '## Features','34 source features: seven arrival lags; rolling mean/std; historical price features; arrival mean/median/std windows; weekly seasonal average; trend/recent ratios; calendar/cyclical fields; state/market/crop codes. Fuel, strikes and inventory do not enter this forecaster. Exact names are in config.json and scrc/production/features.py.','',
        '## Intervals and calibration','A prediction is one expected number. An interval gives a lower and upper range. Coverage is the fraction of actual arrivals inside that range. Width shows how broad the range is. Pooled calibration shares errors across crops; commodity fixed uses separate crop errors; commodity adaptive updates crop error histories after each whole day. We aim for 95%, but empirical results may miss it. Narrower alone is not better.','',
        '## Actual tables','See RESULTS.md for all actual metrics, timing and ablations. Detailed market and monthly results are in the CSV files.','',
        '## Honest answers','**Is Random Forest used?** Yes, now in the separate new research benchmark; the production app remains unchanged LightGBM.','**What is the target?** Market arrivals in tonnes, not demand or stock-out probability.','**Is adaptive calibration new?** No; we reuse existing methods and evaluate them on our cohort. No new theorem is claimed.','**Was fixed commodity calibration already implemented?** Yes; the original state/crop grouping becomes crop-only in this one-state cohort.','**Is the test untouched?** January-March 2024 was already inspected. '+acq,'**Can I call WAPE accuracy?** No. MAE, RMSE, WAPE and R2 are regression metrics.','**Does green guarantee a solution?** No; it passes synthetic scenario checks, not a measured real-world success probability.','**Does this prove causal savings?** No. Logistics is observational and warehouse activity is simulated.','**What did ablations prove?** They measure error changes after removing specified feature groups under the same data/settings; they do not establish causality.','**What is unfinished?** Broader external validation, real business feeds, measured solution outcomes, a deeper novelty review and publication polishing.','',
        '## Review opening','“I strengthened the project with an executed fair benchmark, including Random Forest, controlled feature ablations and three calibration methods. I use historical government market arrivals, preserve chronological timing, and report coverage with interval width and score. I do not claim guaranteed coverage, algorithmic novelty or real causal benefit. The paper is an honest rough draft.”','',
        f"## Checks\n{tests['tests']} tests passed; see TEST_RESULTS.json. Existing artifacts are hash-preserved."]
    (OUT/'BEGINNER_REVIEW_GUIDE.md').write_text('\n\n'.join(review),encoding='utf8')
    save_json(OUT/'paper/draft_status.json',{'status':'rough_draft','generated_from':'completed executed CSVs',
        'novel_method_claimed':False,'fresh_later_period':fresh,'author_affiliation':'unfinished',
        'compiler_status':'pending','unfinished_work':['external cohort','latency audit','serial dependence-aware inference','publication polishing']})
    print('Paper, actual-results report and beginner guide generated from completed results.')


if __name__=='__main__':
    main()
