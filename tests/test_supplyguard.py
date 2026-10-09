"""Timing, no-peeking and calibration mathematics checks for isolated research."""
import numpy as np
import pandas as pd
import pytest

from scrc.production.features import FEATURES, build_features
from scrc.research.benchmark import fit_predictor, predict
from scrc.research.calibration import calibrate, interval_score


def frames():
    dates=pd.date_range('2023-01-01',periods=200)
    cal=pd.DataFrame({'date':dates,'market':'A','commodity':'Rice',
        'actual':np.arange(200)%10+1.,'prediction':5.,'scale':1.})
    test=pd.DataFrame({'date':pd.to_datetime(['2024-01-01','2024-01-01','2024-01-02']),
        'market':['A','B','A'],'commodity':'Rice','actual':[6.,100.,8.],
        'prediction':5.,'scale':1.})
    return cal,test


def test_interval_score_penalties():
    np.testing.assert_allclose(interval_score([5,0,10],[2,2,2],[8,8,8],.05),[6,86,86])


def test_complete_batch_predicted_before_update():
    cal,test=frames()
    first,_=calibrate(cal,test,'commodity_adaptive',window=250,gamma=.02)
    altered=test.copy();altered.loc[1,'actual']=10000
    second,_=calibrate(cal,altered,'commodity_adaptive',window=250,gamma=.02)
    np.testing.assert_allclose(first.loc[:1,['lower','upper']],second.loc[:1,['lower','upper']])
    assert (pd.to_datetime(first.latest_calibration_date)<first.date).all()
    assert first.alpha_used.iloc[2]<first.alpha_used.iloc[0]


def test_group_fallback_and_fixed_not_updated():
    cal,test=frames();test['commodity']='Unseen'
    result,audit=calibrate(cal,test,'commodity_fixed',minimum_count=100)
    assert audit['fallback_rows']==len(test)
    assert np.isfinite(result.upper).all()
    assert result.upper.nunique()==1


def test_calibration_overlap_rejected():
    cal,test=frames();test.loc[0,'date']=cal.date.max()
    with pytest.raises(ValueError,match='strictly precede'):
        calibrate(cal,test,'commodity_fixed')


def observations():
    n=90
    return pd.DataFrame({'date':pd.date_range('2022-01-01',periods=n),'state':'S',
        'market':'M','commodity':'Rice','arrivals':np.arange(n,dtype=float)+1,'modal_price':100.})


def test_calendar_lags_do_not_skip_missing_days_or_peek():
    raw=observations().drop(index=40)
    before=build_features(raw)
    # January 1 + 41 days: prior day was missing, not the last recorded row.
    day=pd.Timestamp('2022-02-11')
    assert np.isnan(before.loc[before.date==day,'lag_1'].iloc[0])
    changed=raw.copy();changed.loc[changed.date>=day,['arrivals','modal_price']]=99999
    after=build_features(changed)
    features=[f for f in FEATURES if not f.endswith('_code')]
    pd.testing.assert_frame_equal(before.loc[before.date==day,features].reset_index(drop=True),
        after.loc[after.date==day,features].reset_index(drop=True))
    assert before.loc[before.date==pd.Timestamp('2022-02-12'),'lag_2'].isna().all()


def test_rf_imputation_and_vocabulary_train_only():
    frame=build_features(observations());train=frame.loc[frame.date<'2022-03-01'].copy()
    spec={'name':'test_rf','family':'random_forest','target':'raw','n_estimators':5,
        'max_depth':3,'min_samples_leaf':2,'max_features':.7}
    fitted=fit_predictor(train,spec,FEATURES)
    imputer=fitted['model'].named_steps['transform'].named_transformers_['numeric']
    frozen=imputer.statistics_.copy()
    future=frame.loc[frame.date>='2022-03-01'].copy();future['market']='Unknown'
    future['lag_1']=999999
    result,_=predict(fitted,future)
    np.testing.assert_array_equal(frozen,imputer.statistics_)
    assert fitted['vocabulary']['market']==['M']
    assert np.isfinite(result).all()
