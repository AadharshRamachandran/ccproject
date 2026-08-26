"""Trace loaders, including the World Cup '98 repository's count format."""
import pandas as pd


def worldcup98_to_workload(frame, bucket='min', start=None, end=None, scale_max=None):
    """Convert ``invocation_count.csv`` to a chronological request-rate series.
    The source repository stores a ``count`` field and a period/timestamp column,
    rather than this project's required ``request_rate`` field.  Timestamp formats
    differ between its generated outputs, so we accept a parseable ``period`` or
    index-like first column and aggregate all events into complete minute buckets.
    Missing minutes are zero-filled: dropping them would leak future load spacing
    into the forecaster and make adaptive-monitoring results misleading.
    """
    if 'count' not in frame.columns:
        raise ValueError("World Cup input must contain a 'count' column")
    timestamp = None
    candidates = ['period'] + [c for c in frame.columns if str(c).lower().startswith('unnamed')]
    for column in candidates:
        if column not in frame.columns: continue
        parsed = pd.to_datetime(frame[column], errors='coerce')
        if parsed.notna().mean() >= .9:
            timestamp = parsed
            break
    if timestamp is None:
        raise ValueError("Could not find a parseable timestamp. Expected a 'period' or CSV index column.")
    result = pd.DataFrame({'timestamp':timestamp,'request_rate':pd.to_numeric(frame['count'],errors='coerce').fillna(0.)})
    result=result.dropna(subset=['timestamp']).groupby(pd.Grouper(key='timestamp',freq=bucket))['request_rate'].sum().asfreq(bucket,fill_value=0.).reset_index()
    if start is not None: result=result[result.timestamp>=pd.Timestamp(start)]
    if end is not None: result=result[result.timestamp<=pd.Timestamp(end)]
    if result.empty: raise ValueError('Selected World Cup time range contains no observations')
    if scale_max is not None:
        if scale_max <= 0: raise ValueError('scale_max must be positive')
        peak=result.request_rate.max()
        if peak > 0: result['request_rate']=result.request_rate/peak*float(scale_max)
    return result.reset_index(drop=True)

def load_workload_csv(path, column='request_rate'):
    frame=pd.read_csv(path)
    if column not in frame:
        if 'count' in frame.columns: frame=worldcup98_to_workload(frame)
        else: raise ValueError(f'{path} must contain a {column!r} column (or World Cup 98 count/period columns)')
    return frame[column].astype(float).clip(lower=0).tolist()

def chronological_split(values, train_fraction=.8):
    cut=int(len(values)*train_fraction); return values[:cut],values[cut:]
