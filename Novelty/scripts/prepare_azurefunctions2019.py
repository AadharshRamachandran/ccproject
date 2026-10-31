"""Aggregate the public Azure Functions 2019 invocation trace by minute.

The official release has one CSV per day and one row per function. Minute
columns 1..1440 contain invocation counts. This converter sums functions into
one chronological workload series without unpacking the full 136 MB archive.
"""
from __future__ import annotations

import argparse
import io
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd


class _NonSeekableReader(io.RawIOBase):
    """Adapt tarfile's streaming member reader for TextIOWrapper."""
    def __init__(self, source):
        self.source = source

    def readable(self):
        return True

    def seekable(self):
        return False

    def readinto(self, buffer):
        data = self.source.read(len(buffer))
        if not data:
            return 0
        buffer[:len(data)] = data
        return len(data)


def aggregate(archive: Path, days: int = 14, chunksize: int = 1000) -> np.ndarray:
    if not 1 <= days <= 14:
        raise ValueError('--days must be between 1 and 14')
    minute_columns = [str(i) for i in range(1, 1441)]
    series = {}
    pending = set(range(1, days + 1))
    with tarfile.open(archive, mode='r|xz') as tar:
        for member in tar:
            name = Path(member.name).name
            if not member.isfile() or not name.startswith('invocations_per_function_md.anon.d'):
                continue
            try:
                day = int(name.removesuffix('.csv').rsplit('d', 1)[1])
            except (IndexError, ValueError):
                continue
            if day not in pending:
                continue
            stream = tar.extractfile(member)
            if stream is None:
                raise OSError(f'Could not read {name} from {archive}')
            daily = np.zeros(1440, dtype=np.int64)
            buffered = io.BufferedReader(_NonSeekableReader(stream))
            with io.TextIOWrapper(buffered, encoding='utf-8', newline='') as text:
                chunks = pd.read_csv(text, chunksize=chunksize)
                for frame in chunks:
                    missing = [column for column in minute_columns
                               if column not in frame.columns]
                    if missing:
                        raise ValueError(f'{name} is missing minute column {missing[0]}')
                    values = frame[minute_columns].apply(pd.to_numeric, errors='coerce')
                    daily += values.fillna(0).to_numpy(dtype=np.int64).sum(axis=0)
            series[day] = daily
            pending.remove(day)
            print(f'Aggregated day {day:02d}: {int(daily.sum()):,} invocations', flush=True)
            if not pending:
                break
    if pending:
        raise FileNotFoundError(f'Missing invocation trace days: {sorted(pending)}')
    return np.concatenate([series[day] for day in range(1, days + 1)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--days', type=int, default=14)
    parser.add_argument('--chunksize', type=int, default=1000)
    parser.add_argument('--out', default='Novelty/data/processed/azurefunctions2019_minute.csv')
    args = parser.parse_args()
    values = aggregate(args.archive, args.days, args.chunksize)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'request_rate': values}).to_csv(output, index=False)
    print(f'Wrote {len(values):,} minute samples to {output.resolve()}; '
          f'total={int(values.sum()):,}, peak={int(values.max()):,}', flush=True)


if __name__ == '__main__':
    main()
