"""WAVE audio file importer

Imports samples from .wav audio file format.

Returns a pandas.DataFrame with 'time' and multiple 'ch<n>' columns.
"""
import wave
import numpy as np
import pandas as pd

TIME_COL_NAME = 'time'
CHN_COL_FMT = 'ch%d'
# WAV-file samples are big-endian signed integers
WAVE_DTYPE_FMT = '<i%d'

def read(fn, time_index=False) -> pd.DataFrame or None:
    """Import data from WAVE audio file"""
    with wave.open(fn, mode='rb') as wav_file:
        # Read samples in numpy array
        frames = np.frombuffer(wav_file.readframes(wav_file.getnframes()),
                dtype=WAVE_DTYPE_FMT%wav_file.getsampwidth())
        frames = frames.reshape(-1, wav_file.getnchannels())

        # Convert to pandas DataFrame
        columns = [CHN_COL_FMT%i for i in range(frames.shape[1])]
        df = pd.DataFrame(frames, columns=columns)
        # Add time (index) column
        df.insert(0, TIME_COL_NAME, df.index / wav_file.getframerate())

        # Use 'time' as index
        if time_index:
            df = df.set_index(TIME_COL_NAME)
    return df

def write(fn, df: pd.DataFrame, sampwidth: int or None=None) -> bool:
    """Export data to WAVE audio file"""
    nchannels = len(df.columns)
    # Time can be separate column or index
    if TIME_COL_NAME in df.columns:
        nchannels -= 1
        df = df.copy()
        time_col = df.pop(TIME_COL_NAME)
        framerate = round((time_col.size - 1) / (time_col.iloc[-1] - time_col.iloc[0]))
    else:
        framerate = round((df.index.size - 1) / (df.index[-1] - df.index[0]))
    # Select sample width
    if sampwidth is None:
        # The bit-values must be symmetric around zero
        maxval = max(df.values.max(), -df.values.min())
        sampwidth = int(np.log2(maxval)) + 1 + 1
        sampwidth = (sampwidth + 7) // 8
        sampwidth = 1 << int(np.ceil(np.log2(sampwidth)))
        assert 1<<8*sampwidth > maxval*2, \
                f'The auto-selected sample-width ({sampwidth} bits) is insufficient:\n'\
                f'  values {df.values.min()} to {df.values.max()}'
    frames = np.asarray(df.values, dtype=WAVE_DTYPE_FMT%sampwidth)

    with wave.open(fn, mode='wb') as wav_file:
        wav_file.setnchannels(nchannels)
        wav_file.setsampwidth(sampwidth)
        wav_file.setframerate(framerate)
        wav_file.writeframes(frames.tobytes())
    return True

if __name__ == '__main__':
    import sys
    df = read(sys.argv[1], True)
    if df is None:
        sys.exit(1)
    df.to_csv(sys.stdout.buffer)
