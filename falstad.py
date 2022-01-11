"""Falstad circuit simulator importer

Imports data generated by the "Data Export" objects.
The header must be in "# time step <val> sec" format

Reurns a pandas.DataFrame with 'time' and 'voltage'.

See http://falstad.com/circuit/
    Draw -> Outputs and Labels -> Add Data Export
"""
import pandas

def read(fn) -> pandas.DataFrame or None:
    df = pandas.read_table(fn)
    hdr, val = df.columns[0].split('=', 1)
    hdr = hdr.strip()
    val, unit = val.strip().split(' ', 1)
    if hdr != '# time step' or unit != 'sec':
        return None
    val = float(val)

    df.columns = ['voltage']
    df.insert(0, 'time', df.index * val)
    return df

if __name__ == '__main__':
    import sys
    df = read(sys.argv[1])
    if df is None:
        sys.exit(1)
    df.to_csv(sys.stdout.buffer, index=False)
