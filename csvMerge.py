import argparse
import sys
import pandas as pd

def trimTS(df, ts_min, ts_max):
    """ trims rows from DataFrame that are not within ts_min and ts_max
        Parameters:
            df - DataFrame to be trimmed
            ts_min - minimum timestamp to keep
            ts_max - maximum timestamp to keep
    """
    return df.loc[(df.index >= ts_min) & (df.index <= ts_max)]

def filterTS(df, ts):
    """ trims rows from DataFrame with timestamps that are not in ts
        Parameters:
            df - the DataFrame to be trimmed
            ts - the timestamps to keep
        Returns:
            trimmed DataFrame
    """
    return df.loc[df.index.isin(ts)]

parser = argparse.ArgumentParser(description='Merge csv.')
parser.add_argument('file1', nargs=1, type=argparse.FileType('r'),
                    help='first file to be merged')
parser.add_argument('file2', nargs=1, type=argparse.FileType('r'),
                    help='second file to be merged')
parser.add_argument('--output', nargs='?', type=argparse.FileType('w'), default=sys.stdout,
                    help='output file for merged csv, default is standart output (screen)')
parser.add_argument('--format', nargs='?', choices=['c','i'], default='i',
                    help='specifies format of data merged from file2 as copy (c) or interpolate (i), default is interpolate')
parser.add_argument('--no-trim', action='store_true')

args = vars(parser.parse_args())

file1_df = pd.read_csv(args['file1'][0],index_col='time', parse_dates=True)
file2_df = pd.read_csv(args['file2'][0], index_col='time', parse_dates=True)


# Calculate overlapping timestamp interval [ts_min, ts_max]
ts1_min, ts1_max = file1_df.index[0], file1_df.index[-1]
ts2_min, ts2_max = file2_df.index[0], file2_df.index[-1]
ts_min = max(ts1_min, ts2_min)
ts_max = min(ts1_max, ts2_max)

ts = file1_df.index

mergedDf = (
    pd.concat([file1_df, file2_df])
      .sort_values(by='time')
)

if args['format'] == 'i':
    # Performs interpolate on only the numeric columns to avoid exception by .interpolate()
    numeric_cols = mergedDf.select_dtypes(exclude=['object']).columns
    mergedDf[numeric_cols] = mergedDf[numeric_cols].interpolate(method='time')
else:
    # Fills in NaN values using last valid value in column
    mergedDf = mergedDf.ffill()

# If --no-trim was specified, don't trim dataframe
if not args['no_trim']:
   mergedDf = trimTS(mergedDf, ts_min, ts_max)
# Only keeps timestamps that are in file1
mergedDf = filterTS(mergedDf, ts)

print(mergedDf.head(5))
mergedDf.to_csv(path_or_buf=args['output'], float_format='%.10f')
