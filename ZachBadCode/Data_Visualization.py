'''
Visualize data in several ways.
DF is not availabe here because each object has a different length. 
the many outputs could be collapsed into a single array.
'''

import Data_Preparation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def crash_map(df):
    ''' Visualize crashes. '''
    plt.scatter(x=Latitude, y=Longitude,
                ec='k',
                marker="o",
                linewidths=0,
                alpha=0.05)

def crash_date_map(df):
    '''
    Attempting to sort by crash time of year.
    This assumes that crashes happen because of the weather conditions.
    Uses 'Crash Month' col
    '''
    month = df['Crash Month']
    month = month.to_numpy(dtype=int)
    # create 12 images.
    for i in range(11):
        print(i+1)
        print(month)
        if i == month:
            plt.scatter(x=Latitude, y=Longitude, c=month,
                        ec='k',
                        marker="o",
                        linewidths=0,
                        alpha=0.05)

def original_entry_keys(df, visualize=False):
    '''
    For each column, show each orignial entry,
    as well as how many times that entry appears. 
    '''
    key_data = []
    count_data = []
    cols = []
    for col in df:
        cols.append(col)
        keys = [i for i in df[col].value_counts(normalize=False, dropna=False).keys()]
        key_data.append(keys)
        counts = df[col].value_counts(normalize=True, dropna=False).tolist()
        count_data.append(counts)
        if visualize:
            print(len(df[col].value_counts(dropna=False)))
            print(df[col].value_counts(dropna=False)[0:10])
            print(df[col].value_counts(dropna=False))
            print()
    return key_data, count_data, cols


def normalized_instance_counts(df):
    ''' Given good/bad crash data, find original entries and enumerate.
        Normalize defined in original_entry_keys. '''

    # get dangerous crash data via iloc. Clean up later.
    df_good_crash = df[df['Person Injury Severity'].str.contains("A - SUSPECTED SERIOUS INJURY")==False]
    df_good_crash = df_good_crash[df_good_crash['Person Injury Severity'].str.contains("K - FATAL INJURY")==False]
    
    df_bad_crash = df[df['Person Injury Severity'].str.contains("N - NOT INJURED")==False]
    df_bad_crash = df_bad_crash[df_bad_crash['Person Injury Severity'].str.contains("99 - UNKNOWN")==False]
    df_bad_crash = df_bad_crash[df_bad_crash['Person Injury Severity'].str.contains("B - SUSPECTED MINOR INJURY")==False]
    df_bad_crash = df_bad_crash[df_bad_crash['Person Injury Severity'].str.contains("C - POSSIBLE INJURY")==False]

    # get deltas in counts
    keys_good, counts_good, _ = original_entry_keys(df=df_good_crash, visualize=False)
    keys_bad, counts_bad, _ = original_entry_keys(df=df_bad_crash, visualize=False)
    return keys_good, counts_good, keys_bad, counts_bad


def find_count_differences(keys_good, counts_good, keys_bad, counts_bad, cols, tolerance):
    ''' Given counts, find elements with largest difference. 
        Then find their respective column, so it can be plotted later. 
        ignore differences if tolerance it too small.'''
    counts_good = np.array(counts_good)
    counts_bad = np.array(counts_bad)

    # I would like to apologize in advance for this monstrosity of for-loops.
    plotting_cols = []
    for idx, col_info in enumerate(keys_good):
        for bad_idx, bad_key in enumerate(keys_bad[idx]):
            for good_idx, good_key in enumerate(keys_good[idx]):
                if good_key == bad_key:
                    if abs(counts_good[idx][good_idx] - counts_bad[idx][bad_idx]) > tolerance:
                        plotting_cols.append(cols[idx])
                        # print(keys_bad[idx][good_idx])
                        # print(cols[idx])
                        # print()
    return plotting_cols


def bar_graphs(key_data, keys_good, counts_good, keys_bad, counts_bad, cols, plotting_cols):
    ''' Use plotly to show cols with biggest deltas btween bad, good crashes. 
        Using: https://community.plotly.com/t/setting-color-scheme-on-bar-chart-grouped-by-two-columns/34801
               https://github.com/plotly/plotly.js/issues/1835 
               python: https://community.plotly.com/t/plotly-add-bar-bar-charts-disappear-when-width-argument-is-included/61779'''
    
    fig = go.Figure()

    for i in plotting_cols:
        idx = cols.index(i)
        useful_set = set(keys_bad[idx]).intersection(keys_good[idx])
        keys_good[idx] = [i for i in keys_good[idx] if i in useful_set]


        fig.add_bar(
            name = i,
            x= keys_bad[idx],
            y= counts_bad[idx],
            opacity = 0.5,
            width = 0.7,
            marker_color='red'
        )

        fig.add_bar(
            name = i,
            x = keys_good[idx],
            y= counts_good[idx],
            opacity = 0.5,
            width = 0.7,
            marker_color='blue'
        )

    fig.show()


def main():
    # visualize as we wish
    data_raw = Data_Preparation.MakeData(pd.read_csv('texas_car_crashes.csv', 
                                         skiprows=9, 
                                         dtype=str
                                    )
                        )
    dataframe = data_raw.full_data()
    key_data, count_data, cols = original_entry_keys(dataframe)
    keys_good, counts_good, keys_bad, counts_bad = normalized_instance_counts(df=dataframe)
    plotting_cols = find_count_differences(keys_good, counts_good, keys_bad, counts_bad, cols, 
                    tolerance=0.20)
    bar_graphs(key_data, keys_good, counts_good, keys_bad, counts_bad, cols, plotting_cols)

if __name__ == "__main__":
    main()


