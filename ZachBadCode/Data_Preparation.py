'''
Here we prepare collsion data for the models.
'''

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

class MakeData():
    ''' Create useful subset of data from CSV. '''
    def __init__(self, data):
        self.data = data

    def full_data(self):
        # Give data with no alterations
        data = self.data
        return data

    def data_all_or_no_nums(self, only_nums = False):
        ''' Given data, return subset which is all numeric or all non numeric. '''
        col_isnum_names = []
        col_not_num_names = []
        data = self.data

        # check middle entry to avoid "no data", and work if len(keys) is 1
        for col in data:
            keys = data[col].value_counts(dropna=False).keys()
            middle_key = keys[len(keys)//2]

            try: 
                float(middle_key)
                col_isnum_names.append(col)
            except ValueError:
                col_not_num_names.append(col)

        if only_nums:
            data_numeric = data[col_isnum_names]
            data_numeric = data_numeric.apply(pd.to_numeric, args=('coerce',))
            return data_numeric
        return data[col_not_num_names]


def drop_large_cols(df, val_threshold=70):
    ''' For each column, if entry has over (val_threshold) categorical variables, 
        remove it. '''
    cols_to_drop = []
    for col in df:
        if len(df[col].value_counts(dropna=False)) > val_threshold:
            cols_to_drop.append(col)
    new_data = df.drop(cols_to_drop, axis=1)
    return new_data

def make_labels(df):
    ''' Cast 'Person Injury Severity' into binary labels.' This counts as panda abuse.'''
    df['Person Injury Severity'].loc[df['Person Injury Severity'] == "N - NOT INJURED"]              = 0
    df['Person Injury Severity'].loc[df['Person Injury Severity'] == "99 - UNKNOWN"]                 = 0
    df['Person Injury Severity'].loc[df['Person Injury Severity'] == "B - SUSPECTED MINOR INJURY"]   = 0
    df['Person Injury Severity'].loc[df['Person Injury Severity'] == "C - POSSIBLE INJURY"]          = 0
    df['Person Injury Severity'].loc[df['Person Injury Severity'] == "A - SUSPECTED SERIOUS INJURY"] = 1
    df['Person Injury Severity'].loc[df['Person Injury Severity'] == "K - FATAL INJURY"]             = 1
    df = df.rename(columns={'Person Injury Severity': 'Labels'})
    # print(df['Labels'].iloc[665:675])
    return df

def tokenize_data_1hot(df):
    ''' One Hot tokenization for categorical data.'''
    # drop label data
    feature_data = df.drop(['Person Injury Severity'], axis=1) 
    enc = OneHotEncoder(handle_unknown='ignore')
    encoded_data = enc.fit_transform(feature_data).toarray()
    return encoded_data

def remove_other_death_cols(df):
    # remove all columns which are too similar to what we are testing for.
    removal_list = ['Unit Death Count',
                    'Driver Time of Death',
                    'Driver Time of Death',
                    'Crash Death Count'
                    ]
    data_reduced = df.drop(removal_list, axis=1)
    return data_reduced

def normalize(df):
    # mean -> 0, var -> 1. Z value.
    variance = df.var(numeric_only=True)
    mean = df.mean(numeric_only=True)
    df_out = (df - mean) / (variance + 1e-6)**0.5  
    return df_out

def fill_nan(df):
    df_filled = df.fillna(0)
    return df_filled

def prepare_data():
    ''' SOLID principles? What are those? '''
    print('\nPreparing Data.\n')

    # Make the data to be read.
    data_raw = MakeData(pd.read_csv('texas_car_crashes.csv', 
                                         skiprows=9, 
                                         dtype=str
                                    )
                        )

    #categorical data.
    data_cat = data_raw.data_all_or_no_nums(only_nums=False)
    labels = make_labels(data_cat)
    data_cat = drop_large_cols(data_cat)
    data_cat = tokenize_data_1hot(data_cat)
    # print(np.shape(encoded_data))

    # numerical data.
    data_num = data_raw.data_all_or_no_nums(only_nums=True)
    data_num = remove_other_death_cols(data_num)
    data_num = normalize(data_num)
    data_num = fill_nan(data_num)
    data_num = np.asarray(data_num)
    # print(np.shape(data_num))

    # combine data:
    data = np.concatenate((data_cat, data_num), axis=1)
    return data, labels 

if __name__ == "__main__":
    prepare_data()