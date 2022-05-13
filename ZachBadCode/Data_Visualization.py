'''
Visualize data in several ways.
'''

import pandas as pd
import matplotlib.pyplot as plt
import dython # for associations

class Visualizations():
    # Edit this class to work later.
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


    def visualize_linear_relationships(df):
        '''
        Covariance matrix for data. Nothing useful IMO. Full data is slow to run.
        '''
        df = df[df["Latitude"].str.contains("No Data")==False]
        associations(df)
        plt.show()

if __name__ == "__main__":
    main()