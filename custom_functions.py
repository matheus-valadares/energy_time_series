import pandas as pd
import numpy as np
from datetime import datetime as dt
from statsmodels.tsa.seasonal import seasonal_decompose

import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import lag_plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy import stats

#__________________________________________________________________________________________________________________________________
#________________________________________________________SPECIFIC FUNCTIONS________________________________________________________
#__________________________________________________________________________________________________________________________________
def energy_df_treatment(df, column_name):
    """
    Apply some functions to transform the wide formatted dataframe into long format, and clean its data.
    Args:
        df (pandas dataframe): dataframe with specific uncleaned format
        column_name (str): the name of the values column
    Returns:
        cleaned dataframe
    """
    df['YEAR'] = list(range(2022, 2003, -1))

    df = df.melt(id_vars=['YEAR'],
                value_vars=['JAN', 'FEV', 'MAR', 'ABR', 'MAI', 'JUN', 'JUL', 'AGO', 'SET', 'OUT', 'NOV', 'DEZ'],
                var_name='MONTH',
                value_name=column_name)

    # Make a DATE column
    month = {
        'JAN':1,
        'FEV':2,
        'MAR':3,
        'ABR':4,
        'MAI':5,
        'JUN':6,
        'JUL':7,
        'AGO':8,
        'SET':9,
        'OUT':10,
        'NOV':11,
        'DEZ':12
    }
    df['DATE'] = 1
    df['DATE'] = df['DATE'].astype(str) + '-' + df['MONTH'].map(month).astype(str) + '-' + df['YEAR'].astype(int).astype(str)
    df['DATE'] = df['DATE'].apply(lambda x: dt.strptime(x, '%d-%m-%Y'))
    df.drop(columns=['YEAR', 'MONTH'], inplace=True)
    df.set_index('DATE', inplace=True)
    return df

def concat_df(df_list):
    """
    Concatenates the list of dataframes into one, establishes a monthly start frequency for the datetime index, and round the values to 4 decimals.
    Args:
        df_list (list): list of pandas dataframes with datetime index
    Returns:
        dataframe
    """
    df = pd.concat(df_list, axis=1)
    df = df.asfreq('MS')
    # df = df.map(lambda x: round(x, 4))
    return df

#__________________________________________________________________________________________________________________________________
#___________________________________________________________DATA CLEANING__________________________________________________________
#__________________________________________________________________________________________________________________________________
def filter_df(df):
    """
    Filters the dataframe range by the date of the index.
    Args:
        df (pandas dataframe): dataframe with datetime index
    Returns:
        filtered dataframe
    """
    return df.sort_index().loc['2004-01-01':'2022-12-01']


#__________________________________________________________________________________________________________________________________
#________________________________________________________PLOTTING FUNCTIONS________________________________________________________
#__________________________________________________________________________________________________________________________________
def plot_decomposition(df, seasonality='multiplicative', title=''):
    """
    Decomposes the time series by the chosen method (multiplicative or additive) into Trend, Seasonality, and Residuals.
    Plots, respectively, the original Time Series, its Trend, Seasonality, and Residuals.

    Args:
        df (pandas dataframe): single column dataframe with datetime index
        seasonality (str, default:multiplicative): method of time series decomposition (multiplicative/additive)
    """
    decomposition = seasonal_decompose(df, model=seasonality, extrapolate_trend=0)

    plt.figure(figsize=(15, 12))

    plt.subplot(411)
    plt.plot(df.index, df.values)
    title = ' ' + title
    plt.title(f'Decomposition of the{title} time series', fontsize=15)
    plt.ylabel('Time Series')

    plt.subplot(412)
    plt.plot(decomposition.trend.index, decomposition.trend.values)
    plt.ylabel('Trend')

    plt.subplot(413)
    plt.plot(decomposition.seasonal.index, decomposition.seasonal.values)
    plt.ylabel('Seasonality')

    plt.subplot(414)
    plt.plot(decomposition.resid.index, decomposition.resid.values)
    plt.ylabel('Residuals')
    
    plt.tight_layout()


def plot_differentiation(df, differentiation=1):
    """
    Plots the differentiated time series, and the histogram of its frequencies.

    Args:
        df (pandas dataframe): single column dataframe with datetime index
        differentiation (int): number of differentiations to be applied to the time series
    """
    # Data
    if differentiation <= 0:
        data = df
    else:
        data = df.diff(differentiation).dropna()

    # Differentiated series
    plt.figure(figsize=(15, 5))

    plt.subplot(211)
    sns.lineplot(data=data, dashes=False)
    plt.title(f'Differentiated time series | {differentiation} diff', fontsize=15)
    plt.ylabel('Time Series')
    plt.legend('')

    # Histogram of the differentiated series
    plt.subplot(212)
    sns.histplot(data=data)
    plt.ylabel('Distribution')

    plt.tight_layout()


def plot_correlation(df, method='spearman'):
    """
    Plot the correlation matrix using the specified method.

    Parameters:
        df (DataFrame): Pandas DataFrame containing the data for correlation computation.
        method (str, optional): Method of correlation. Defaults to 'spearman'. Other options include 'pearson' and 'kendall'.

    Returns:
        None. Displays a heatmap of the correlation matrix.
    """
    df = df.corr(method=method)
    mask = np.triu(df)
    sns.heatmap(df, annot=True, vmin=-1, vmax=1, mask=mask, cmap='YlGnBu')
    plt.title('Spearman Correlation', fontsize=15)
    plt.show()


def plot_autocorrelation(df, lags=None):
    """
    Plots the autocorrelation (ACF) and partial autocorrelation (PACF) graphs.

    Args:
        df (pandas dataframe): single column dataframe with datetime index
        lags (int, default:None): number of lags to be plotted on the graphs. If None, uses the maximum number of records in the series
    """
    if lags == None:
        lags = df.shape[0] - 1
    plt.figure(figsize=(15, 8))

    plt.subplot(211)
    plot_acf(df, lags=lags, ax=plt.gca())
    plt.title(f'ACF | {lags} Lags', fontsize=15)

    lags = int(lags/2)
    plt.subplot(212)
    plot_pacf(df, lags=lags, ax=plt.gca())
    plt.title(f'PACF | {lags} Lags', fontsize=15)

    plt.tight_layout()


def plot_lag_scatter(df, lags=1):
    """
    Plots a scatter plot of the time series points (x) and their specified order lags (y).

    Args:
        df (pandas dataframe): single column dataframe with datetime index
        lags (int, default:None): number of lags to be plotted on the y-axis
    """
    plt.figure(figsize=(7,7))
    lag_plot(df, lags)
    plt.title(f'Lag Plot', fontsize=15)
    plt.show()


#__________________________________________________________________________________________________________________________________
#__________________________________________________________EXPERIMENTATION_________________________________________________________
#__________________________________________________________________________________________________________________________________
def adf_test(series, title=''):
    """
    Takes a time series and returns a report of the Augmented Dickey-Fuller (ADF) test.

    Args:
        series (pandas series): time series
        title (str): optional title
    Returns:
        Interpretation of the stationarity test results
    """
    
    differenciation = 0
    print(f'Augmented Dickey-Fuller Test: {title}')
    print(f'Differentiation: {differenciation}')
    result = adfuller(series, autolag='AIC')

    labels = ['ADF test statistic', 'p-value', '# lags used', '# observations']
    output = pd.Series(result[0:4], index=labels)
    for key, val in result[4].items():
        output[f'critical value ({key})'] = val
    print(output.to_string())

    if result[1] <= 0.05:
        print("Rejects the null hypothesis")
        print("Data does NOT have a unit root and IS stationary")
    else:
        print("Does not reject the null hypothesis")
        print("Data HAS a unit root and IS NOT stationary")
        
        result = {}
        result[1] = 1
        while result[1] > 0.05:
            differenciation += 1
            print(f'\n\nDifferentiation: {differenciation}')
            data = series.diff(differenciation).dropna()
            result = adfuller(data, autolag='AIC') # .dropna() deals with differentiated data

            labels = ['ADF test statistic', 'p-value', '# lags used', '# observations']
            output = pd.Series(result[0:4], index=labels)

            for key, val in result[4].items():
                output[f'critical value ({key})'] = val

            print(output.to_string())          # .to_string() removes "dtype: float64"

            if result[1] <= 0.05:
                print("Rejects the null hypothesis")
                print("Data does NOT have a unit root and IS stationary")
            else:
                print("Does not reject the null hypothesis")
                print("Data HAS a unit root and IS NOT stationary")


def perform_normality_tests(data):
    """
    Perform the Shapiro-Wilk test on a dataset.
    
    Parameters:
        data: Pandas Series, the dataset to test for normality.
    
    Returns:
        A print statement containing the results for the normality test.
    """

    # Shapiro-Wilk Test
    shapiro_stat, shapiro_p = stats.shapiro(data)

    print('Shapiro-Wilk Test')
    print(f'Test Statistic: {shapiro_stat:.4f}\np-value: {shapiro_p:.4f}')
    
    if shapiro_p > 0.05:
        print("Does not reject the null hypothesis")
        print("Data does IS normally distributed\n")
    else:
        print("Rejects the null hypothesis")
        print("Data does IS NOT normally distributed\n")


def error_statistics(series, predictions):
    """
    Takes the actual series and prediction to return error statistics.

    Args:
        series (pandas series): actual values of the time series
        predictions (pandas series): time series predictions generated by the model

    Returns:
        Error statistics (MAE, MSE, RMSE, MAPE)
    """
    mean_series = series.mean()
    mae = mean_absolute_error(series, predictions)
    mse = mean_squared_error(series, predictions)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((series - predictions) / series)) * 100

    print('Mean of the real values: {:,.2f}'.format(mean_series))
    print('MAE - Mean Absolute Error: {:,.2f}'.format(mae))
    print('MSE - Mean Squared Error: {:.2f}'.format(mse))
    print('RMSE - Root Mean Squared Error: {:,.2f}'.format(rmse))
    print('MAPE - Mean Absolute Percentage Error: {:.2f}%'.format(mape))


def model_evaluation(main_dataframe, test_data, test_predictions, title='Forecast', aic=-1):
    """
    Presents the model's error statistics, plots the complete time series with the prediction, and
    plots the time series window of the forecasted period.

    Args:
        main_dataframe (pandas series): complete time series with datetime index
        test_data (pandas series): actual time series with datetime index of the forecasted period
        test_predictions (pandas series): time series prediction with datetime index
        title (str|optional): title of the graph
    """
    error_statistics(test_data, test_predictions)
    mape = np.mean(np.abs((test_data - test_predictions) / test_data)) * 100
    title = title + ' | MAPE - {:.2f}%'.format(mape)
    if aic != -1:
        title = title + ' | AIC - {:.4f}'.format(aic)

    plt.figure(figsize=(25, 10))

    plt.subplot(211)
    plt.plot(test_data, label='Real', color='orange')
    plt.plot(test_predictions, label='Forecast', color='green')
    plt.title(title, fontsize=15)
    plt.legend(loc='upper left', fontsize=12)

    plt.subplot(212)
    plt.plot(main_dataframe, label='Real', color='orange')
    plt.plot(test_predictions, label='Forecast', color='green')
    plt.show()