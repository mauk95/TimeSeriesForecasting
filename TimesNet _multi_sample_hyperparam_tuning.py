import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt

from neuralforecast.core import NeuralForecast
from neuralforecast.models import NHITS, NBEATS, TimesNet
from neuralforecast.losses.numpy import mae, mse
import os

pd.set_option('display.max_rows', None)
plt.rcParams["figure.figsize"] = (10,6)

plot_path = './results_plots/hyperparam_tuning/multi_sample'
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

file_name = 'data/Multiple_sample_series.csv'

df = pd.read_csv(file_name)
df = df.drop(columns=['Unnamed: 0', 'Timeseries ID'], axis=1)

# Melt the DataFrame to unpivot it
melted_df = pd.melt(df, id_vars=['Date', 'ID'], var_name='Hour', value_name='Value')

# # Combine 'Date' and 'Hour' columns into a new 'Datetime' column
melted_df['Datetime'] = pd.to_datetime(melted_df['Date'] + ' ' + melted_df['Hour'])
melted_df.drop(['Date', 'Hour'], axis=1, inplace=True)

# Change the order of columns
new_order = ['Datetime', 'ID', 'Value']
melted_df = melted_df[new_order]

# Rename columns
column_mapping = {'Datetime': 'ds', 'ID': 'unique_id', 'Value': 'y'}
melted_df.rename(columns=column_mapping, inplace=True)

# **Replace the NaN values wilth "mean" value of that column for each unique_id separately**
melted_df['y'] = melted_df['y'].fillna(melted_df.groupby('unique_id')['y'].transform('mean'))

# **Plot All TimeSeries**

# Pivot the DataFrame to reshape it for plotting
df_pivot = melted_df.pivot(index='ds', columns='unique_id', values='y')

# **PLot the Series separately, to avoid overlap**
ax = df_pivot.plot(subplots=True, legend=True)
print (ax.shape)
fig = ax[0].get_figure()
fig.savefig(os.path.join(plot_path, 'data_multi.png'))

# # **Visualize the plots for "Jan 2020" only**

# # Define a date range for filtering
# start_date = '2020-01-01 00:00:00'
# end_date = '2020-02-01 00:00:00'

# ax = df_pivot.loc[start_date:end_date].plot(subplots=True, legend=True)
# fig = ax[0].get_figure()
# fig.savefig(os.path.join(plot_path, 'yearly_data_single.png'))

# # **Visualize the plots for "Feb 2020" only**

# # Define a date range for filtering
# start_date = '2020-02-01 00:00:00'
# end_date = '2020-03-01 00:00:00'

# df_pivot.loc[start_date:end_date].plot(subplots=True, legend=True)

# **The values for "2023-04-01" were repalced by the mean of the respective timeseries since they were initially nulls. For forecasting we can ignore these values and check the predictions on timestamps for which "true" values are given.**
melted_df_with_index = melted_df.set_index('ds')
melted_df_with_index = melted_df_with_index.sort_index()

start_date = '2022-04-01 01:00:00'

melted_df_filtered = melted_df_with_index.loc[:start_date]

# **Reset the index as the dataframe column, to make compatible with NeuralForecast input**
melted_df_filtered = melted_df_filtered.reset_index()


# ## Model 
results_df = pd.DataFrame(columns=['Horizon', 'Input_Size', 'Train_Steps', 'MAE', 'MSE'])

horizon_list = [24, 48, 96]
window_factor_list = [2, 4, 8]
train_steps_list = [100, 200, 300]

trial_num = 1
for horizon in horizon_list:
    for window_factor in window_factor_list:
        for train_steps in train_steps_list:
            print ('\n===== Running Trial: {} =====\n'.format(trial_num))
            print ('\nHorizon: {}, Input_Size: {}, Train_Steps: {}\n'.format(horizon, window_factor*horizon, train_steps))
            
            mae_list = []
            mse_list = []

            models = [NHITS(h=horizon,
                        input_size=window_factor*horizon,
                        max_steps=train_steps),
                    NBEATS(h=horizon,
                        input_size=window_factor*horizon,
                        max_steps=train_steps),
                    TimesNet(h=horizon,
                            input_size=window_factor*horizon,
                            max_steps=train_steps)]

            nf = NeuralForecast(models=models, freq='H')

            preds_df = nf.cross_validation(df=melted_df_filtered, step_size=horizon, n_windows=2)
            print ('\npreds_df', preds_df)

            ### plot preds for Combined Time Series
            fig, ax = plt.subplots()

            ax.plot(preds_df['y'], label='actual')
            ax.plot(preds_df['NHITS'], label='N-HITS', ls='--')
            ax.plot(preds_df['NBEATS'], label='N-BEATS', ls=':')
            ax.plot(preds_df['TimesNet'], label='TimesNet', ls='-.')

            ax.legend(loc='best')
            ax.set_xlabel('Time steps')
            ax.set_ylabel('Value')

            fig.autofmt_xdate()
            plt.tight_layout()
            plt.title('Predictions for Combined TimeSeries')
            plt.savefig(os.path.join(plot_path, 'preds_multi_combined_h={}_wf={}_ts={}.png'.format(horizon, window_factor, train_steps)))

            preds_df_pivot = preds_df.pivot(index='ds', columns='unique_id', values=['NHITS','NBEATS','TimesNet','y'])

            # **Collapse the Multiindex Columns to a Single Index Columns**
            preds_df_pivot.columns = preds_df_pivot.columns.map('_'.join)

            ### plot preds for BE Time Series
            fig, ax = plt.subplots()

            ax.plot(preds_df_pivot['y_BE'], label='actual')
            ax.plot(preds_df_pivot['NHITS_BE'], label='N-HITS', ls='--')
            ax.plot(preds_df_pivot['NBEATS_BE'], label='N-BEATS', ls=':')
            ax.plot(preds_df_pivot['TimesNet_BE'], label='TimesNet', ls='-.')

            ax.legend(loc='best')
            ax.set_xlabel('Time steps')
            ax.set_ylabel('Value')

            fig.autofmt_xdate()
            plt.tight_layout()
            plt.title('Predictions for BE TimeSeries')
            plt.savefig(os.path.join(plot_path, 'preds_multi_BE_h={}_wf={}_ts={}.png'.format(horizon, window_factor, train_steps)))

            ### plot preds for PT Time Series
            fig, ax = plt.subplots()

            ax.plot(preds_df_pivot['y_PT'], label='actual')
            ax.plot(preds_df_pivot['NHITS_PT'], label='N-HITS', ls='--')
            ax.plot(preds_df_pivot['NBEATS_PT'], label='N-BEATS', ls=':')
            ax.plot(preds_df_pivot['TimesNet_PT'], label='TimesNet', ls='-.')

            ax.legend(loc='best')
            ax.set_xlabel('Time steps')
            ax.set_ylabel('Value')

            fig.autofmt_xdate()
            plt.tight_layout()
            plt.title('Predictions for PT TimeSeries')
            plt.savefig(os.path.join(plot_path, 'preds_multi_PT_h={}_wf={}_ts={}.png'.format(horizon, window_factor, train_steps)))


            fig, ax = plt.subplots()

            ax.plot(preds_df_pivot['y_RO'], label='actual')
            ax.plot(preds_df_pivot['NHITS_RO'], label='N-HITS', ls='--')
            ax.plot(preds_df_pivot['NBEATS_RO'], label='N-BEATS', ls=':')
            ax.plot(preds_df_pivot['TimesNet_RO'], label='TimesNet', ls='-.')

            ax.legend(loc='best')
            ax.set_xlabel('Time steps')
            ax.set_ylabel('Value')

            fig.autofmt_xdate()
            plt.tight_layout()
            plt.title('Predictions for RO TimeSeries')
            plt.savefig(os.path.join(plot_path, 'preds_multi_RO_h={}_wf={}_ts={}.png'.format(horizon, window_factor, train_steps)))


            # **Plot all the TimeSeries together**
            ax = preds_df_pivot.plot(subplots=True, legend=True)
            fig = ax[0].get_figure()
            fig.savefig(os.path.join(plot_path, 'preds_multi_combined_stacked_h={}_wf={}_ts={}.png'.format(horizon, window_factor, train_steps)))


            # **Average Results for all the TimeSeries Combined**
            data = {'N-HiTS': [mae(preds_df['NHITS'], preds_df['y']), mse(preds_df['NHITS'], preds_df['y'])],
                'N-BEATS': [mae(preds_df['NBEATS'], preds_df['y']), mse(preds_df['NBEATS'], preds_df['y'])],
                'TimesNet': [mae(preds_df['TimesNet'], preds_df['y']), mse(preds_df['TimesNet'], preds_df['y'])]}

            metrics_df = pd.DataFrame(data=data)
            metrics_df.index = ['mae', 'mse']

            # metrics_df.style.highlight_min(color='lightgreen', axis=1)
            mae_list += [metrics_df['N-HiTS'][0],metrics_df['N-BEATS'][0],metrics_df['TimesNet'][0]]
            mse_list += [metrics_df['N-HiTS'][1],metrics_df['N-BEATS'][1],metrics_df['TimesNet'][1]]

            # **Results for all the BE TimeSeries**
            data = {'N-HiTS': [mae(preds_df_pivot['NHITS_BE'], preds_df_pivot['y_BE']), mse(preds_df_pivot['NHITS_BE'], preds_df_pivot['y_BE'])],
                'N-BEATS': [mae(preds_df_pivot['NBEATS_BE'], preds_df_pivot['y_BE']), mse(preds_df_pivot['NBEATS_BE'], preds_df_pivot['y_BE'])],
                'TimesNet': [mae(preds_df_pivot['TimesNet_BE'], preds_df_pivot['y_BE']), mse(preds_df_pivot['TimesNet_BE'], preds_df_pivot['y_BE'])]}

            metrics_df = pd.DataFrame(data=data)
            metrics_df.index = ['mae', 'mse']

            # metrics_df.style.highlight_min(color='lightgreen', axis=1)
            mae_list += [metrics_df['N-HiTS'][0],metrics_df['N-BEATS'][0],metrics_df['TimesNet'][0]]
            mse_list += [metrics_df['N-HiTS'][1],metrics_df['N-BEATS'][1],metrics_df['TimesNet'][1]]

            # **Results for all the PT TimeSeries**
            data = {'N-HiTS': [mae(preds_df_pivot['NHITS_PT'], preds_df_pivot['y_PT']), mse(preds_df_pivot['NHITS_PT'], preds_df_pivot['y_PT'])],
                'N-BEATS': [mae(preds_df_pivot['NBEATS_PT'], preds_df_pivot['y_PT']), mse(preds_df_pivot['NBEATS_PT'], preds_df_pivot['y_PT'])],
                'TimesNet': [mae(preds_df_pivot['TimesNet_PT'], preds_df_pivot['y_PT']), mse(preds_df_pivot['TimesNet_PT'], preds_df_pivot['y_PT'])]}

            metrics_df = pd.DataFrame(data=data)
            metrics_df.index = ['mae', 'mse']

            # metrics_df.style.highlight_min(color='lightgreen', axis=1)
            mae_list += [metrics_df['N-HiTS'][0],metrics_df['N-BEATS'][0],metrics_df['TimesNet'][0]]
            mse_list += [metrics_df['N-HiTS'][1],metrics_df['N-BEATS'][1],metrics_df['TimesNet'][1]]

            # **Results for all the RO TimeSeries**

            data = {'N-HiTS': [mae(preds_df_pivot['NHITS_RO'], preds_df_pivot['y_RO']), mse(preds_df_pivot['NHITS_RO'], preds_df_pivot['y_RO'])],
                'N-BEATS': [mae(preds_df_pivot['NBEATS_RO'], preds_df_pivot['y_RO']), mse(preds_df_pivot['NBEATS_RO'], preds_df_pivot['y_RO'])],
                'TimesNet': [mae(preds_df_pivot['TimesNet_RO'], preds_df_pivot['y_RO']), mse(preds_df_pivot['TimesNet_RO'], preds_df_pivot['y_RO'])]}

            metrics_df = pd.DataFrame(data=data)
            metrics_df.index = ['mae', 'mse']

            # metrics_df.style.highlight_min(color='lightgreen', axis=1)
            mae_list += [metrics_df['N-HiTS'][0],metrics_df['N-BEATS'][0],metrics_df['TimesNet'][0]]
            mse_list += [metrics_df['N-HiTS'][1],metrics_df['N-BEATS'][1],metrics_df['TimesNet'][1]]

            results_df = pd.concat([results_df, pd.DataFrame({'Horizon': [horizon] * 12, 
                                'Input_Size': [window_factor*horizon] * 12, 
                                'Train_Steps': [train_steps] * 12, 
                                'Series_ID': ['Combined']*3 + ['BE']*3 + ['PT']*3 + ['RO']*3, 
                                'Model': ['NHITS','NBEATS','TimesNet'] * 4, 
                                'MAE': mae_list, 
                                'MSE': mse_list})], 
                                ignore_index=True)
            
            trial_num+=1
            print (results_df)

# Change the order of columns
new_order = ['Horizon', 'Input_Size', 'Train_Steps', 'Series_ID', 'Model','MAE', 'MSE']
results_df = results_df[new_order]
print ('\nFinal results_df\n', results_df)

results_df.to_csv(os.path.join(plot_path, 'results_multi_hyperparams.csv'), index=False)


