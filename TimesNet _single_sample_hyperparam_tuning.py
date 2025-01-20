import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt

from neuralforecast.core import NeuralForecast
from neuralforecast.models import NHITS, NBEATS, TimesNet
from neuralforecast.losses.numpy import mae, mse

import os

pd.set_option('display.max_rows', None)
plt.rcParams["figure.figsize"] = (9,6)

plot_path = './results_plots/hyperparam_tuning/single_sample'
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

file_name = 'data/Single_sample.csv'
df = pd.read_csv(file_name, header=0, names=['ds', 'y'])

# **Add a new column 'unique_id', because NeuralForecast expects it.**
df['unique_id'] = 'Value'

# **Some dates are in a different format so, we convert all dates to a single format.**
df['ds'] = df['ds'].apply(lambda x: pd.to_datetime(x).strftime('%Y-%m-%d %H:%M:%S'))
df['ds'] = pd.to_datetime(df['ds'])

fig, ax = plt.subplots()

ax.plot(df['y'])
ax.set_xlabel('Time')
ax.set_ylabel('Value')

fig.autofmt_xdate()
plt.tight_layout()
plt.savefig(os.path.join(plot_path, 'raw_data_single.png'))

# Define a date range for filtering
start_date = '2022-02-14 00:00:00'
end_date = '2022-02-16 23:00:00'

# Create a boolean condition for filtering
condition = (df['ds'] >= start_date) & (df['ds'] <= end_date)

# filtered_df = df[condition]

# # **Visualize the missing values**
# fig, ax = plt.subplots()
# ax.plot(filtered_df['ds'], filtered_df['y'])
# ax.set_xlabel('Time')
# ax.set_ylabel('Value')
# fig.autofmt_xdate()
# plt.tight_layout()

# # **Now visualize the values from the same period, but from a different month**
# start_date = '2022-01-14 00:00:00'
# end_date = '2022-01-16 23:00:00'

# condition_2 = (df['ds'] >= start_date) & (df['ds'] <= end_date)

# filtered_df = df[condition_2]

# fig, ax = plt.subplots()

# ax.plot(filtered_df['ds'], filtered_df['y'])
# ax.set_xlabel('Time')
# ax.set_ylabel('Value')

# fig.autofmt_xdate()
# plt.tight_layout()

# **Fill the missing values with the mean value**
df['y'].fillna(df['y'].mean(), inplace=True)
df.head()

# # Apply the boolean condition to filter the DataFrame
# filtered_df_clean = df[condition]

# fig, ax = plt.subplots()

# ax.plot(filtered_df_clean['ds'], filtered_df_clean['y'])
# ax.set_xlabel('Time')
# ax.set_ylabel('Value')
# fig.autofmt_xdate()
# plt.tight_layout()

# **We remove the last 48 records since, there are some null values in them. Doing forecast on these values will be difficult**
df = df[:-48]
df_with_index = df.set_index('ds')

# ## visualize the series after replacing nulls
# fig, ax = plt.subplots()
# ax.plot(df_with_index['y'])
# ax.set_xlabel('Time')
# ax.set_ylabel('Value')
# fig.autofmt_xdate()
# plt.tight_layout()

# **Visualize the Yearly data**
groups = df_with_index.groupby(pd.Grouper(freq='A'))
years = pd.DataFrame()
for name, group in groups:
    year = pd.DataFrame({name.year:group['y'].tolist()})
    years = pd.concat([years,year], ignore_index=True, axis=1)

years.columns = df['ds'].dt.year.unique()
ax_years = years.plot(subplots=True, legend=True)
fig = ax_years[0].get_figure()
fig.savefig(os.path.join(plot_path, 'yearly_data_single.png'))

# **Visualize the Monthly data for Year 2020**
groups = df_with_index.loc[f'{2020}'].groupby(pd.Grouper(freq='M'))
months = pd.DataFrame()
for name, group in groups:
    month = pd.DataFrame({name.month:group['y'].tolist()})
    months = pd.concat([months,month], ignore_index=True, axis=1)

months.columns = sorted(df['ds'].dt.month.unique())
ax_months  = months.plot(subplots=True, legend=True)
print (ax_months.shape)
fig = ax_months[0].get_figure()
fig.savefig(os.path.join(plot_path, 'monthly_data_single.png'))

# **We can NOT see a monthly period!!!**

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

            preds_df = nf.cross_validation(df=df, step_size=horizon, n_windows=2)

            preds_df_with_index = preds_df.set_index('ds')

            ### plot the preds
            fig, ax = plt.subplots()

            ax.plot(preds_df_with_index['y'], label='actual')
            ax.plot(preds_df_with_index['NHITS'], label='N-HITS', ls='--')
            ax.plot(preds_df_with_index['NBEATS'], label='N-BEATS', ls=':')
            ax.plot(preds_df_with_index['TimesNet'], label='TimesNet', ls='-.')

            ax.legend(loc='best')
            ax.set_xlabel('Time steps')
            ax.set_ylabel('Value')
            fig.autofmt_xdate()
            plt.tight_layout()
            plt.savefig(os.path.join(plot_path, 'preds_single_h={}_wf={}_ts={}.png'.format(horizon, window_factor, train_steps)))

            data = {'N-HiTS': [mae(preds_df['NHITS'], preds_df['y']), mse(preds_df['NHITS'], preds_df['y'])],
                'N-BEATS': [mae(preds_df['NBEATS'], preds_df['y']), mse(preds_df['NBEATS'], preds_df['y'])],
                'TimesNet': [mae(preds_df['TimesNet'], preds_df['y']), mse(preds_df['TimesNet'], preds_df['y'])]}

            metrics_df = pd.DataFrame(data=data)
            metrics_df.index = ['mae', 'mse']
            # metrics_df.style.highlight_min(color='lightgreen', axis=1)

            results_df = pd.concat([results_df, pd.DataFrame({'Horizon': [horizon, horizon, horizon], 
                                            'Input_Size': [window_factor*horizon, window_factor*horizon, window_factor*horizon], 
                                            'Train_Steps': [train_steps,train_steps,train_steps], 
                                            'Model': ['NHITS','NBEATS','TimesNet'], 
                                            'MAE': [metrics_df['N-HiTS'][0],metrics_df['N-BEATS'][0],metrics_df['TimesNet'][0]], 
                                            'MSE': [metrics_df['N-HiTS'][1], metrics_df['N-BEATS'][1], metrics_df['TimesNet'][1]]})], 
                                            ignore_index=True)
            trial_num+=1
            print (results_df)

# Change the order of columns
new_order = ['Horizon', 'Input_Size', 'Train_Steps', 'Model', 'MAE', 'MSE']
results_df = results_df[new_order]
print ('\nFinal results_df\n', results_df)

results_df.to_csv(os.path.join(plot_path, 'results_single_hyperparams.csv'), index=False)