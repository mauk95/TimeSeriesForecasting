import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt

from neuralforecast.core import NeuralForecast
from neuralforecast.models import LSTM, RNN, DeepAR
from neuralforecast.models import MLP, NHITS, NBEATS
from neuralforecast.models import TFT, Autoformer, PatchTST
from neuralforecast.models import TimesNet
from neuralforecast.auto import AutoLSTM, AutoRNN, AutoDeepAR, AutoMLP, AutoNHITS, AutoNBEATS, AutoTFT, AutoTimesNet

from neuralforecast.losses.numpy import mae, mse, mape
from ray import tune
import os

pd.set_option('display.max_rows', None)
plt.rcParams["figure.figsize"] = (10,6)

output_path = './results_plots/all_models_except_rnns/multi_sample'
if not os.path.exists(output_path):
    os.makedirs(output_path)

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
fig.savefig(os.path.join(output_path, 'data_multi.png'))

# **The values for "2023-04-01" were repalced by the mean of the respective timeseries since they were initially nulls. For forecasting we can ignore these values and check the predictions on timestamps for which "true" values are given.**
melted_df_with_index = melted_df.set_index('ds')
melted_df_with_index = melted_df_with_index.sort_index()

start_date = '2022-04-01 01:00:00'

melted_df_filtered = melted_df_with_index.loc[:start_date]

# **Reset the index as the dataframe column, to make compatible with NeuralForecast input**
melted_df_filtered = melted_df_filtered.reset_index()
print ('\nmelted_df_filtered.head\n', melted_df_filtered.head())


# ## Model 

# horizon = 96
horizon = 24
# train_steps = 5
train_steps = 100
window_factor = 4
# window_factor = 7
batch_size = 32

config_auto = dict(max_steps=train_steps, val_check_steps=1, input_size=window_factor*horizon, batch_size=batch_size)

models_rnn_based = [
            RNN(h=horizon,
               input_size=window_factor*horizon,
               max_steps=train_steps,
               batch_size=batch_size), 
            LSTM(h=horizon,
               input_size=window_factor*horizon,
               max_steps=train_steps,
               batch_size=batch_size),
            # DeepAR(h=horizon,
            #      input_size=window_factor*horizon,
            #      max_steps=train_steps,
            #    batch_size=batch_size), 
            AutoRNN(h=horizon,
               config=config_auto,
               num_samples=1,
               cpus=1,
               gpus=1),
            AutoLSTM(h=horizon,
               config=config_auto,
               num_samples=1,
               cpus=1,
               gpus=1),
            # AutoDeepAR(h=horizon,
            #    config=config_auto,
            #    num_samples=1,
            #    cpus=1,
            #    gpus=1)
                 ]

models_mlp_based = [
            MLP(h=horizon,
               input_size=window_factor*horizon,
               max_steps=train_steps,
               batch_size=batch_size), 
            NHITS(h=horizon,
               input_size=window_factor*horizon,
               max_steps=train_steps,
               batch_size=batch_size),
            NBEATS(h=horizon,
                 input_size=window_factor*horizon,
                 max_steps=train_steps,
               batch_size=batch_size), 
            AutoMLP(h=horizon,
               config=config_auto,
               num_samples=1,
               cpus=1,
               gpus=1),
            AutoNHITS(h=horizon,
               config=config_auto,
               num_samples=1,
               cpus=1,
               gpus=1),
            AutoNBEATS(h=horizon,
               config=config_auto,
               num_samples=1,
               cpus=1,
               gpus=1)
                 ]

models_transformer_based = [
            TFT(h=horizon,
               input_size=window_factor*horizon,
               max_steps=train_steps,
               batch_size=batch_size), 
            Autoformer(h=horizon,
               input_size=window_factor*horizon,
               max_steps=train_steps,
               batch_size=batch_size),
            PatchTST(h=horizon,
                 input_size=window_factor*horizon,
                 max_steps=train_steps,
               batch_size=batch_size), 
            AutoTFT(h=horizon,
               config=config_auto,
               num_samples=1,
               cpus=1,
               gpus=1)
               ]

models_cnn_based = [
            TimesNet(h=horizon,
               input_size=window_factor*horizon,
               max_steps=train_steps,
               batch_size=batch_size), 
            AutoTimesNet(h=horizon,
               config=config_auto,
               num_samples=1,
               cpus=1,
               gpus=1)
               ]

# models_to_train = models_rnn_based + models_mlp_based + models_transformer_based + models_cnn_based
models_to_train = models_mlp_based + models_transformer_based + models_cnn_based
# models = models_rnn_based
# models_to_train = models_mlp_based
print ('model names: ', models_to_train)
print ('\nTotal Models: ', len(models_to_train))

nf = NeuralForecast(models=models_to_train, freq='H')

# preds_df = nf.cross_validation(df=melted_df_filtered, n_windows=2)
preds_df = nf.cross_validation(df=melted_df_filtered, step_size=horizon, n_windows=2)
print ('\npreds_df', preds_df)
print ('preds_df.columns: ', preds_df.columns)

model_names = [i for i in preds_df.columns if i not in ['unique_id', 'ds', 'cutoff','y']]
print ('models to plot: ', model_names)

fig, ax = plt.subplots()

ax.plot(preds_df['y'], label='actual')
for model_name in model_names:
    ax.plot(preds_df[model_name], label=model_name)

ax.legend(loc='best')
ax.set_xlabel('Time steps')
ax.set_ylabel('Value')

fig.autofmt_xdate()
plt.tight_layout()
plt.title('Predictions for Combined TimeSeries')
plt.savefig(os.path.join(output_path, 'preds_multi_combined_h={}_wf={}_ts={}.png'.format(horizon, window_factor, train_steps)))

# Pivot the DataFrame to reshape it for plotting
names = model_names+['y']
print ('col names: ', names)
preds_df_pivot = preds_df.pivot(index='ds', columns='unique_id', values=names)
print ('\npreds_df_pivot.head()\n', preds_df_pivot.head())

# **Collapse the Multiindex Columns to a Single Index Columns**
preds_df_pivot.columns = preds_df_pivot.columns.map('_'.join)
print ('preds_df_pivot cols: ', preds_df_pivot.columns)

### plot preds for BE Time Series
fig, ax = plt.subplots()

ax.plot(preds_df_pivot['y_BE'], label='actual')
for model_name in model_names:
    ax.plot(preds_df_pivot[model_name+'_BE'], label=model_name)

ax.legend(loc='best')
ax.set_xlabel('Time steps')
ax.set_ylabel('Value')

fig.autofmt_xdate()
plt.tight_layout()
plt.title('Predictions for BE TimeSeries')
plt.savefig(os.path.join(output_path, 'preds_multi_BE_h={}_wf={}_ts={}.png'.format(horizon, window_factor, train_steps)))

### plot preds for PT Time Series
fig, ax = plt.subplots()

ax.plot(preds_df_pivot['y_PT'], label='actual')
for model_name in model_names:
    ax.plot(preds_df_pivot[model_name+'_PT'], label=model_name)

ax.legend(loc='best')
ax.set_xlabel('Time steps')
ax.set_ylabel('Value')

fig.autofmt_xdate()
plt.tight_layout()
plt.title('Predictions for PT TimeSeries')
plt.savefig(os.path.join(output_path, 'preds_multi_PT_h={}_wf={}_ts={}.png'.format(horizon, window_factor, train_steps)))


fig, ax = plt.subplots()

ax.plot(preds_df_pivot['y_RO'], label='actual')
for model_name in model_names:
    ax.plot(preds_df_pivot[model_name+'_RO'], label=model_name)

ax.legend(loc='best')
ax.set_xlabel('Time steps')
ax.set_ylabel('Value')

fig.autofmt_xdate()
plt.tight_layout()
plt.title('Predictions for RO TimeSeries')
plt.savefig(os.path.join(output_path, 'preds_multi_RO_h={}_wf={}_ts={}.png'.format(horizon, window_factor, train_steps)))


# **Plot all the TimeSeries together**
ax = preds_df_pivot.plot(subplots=True, legend=True)
fig = ax[0].get_figure()
fig.savefig(os.path.join(output_path, 'preds_multi_combined_stacked_h={}_wf={}_ts={}.png'.format(horizon, window_factor, train_steps)))


################## RESULTS ######################
mae_list = []
mse_list = []
mape_list = []

# **Average Results for all the TimeSeries Combined**
data = {i: [mae(preds_df[i], preds_df['y']), mse(preds_df[i], preds_df['y']), mape(preds_df[i], preds_df['y'])] for i in model_names}

metrics_df = pd.DataFrame(data=data)
metrics_df.index = ['mae', 'mse', 'mape']

mae_list += [metrics_df[i][0] for i in model_names]
mse_list += [metrics_df[i][1] for i in model_names]
mape_list += [metrics_df[i][2] for i in model_names]

print (preds_df_pivot.head())

# **Results for all the BE TimeSeries**
data = {i: [mae(preds_df_pivot[i+'_BE'], preds_df_pivot['y'+'_BE']), mse(preds_df_pivot[i+'_BE'], preds_df_pivot['y'+'_BE']), 
            mape(preds_df_pivot[i+'_BE'], preds_df_pivot['y'+'_BE'])] for i in model_names}

metrics_df = pd.DataFrame(data=data)
metrics_df.index = ['mae', 'mse', 'mape']

mae_list += [metrics_df[i][0] for i in model_names]
mse_list += [metrics_df[i][1] for i in model_names]
mape_list += [metrics_df[i][2] for i in model_names]

# **Results for all the PT TimeSeries**
data = {i: [mae(preds_df_pivot[i+'_PT'], preds_df_pivot['y'+'_PT']), mse(preds_df_pivot[i+'_PT'], preds_df_pivot['y'+'_PT']), 
            mape(preds_df_pivot[i+'_PT'], preds_df_pivot['y'+'_PT'])] for i in model_names}

metrics_df = pd.DataFrame(data=data)
metrics_df.index = ['mae', 'mse', 'mape']

mae_list += [metrics_df[i][0] for i in model_names]
mse_list += [metrics_df[i][1] for i in model_names]
mape_list += [metrics_df[i][2] for i in model_names]

# **Results for all the RO TimeSeries**
data = {i: [mae(preds_df_pivot[i+'_RO'], preds_df_pivot['y'+'_RO']), mse(preds_df_pivot[i+'_RO'], preds_df_pivot['y'+'_RO']), 
            mape(preds_df_pivot[i+'_RO'], preds_df_pivot['y'+'_RO'])] for i in model_names}

metrics_df = pd.DataFrame(data=data)
metrics_df.index = ['mae', 'mse', 'mape']

mae_list += [metrics_df[i][0] for i in model_names]
mse_list += [metrics_df[i][1] for i in model_names]
mape_list += [metrics_df[i][2] for i in model_names]

results_df = pd.DataFrame(columns=['Series_ID', 'Model', 'MAE', 'MSE','MAPE'])

results_df = pd.concat([results_df, pd.DataFrame({ 
                    'Series_ID': ['Combined']*len(model_names) + ['BE']*len(model_names) + ['PT']*len(model_names) + ['RO']*len(model_names), 
                    'Model': [i for i in model_names] * 4, 
                    'MAE': mae_list, 
                    'MSE': mse_list, 
                    'MAPE': mape_list})], 
                    ignore_index=True)

# Change the order of columns
new_order = ['Series_ID', 'Model', 'MAE', 'MSE','MAPE']
results_df = results_df[new_order]
print ('\nFinal results_df\n', results_df)

results_df.to_csv(os.path.join(output_path, 'results_multi_all_models.csv'), index=False)


