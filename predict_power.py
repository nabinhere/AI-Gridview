import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

PHYSICS = 0
BIOTECH = 1
MANAGEMENT = 2
CIVIL = 3

weekday_encoding = {'Sunday': 1, 'Monday': 2, 'Tuesday': 3, 'Wednesday': 4, 'Thursday': 5, 'Friday': 6, 'Saturday': 7}

dataset_root_PATH = r"C:\Users\nabin\OneDrive\Desktop\AI Gridview\[0] Datasets"
dataset_PATH = {PHYSICS: dataset_root_PATH + r"Physics_preprocessed_dataset_with_outliers",
                BIOTECH: dataset_root_PATH + r"Biotech_preprocessed_dataset_with_outliers",
                MANAGEMENT: dataset_root_PATH + r"Management_preprocessed_dataset_with_outliers",
                CIVIL: dataset_root_PATH + r"Civil_preprocessed_dataset_with_outliers"}

# get original dataframes
original_df = {PHYSICS: pd.read_csv(dataset_PATH[PHYSICS]),
                BIOTECH: pd.read_csv(dataset_PATH[BIOTECH]),
                MANAGEMENT: pd.read_csv(dataset_PATH[MANAGEMENT]),
                CIVIL: pd.read_csv(dataset_PATH[CIVIL])}
# get only the required columns from each dataframe
original_df = {PHYSICS: original_df[PHYSICS][['Hour', 'Day', 'Total (W)']],
                BIOTECH: original_df[BIOTECH][['Hour', 'Day', 'Total (W)']],
                MANAGEMENT: original_df[MANAGEMENT][['Hour', 'Day', 'Total (W)']],
                CIVIL: original_df[CIVIL][['Hour', 'Day', 'Total (W)']]}

# model class
class LSTM(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, drop_out=0.0):
    super(LSTM, self).__init__()
    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=drop_out)
    self.fc = nn.Linear(hidden_size, 1)

  def forward(self, x):
    out, _ = self.lstm(x)
    out = self.fc(out)
    return out
  
class CNNLSTM(nn.Module):
  def __init__(self, input_features, channels, hidden_size, num_lstm_layers, kernel_size, drop_out=0.0):
    super(CNNLSTM, self).__init__()
    self.conv1 = nn.Conv1d(in_channels=input_features, out_channels=channels, kernel_size=kernel_size, padding='same')
    self.pool = nn.MaxPool1d(kernel_size=kernel_size, stride=1, padding=int((kernel_size-1)/2))
    # self.conv2 = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding='same')
    self.lstm = nn.LSTM(channels, hidden_size, num_layers=num_lstm_layers, bias=True, batch_first=True, dropout=drop_out)
    self.fc = nn.Linear(hidden_size, 1)

  def forward(self, x):
    out = self.pool(F.relu(self.conv1(x)))
    # out = self.pool(F.relu(self.conv2(x)))
    out = out.permute(0, 2, 1)
    out, _ = self.lstm(out)
    out = self.fc(out)
    return out
  

# load model
PATH = {PHYSICS:'physics_cnnlstm_mip.pth', 
        BIOTECH:'biotech_cnnlstm_mip.pth', 
        CIVIL: 'civil_cnnlstm_mip.pth', 
        MANAGEMENT: 'management_cnnlstm_mip.pth'}

# create corresponding load models
models = {PHYSICS: CNNLSTM(input_features=3, channels=54, hidden_size=186, num_lstm_layers=1, kernel_size=3, drop_out=0.0),
          BIOTECH: CNNLSTM(input_features=3, channels=86, hidden_size=120, num_lstm_layers=1, kernel_size=7, drop_out=0.0),
          MANAGEMENT: CNNLSTM(input_features=3, channels=61, hidden_size=71, num_lstm_layers=1, kernel_size=5, drop_out=0.0),
          CIVIL: CNNLSTM(input_features=3, channels=41, hidden_size=163, num_lstm_layers=3, kernel_size=3, drop_out=0.0)}

for key, value in models.items():
  models[key].load_state_dict(torch.load(PATH[key]))
  models[key].eval()

def get_predicted_power(total_power, building:int):
  timestamp = pd.Timestamp(datetime.datetime.now())
  # add 5 hours and 45 minutes to get Nepal Time
  timestamp=  timestamp + pd.Timedelta(hours=5, minutes=45)
  # get hour and weekday from timestamp
  Hour = timestamp.hour
  Day = timestamp.day_name()
  # Day = weekday_encoding[Day]

  input_dict = {"Hour": [Hour], "Day": [Day], "Total (W)": [total_power]}
  input_df = pd.DataFrame(input_dict)

  new_df = pd.concat([original_df[building], input_df], ignore_index=True)
  new_df['Day'] = new_df['Day'].map(weekday_encoding)
  columns_to_scale = ['Hour', 'Day', 'Total (W)']
  scaler = MinMaxScaler(feature_range=(-1, 1))
  new_df[columns_to_scale] = scaler.fit_transform(new_df[columns_to_scale])

  last_row = new_df.iloc[-1]
  last_row = last_row.values.reshape(1, -1)

  last_row = last_row.reshape(-1, 3, 1)

  input_tensor = torch.tensor(last_row, dtype=torch.float32)
  # print(building, input_tensor.shape)

  model = models[building]
  with torch.no_grad():
      output = model(input_tensor)
      output = output.numpy()
      predicted_total_power = [output[0][0][0]]
      predicted_total_power = np.array(predicted_total_power)
      predicted_total_power = np.expand_dims(predicted_total_power, axis=1)
      predicted_total_power = np.repeat(predicted_total_power, 3, axis=1)
      predicted_total_power = scaler.inverse_transform(predicted_total_power)
      predicted_total_power = predicted_total_power[:, 2]


  return int(predicted_total_power[0])