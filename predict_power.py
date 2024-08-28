import torch
import torch.nn as nn
import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

weekday_encoding = {'Sunday': 1, 'Monday': 2, 'Tuesday': 3, 'Wednesday': 4, 'Thursday': 5, 'Friday': 6, 'Saturday': 7}

dataset_PATH = r"C:\Users\nabin\OneDrive\Desktop\AI Gridview\[0] Datasets\Biotech_preprocessed_dataset_with_outliers.csv"
original_df = pd.read_csv(dataset_PATH)
original_df = original_df[['Hour', 'Day', 'Total (W)']]

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
  

# load model
PATH = 'lstm_mip.pth'
model = LSTM(input_size=3, hidden_size=75, num_layers=2, drop_out=0.0)
model.load_state_dict(torch.load(PATH))
model.eval()

def get_predicted_power(total_power):
  timestamp = pd.Timestamp(datetime.datetime.now())
  # add 5 hours and 45 minutes to get Nepal Time
  timestamp=  timestamp + pd.Timedelta(hours=5, minutes=45)
  # get hour and weekday from timestamp
  Hour = timestamp.hour
  Day = timestamp.day_name()
  # Day = weekday_encoding[Day]

  input_dict = {"Hour": [Hour], "Day": [Day], "Total (W)": [total_power]}
  input_df = pd.DataFrame(input_dict)

  new_df = pd.concat([original_df, input_df], ignore_index=True)
  new_df['Day'] = new_df['Day'].replace(weekday_encoding)
  columns_to_scale = ['Hour', 'Day', 'Total (W)']
  scaler = MinMaxScaler(feature_range=(-1, 1))
  new_df[columns_to_scale] = scaler.fit_transform(new_df[columns_to_scale])

  last_row = new_df.iloc[-1]
  last_row = last_row.values.reshape(1, -1)

  last_row = last_row.reshape(-1, 1, 3)

  input_tensor = torch.tensor(last_row, dtype=torch.float32)

  with torch.no_grad():
      output = model(input_tensor)
      output = output.numpy()
      predicted_total_power = [output[0][0][0]]
      predicted_total_power = np.array(predicted_total_power)
      predicted_total_power = np.expand_dims(predicted_total_power, axis=1)
      predicted_total_power = np.repeat(predicted_total_power, 3, axis=1)
      predicted_total_power = scaler.inverse_transform(predicted_total_power)
      predicted_total_power = predicted_total_power[:, 2]
      print(timestamp)


  return int(predicted_total_power[0])