import torch.nn as nn
import torch

from src.Training.TimeSeriesTrainer import TimeSeriesTrainer
from src.time_series_transformer.TimeSeriesTransformerTestTrain import get_etth1_dataloader

class LinearRegression(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(num_inputs, num_outputs))
    
    def forward(self, x):
        return self.model(x)

sequence_length = 4*24
num_variables = 7
learning_rate = 1e-4
ETTh1_csv = './ETTh1.csv'

train_dataloader = get_etth1_dataloader(ETTh1_csv,
                                        seq_length=sequence_length,
                                        pred_length=1,
                                        batch_size=32)

benchmark_model = LinearRegression(num_inputs=sequence_length*num_variables, num_outputs=num_variables)
optimizer = torch.optim.Adam(benchmark_model.parameters(), lr=learning_rate)
    
linearRegressionTrainer = TimeSeriesTrainer(benchmark_model, optimizer, train_dataloader)

linearRegressionTrainer.train(10)

