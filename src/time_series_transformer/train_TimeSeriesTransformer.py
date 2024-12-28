import torch.nn as nn
import torch

from src.Training.TimeSeriesTrainer import TimeSeriesTrainer
from src.time_series_transformer.ETTh1Dataset import get_etth1_dataloader
from src.time_series_transformer.ETTh1_pretraining import get_etth1_pretrain_dataloader
from src.time_series_transformer.TimeSeriesTransformer import TimeSeriesTransformer, ReconstructionModule, RegressionModule
from src.encoder.Masking import BertMasking

#set training variables
sequence_length =24*4
num_encoder_blocks = 1
head_sizes = [64, 64]
ETTh1_csv = './ETTh1.csv'
num_variables = 7
hidden_size_regressor = 32

learning_rate = 1e-3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_head_sizes = [head_sizes for _ in range (num_encoder_blocks)]

#get pretraining dataloader
pretraining_dataloader = get_etth1_pretrain_dataloader(ETTh1_csv,
                                                       seq_length=sequence_length,
                                                       mask_size=int(96*0.2),
                                                       mask_value=-99)

#get regression dataloader
regression_dataloader = get_etth1_dataloader(ETTh1_csv,
                                        seq_length=sequence_length,
                                        pred_length=1,
                                        batch_size=32)

print(f'Number of batches: {len(pretraining_dataloader)}')

#initialize model
pretraining_masking = BertMasking(replace_token=-1, probability=0.0)
model = TimeSeriesTransformer(input_size=num_variables,
                              head_sizes=input_head_sizes,
                              output_size=num_variables,
                              mask=pretraining_masking,
                              task='reconstruction')

print(f'Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

#initialize optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#pretrain model
timeSeriesTrainer = TimeSeriesTrainer(model, optimizer, pretraining_dataloader, device=device)
timeSeriesTrainer.train(10)

#switch to regression task
timeSeriesTrainer.model.output = RegressionModule(num_variables, hidden_size_regressor, num_variables)
timeSeriesTrainer.train_loader = regression_dataloader

timeSeriesTrainer.train(10)



