import torch.nn as nn
import torch

from src.Training.TimeSeriesTrainer import TimeSeriesTrainer
from src.time_series_transformer.ETTh1Dataset import get_etth1_dataloader
from src.time_series_transformer.ETTh1_pretraining import get_etth1_pretrain_dataloader
from src.time_series_transformer.TimeSeriesTransformer import TimeSeriesTransformer
from src.encoder.Masking import BertMasking

sequence_length =24*4
num_encoder_blocks = 4
head_sizes = [64, 64, 64]
ETTh1_csv = './ETTh1.csv'
num_variables = 7

learning_rate = 1e-3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_head_sizes = [head_sizes for _ in range (num_encoder_blocks)]

pretraining_dataloader = get_etth1_pretrain_dataloader(ETTh1_csv,
                                                       seq_length=sequence_length,
                                                       mask_size=1,
                                                       mask_value=-999)

forecasting_dataloader = get_etth1_dataloader(ETTh1_csv,
                                        seq_length=sequence_length,
                                        pred_length=1,
                                        batch_size=32)

print(f'Number of batches: {len(forecasting_dataloader)}')

pretaining_masking = BertMasking(replace_token=-99, probability=0.4)
model = TimeSeriesTransformer(input_size=num_variables,
                              head_sizes=input_head_sizes,
                              output_size=num_variables,
                              mask=pretaining_masking)

print(f'Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

timeSeriesTrainer = TimeSeriesTrainer(model, optimizer, pretraining_dataloader, device=device)

timeSeriesTrainer.train(10)



