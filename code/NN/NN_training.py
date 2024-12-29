import os
import pickle
import polars as pl
import numpy as np
import pandas as pd

import os
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning
from pytorch_lightning import (LightningDataModule, LightningModule, Trainer)
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Timer
from pytorch_lightning.loggers import WandbLogger
import wandb
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import gc

class custom_args():
    def __init__(self):
        self.usegpu = False
        self.accelerator = 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.gpuid = 0
        self.seed = 42
        self.model = 'nn'
        self.use_wandb = False
        self.project = 'js-xs-nn-with-lags'
        self.dname = "./code/"
        self.loader_workers = 0
        self.bs = 4096
        self.lr = 1e-3
        self.weight_decay = 5e-4
        self.dropouts = [0.1, 0.1]
        self.n_hidden = [512, 512, 256]
        self.patience = 25
        self.max_epochs = 2000
        self.N_fold = 5
        
class CustomDataset(Dataset):
    def __init__(self, df, accelerator):
        self.features = torch.FloatTensor(df[feature_names].values).to(accelerator)
        self.labels = torch.FloatTensor(df[label_name].values).to(accelerator)
        self.weights = torch.FloatTensor(df[weight_name].values).to(accelerator)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        w = self.weights[idx]
        return x, y, w
    
class DataModule(LightningDataModule):
    """ 数据模块，用于加载数据集并生成数据加载器
    支持 Polars LazyFrame 输入

    Args:
        LightningDataModule ([type]): LightningDataModule
        train_df (pl.LazyFrame | pl.DataFrame): 训练数据，支持 LazyFrame
        batch_size (int): 批次大小
        valid_df (pl.LazyFrame | pl.DataFrame, optional): 验证数据，支持 LazyFrame
        accelerator (str, optional): 计算设备. Defaults to 'cpu'.
    """
    def __init__(self, train_df, batch_size, valid_df=None, accelerator='cpu'):
        super().__init__()
        self.df = train_df
        self.batch_size = batch_size
        # 收集唯一日期ID
        self.dates = self.df['date_id'].unique()
        self.accelerator = accelerator
        self.train_dataset = None
        self.valid_df = valid_df
        self.val_dataset = None

    def setup(self, fold=0, N_fold=5, stage=None):
        selected_dates = [date for ii, date in enumerate(self.dates) if ii % N_fold != fold]
        df_train = self.df.loc[self.df['date_id'].isin(selected_dates)]
        self.train_dataset = CustomDataset(df_train, self.accelerator)
        if self.valid_df is not None:
            df_valid = self.valid_df
            self.val_dataset = CustomDataset(df_valid, self.accelerator)
            
    def train_dataloader(self, n_workers=0):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=n_workers
        )

    def val_dataloader(self, n_workers=0):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=n_workers
        )
        
        
# Custom R2 metric for validation
def r2_val(y_true, y_pred, sample_weight):
    r2 = 1 - np.average((y_pred - y_true) ** 2, weights=sample_weight) / (np.average((y_true) ** 2, weights=sample_weight) + 1e-38)
    return r2

class NN(LightningModule):
    def __init__(self, input_dim, hidden_dims, dropouts, lr, weight_decay):
        super().__init__()
        self.save_hyperparameters()
        layers = []
        in_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.BatchNorm1d(in_dim))
            if i > 0:
                layers.append(nn.SiLU())
            if i < len(dropouts):
                layers.append(nn.Dropout(dropouts[i]))
            layers.append(nn.Linear(in_dim, hidden_dim))
            # layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1)) 
        layers.append(nn.Tanh())
        self.model = nn.Sequential(*layers)
        self.lr = lr
        self.weight_decay = weight_decay
        self.validation_step_outputs = []

    def forward(self, x):
        return 5 * self.model(x).squeeze(-1)  

    def training_step(self, batch):
        x, y, w = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y, reduction='none') * w  #
        loss = loss.mean()
        self.log('train_loss', loss, on_step=False, on_epoch=True, batch_size=x.size(0))
        return loss

    def validation_step(self, batch):
        x, y, w = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y, reduction='none') * w
        loss = loss.mean()
        self.log('val_loss', loss, on_step=False, on_epoch=True, batch_size=x.size(0))
        self.validation_step_outputs.append((y_hat, y, w))
        return loss

    def on_validation_epoch_end(self):
        """Calculate validation WRMSE at the end of the epoch."""
        y = torch.cat([x[1] for x in self.validation_step_outputs]).cpu().numpy()
        if self.trainer.sanity_checking:
            prob = torch.cat([x[0] for x in self.validation_step_outputs]).cpu().numpy()
        else:
            prob = torch.cat([x[0] for x in self.validation_step_outputs]).cpu().numpy()
            weights = torch.cat([x[2] for x in self.validation_step_outputs]).cpu().numpy()
            # r2_val
            val_r_square = r2_val(y, prob, weights)
            self.log("val_r_square", val_r_square, prog_bar=True, on_step=False, on_epoch=True)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5,
                                                               verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
            }
        }

    def on_train_epoch_end(self):
        if self.trainer.sanity_checking:
            return
        epoch = self.trainer.current_epoch
        metrics = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in self.trainer.logged_metrics.items()}
        formatted_metrics = {k: f"{v:.5f}" for k, v in metrics.items()}
        print(f"Epoch {epoch}: {formatted_metrics}")

if __name__ == "__main__":

    nn_config_args = custom_args()
    # input_path = './kaggle_competition-main/code/' if os.path.exists('./kaggle_competition-main/code/') else '/Users/victoriazhang/kaggle/kaggle_competition-main/code/'
    input_path = '/Users/kyleee/code/project/kaggle_competition/code/xgboost'
    TRAINING = True
    feature_names = ["symbol_id","time_id"] + [f"feature_{i:02d}" for i in range(79)] + [f"responder_{idx}_lag_1" for idx in range(9)]
    label_name = 'responder_6'
    weight_name = 'weight'

    # train_name = os.path.join("./code/", "nn_input_df_with_lags.pickle")
    # valid_name = os.path.join("./code/", "nn_valid_df_with_lags.pickle")

    train_name = 'nn_input_df_with_lags.pickle'
    valid_name = 'nn_valid_df_with_lags.pickle'

    #! Load data
    if TRAINING and not os.path.exists(train_name):
        df = pl.scan_parquet(f"{input_path}/training.parquet").collect().to_pandas()
        valid = pl.scan_parquet(f"{input_path}/validation.parquet").collect().to_pandas()
        # df = pl.scan_parquet(f"{input_path}/training.parquet")
        # valid = pl.scan_parquet(f"{input_path}/validation.parquet")
        # df = pd.concat([df, valid]).reset_index(drop=True)      # A trick to boost LB from 0.0045->0.005
        # with open(train_name, "wb") as w:
        #     pickle.dump(df, w)
        # with open(valid_name, "wb") as w:
        #     pickle.dump(valid, w)
    elif TRAINING:
        with open(train_name, "rb") as r:
            df = pickle.load(r)
        with open(valid_name, "rb") as r:
            valid = pickle.load(r)
            
    df[feature_names] = df[feature_names].fillna(method = 'ffill').fillna(0)
    valid[feature_names] = valid[feature_names].fillna(method = 'ffill').fillna(0)
    
    # df = (df.with_columns([
    #           pl.col(feature_names)
    #             .fill_null(strategy="forward")
    #             .fill_null(0)
    #             .alias(col) for col in feature_names
    #       ]))
    # valid = (valid.with_columns([
    #              pl.col(feature_names)
    #                .fill_null(strategy="forward")
    #                .fill_null(0)
    #                .alias(col) for col in feature_names
    #          ]))
    
    data_module = DataModule(df, 
                             batch_size=nn_config_args.bs, 
                             valid_df=valid, 
                             accelerator=nn_config_args.accelerator)
    
    gc.collect()
    
    for fold in range(nn_config_args.N_fold):
        data_module.setup(fold=fold, N_fold=nn_config_args.N_fold)
        model = NN(input_dim=len(feature_names), 
                    hidden_dims=nn_config_args.n_hidden, 
                    dropouts=nn_config_args.dropouts, 
                    lr=nn_config_args.lr, 
                    weight_decay=nn_config_args.weight_decay)
        early_stopping = EarlyStopping('val_loss', patience=nn_config_args.patience, mode='min')
        checkpoint = ModelCheckpoint(monitor='val_loss', 
                                     mode='min',
                                     save_top_k=1)
        trainer = Trainer(max_epochs=nn_config_args.max_epochs, 
                          accelerator=nn_config_args.accelerator,
                          logger=None, 
                          callbacks=[early_stopping, checkpoint],
                          enable_progress_bar=True,)
        trainer.fit(model, 
                    data_module.train_dataloader(n_workers=nn_config_args.loader_workers),
                    data_module.val_dataloader(n_workers=nn_config_args.loader_workers)
                    )
        
        del model
        gc.collect()
        torch.cuda.empty_cache()
            
    # X_train = df[ feature_names ]
    # y_train = df[ label_name ]
    # w_train = df[ "weight" ]
    # X_valid = valid[ feature_names ]
    # y_valid = valid[ label_name ]
    # w_valid = valid[ "weight" ]

    # print(f"Training set: {X_train.shape[0]} rows, {X_train.shape[1]} columns")
    # print(f"Training set gt: {y_train.shape[0]} rows")
    # print(f"Validation set: {X_valid.shape[0]} rows, {X_valid.shape[1]} columns")
    # print(f"Validation set gt: {y_valid.shape[0]} rows")

    
