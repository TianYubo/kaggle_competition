import os
import joblib
import gc
import pandas as pd
import polars as pl
import lightgbm as lgb
import xgboost as xgb
import catboost as cbt
import numpy as np 
from typing import Union

from tqdm import tqdm
from joblib import Parallel, delayed
# from data.kaggle_evaluation import jane_street_inference_server

# ---------------------- Functions ----------------------
# Custom R2 metric for XGBoost
def r2_xgb(y_true, y_pred, sample_weight):
    r2 = 1 - np.average((y_pred - y_true) ** 2, weights=sample_weight) / (np.average((y_true) ** 2, weights=sample_weight) + 1e-38)
    return -r2

# Custom R2 metric for LightGBM
def r2_lgb(y_true, y_pred, sample_weight):
    r2 = 1 - np.average((y_pred - y_true) ** 2, weights=sample_weight) / (np.average((y_true) ** 2, weights=sample_weight) + 1e-38)
    return 'r2', r2, True

# Custom R2 metric for CatBoost
class r2_cbt(object):
    def get_final_error(self, error, weight):
        return 1 - error / (weight + 1e-38)

    def is_max_optimal(self):
        return True

    def evaluate(self, approxes, target, weight):
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])

        approx = approxes[0]
        error_sum = 0.0
        weight_sum = 0.0

        for i in range(len(approx)):
            w = 1.0 if weight is None else weight[i]
            weight_sum += w * (target[i] ** 2)
            error_sum += w * ((approx[i] - target[i]) ** 2)

        return error_sum, weight_sum
    
def reduce_mem_usage(df, float16_as32=True):
    #memory_usage()是df每列的内存使用量,sum是对它们求和, B->KB->MB
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:#遍历每列的列名
        col_type = df[col].dtype#列名的type
        if col_type != object and str(col_type)!='category':#不是object也就是说这里处理的是数值类型的变量
            c_min,c_max = df[col].min(),df[col].max() #求出这列的最大值和最小值
            if str(col_type)[:3] == 'int':#如果是int类型的变量,不管是int8,int16,int32还是int64
                #如果这列的取值范围是在int8的取值范围内,那就对类型进行转换 (-128 到 127)
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                #如果这列的取值范围是在int16的取值范围内,那就对类型进行转换(-32,768 到 32,767)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                #如果这列的取值范围是在int32的取值范围内,那就对类型进行转换(-2,147,483,648到2,147,483,647)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                #如果这列的取值范围是在int64的取值范围内,那就对类型进行转换(-9,223,372,036,854,775,808到9,223,372,036,854,775,807)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:#如果是浮点数类型.
                #如果数值在float16的取值范围内,如果觉得需要更高精度可以考虑float32
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    if float16_as32:#如果数据需要更高的精度可以选择float32
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float16)  
                #如果数值在float32的取值范围内，对它进行类型转换
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                #如果数值在float64的取值范围内，对它进行类型转换
                else:
                    df[col] = df[col].astype(np.float64)
    #计算一下结束后的内存
    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    #相比一开始的内存减少了百分之多少
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df

# Function to train a model or load a pre-trained model
def train(model_dict, model_name='lgb', nfold_idx=0, model_ckpt=None, train_mode=False):
    if train_mode:
        # Select dates for training based on the fold number
        selected_dates = [date for ii, date in enumerate(train_dates) if ii % N_fold != i]
        
        # Get the model from the dictionary
        model = model_dict[model_name]
        
        # Extract features, target, and weights for the selected training dates
        X_train = df[feature_names].loc[df['date_id'].isin(selected_dates)]
        y_train = df['responder_6'].loc[df['date_id'].isin(selected_dates)]
        w_train = df['weight'].loc[df['date_id'].isin(selected_dates)]

        # Train the model based on the type (LightGBM, XGBoost, or CatBoost)
        if model_name == 'lgb':
            # Train LightGBM model with early stopping and evaluation logging
            model.fit(X_train, y_train, w_train,  
                      eval_metric=[r2_lgb],
                      eval_set=[(X_valid, y_valid, w_valid)], 
                      callbacks=[
                          lgb.early_stopping(100), 
                          lgb.log_evaluation(10)
                      ])
            
        elif model_name == 'cbt':
            # Prepare evaluation set for CatBoost
            evalset = cbt.Pool(X_valid, y_valid, weight=w_valid)
            
            # Train CatBoost model with early stopping and verbose logging
            model.fit(X_train, y_train, sample_weight=w_train, 
                      eval_set=[evalset], 
                      verbose=10, 
                      early_stopping_rounds=100)
            
        else:
            # Train XGBoost model with early stopping and verbose logging
            model.fit(X_train, y_train, sample_weight=w_train, 
                      eval_set=[(X_valid, y_valid)], 
                      sample_weight_eval_set=[w_valid], 
                      verbose=10, 
                      early_stopping_rounds=100)
        # Append the trained model to the list
        models.append(model)
        # Save the trained model to a file
        joblib.dump(model, f'./models/{model_name}_{i}.model')
        
        # Delete training data to free up memory
        del X_train
        del y_train
        del w_train
        
        # Collect garbage to free up memory
        gc.collect()
        
    else:
        # If not in training mode, load the pre-trained model from the specified path
        models.append(joblib.load(f'{model_ckpt}/{model_name}_{nfold_idx}.model'))
    return 

lags_ = None

# Replace this function with your inference code.
# You can return either a Pandas or Polars dataframe, though Polars is recommended.
# Each batch of predictions (except the very first) must be returned within 10 minutes of the batch features being provided.
def predict(test: pl.DataFrame, lags: Union[pl.DataFrame, None]) -> Union[pl.DataFrame, pd.DataFrame]:
    """Make a prediction."""
    # All the responders from the previous day are passed in at time_id == 0. We save them in a global variable for access at every time_id.
    # Use them as extra features, if you like.
    global lags_
    if lags is not None:
        lags_ = lags

    predictions = test.select(
        'row_id',
        pl.lit(0.0).alias('responder_6'),
    )
    
    feat = test[feature_names].to_numpy()
    
    pred = [model.predict(feat) for model in models]
    pred = np.mean(pred, axis=0)
    
    predictions = predictions.with_columns(pl.Series('responder_6', pred.ravel()))

    # The predict function must return a DataFrame
    assert isinstance(predictions, pl.DataFrame | pd.DataFrame)
    # with columns 'row_id', 'responer_6'
    assert list(predictions.columns) == ['row_id', 'responder_6']
    # and as many rows as the test data.
    assert len(predictions) == len(test)

    return predictions

# ---------------------- Main ----------------------
#! Parameters
input_path = "~/kaggle/jane-street-project/data"
TRAINING = True    # 是否开启训练模式
feature_names = [f"feature_{i:02d}" for i in range(79)]     # 记录所有特征的名字，从'feature_00' 到 'feature_79'
num_valid_dates = 100   # 从数据的最后选取若干帧作为验证集
skip_dates = 500    # 从数据的开始跳过若干帧，取中间帧作为起始点
N_fold = 5  # Number of folds for cross-validation

# 记录所有使用的模型，以及每一个模型所采取的不同配置
model_dict = {
    'lgb': lgb.LGBMRegressor(n_estimators=500, device='gpu', gpu_use_dp=True, objective='l2'),
    'xgb': xgb.XGBRegressor(n_estimators=2000, learning_rate=0.1, max_depth=6, tree_method='hist', device="cuda", objective='reg:squarederror', eval_metric=r2_xgb, disable_default_eval_metric=True),
    'cbt': cbt.CatBoostRegressor(iterations=1000, learning_rate=0.05, task_type='GPU', loss_function='RMSE', eval_metric=r2_cbt()),
}

if TRAINING:
    # 加载并预处理数据
    # df = pd.read_parquet(f'{input_path}/train.parquet/partition_id=1/part-0.parquet')
    df = pd.read_parquet(f'{input_path}/train.parquet', filters=[('partition_id', '=', 4)])
    df = reduce_mem_usage(df, False)
    df = df[df['date_id'] >= skip_dates].reset_index(drop=True)
    
    # 划分训练集和验证集
    print("----------- Start to Load Dataset! -----------")
    # 获取唯一日期并划分
    dates = df['date_id'].unique()  # 获取所有唯一日期
    valid_dates = dates[-num_valid_dates:]  # 最后 n 个日期作为验证集
    train_dates = dates[:-num_valid_dates]  # 剩余日期作为训练集

    # 构建验证集，取对应的列，并且 date_id 需要在 valid_dates 当中
    X_valid = df[feature_names][df['date_id'].isin(valid_dates)]  # 验证集特征
    y_valid = df['responder_6'][df['date_id'].isin(valid_dates)]  # 验证集标签
    w_valid = df['weight'][df['date_id'].isin(valid_dates)]      # 验证集权重
    print("----------- Dataset Loaded! -----------")

models = []
model_ckpt_path = "./jsbaselinezyz/versions/1"
# Train models for each fold
for i in tqdm(range(N_fold), desc="Training: ", total=N_fold):
    train(model_dict, train_dates, 'lgb', i, model_ckpt_path, TRAINING)
    train(model_dict, 'xgb', i, model_ckpt_path, TRAINING)
    train(model_dict, 'cbt', i, model_ckpt_path, TRAINING)
    

