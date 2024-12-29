from typing import List, Dict, Any, Union, Tuple
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import catboost as cbt
import joblib
import gc
from tqdm import tqdm
import numpy as np
import polars as pl
import sys
import os

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from data.kaggle_evaluation.jane_street_inference_server import *

def r2_xgb(y_true, y_pred, sample_weight):
   """
   为 XGBoost 计算加权 R2 分数的自定义评估指标
   
   Args:
       y_true: 真实标签值
       y_pred: 模型预测值  
       sample_weight: 样本权重

   Returns:
       float: 负的 R2 分数（XGBoost 默认最小化损失，所以返回负值）
       
   Note:
       R2 = 1 - weighted_mse(y_true, y_pred) / weighted_var(y_true)
       分母加入小值 1e-38 避免除零错误
   """
   r2 = 1 - np.average((y_pred - y_true) ** 2, weights=sample_weight) / (np.average((y_true) ** 2, weights=sample_weight) + 1e-38)
   return -r2

def r2_lgb(y_true, y_pred, sample_weight):
   """
   为 LightGBM 计算加权 R2 分数的自定义评估指标
   
   Args:
       y_true: 真实标签值
       y_pred: 模型预测值
       sample_weight: 样本权重

   Returns:
       tuple: ('r2', r2_score, is_higher_better)
       - 指标名称
       - R2 分数 
       - 是否更大的值更好
   """
   r2 = 1 - np.average((y_pred - y_true) ** 2, weights=sample_weight) / (np.average((y_true) ** 2, weights=sample_weight) + 1e-38)
   return 'r2', r2, True

class r2_cbt(object):
   """
   为 CatBoost 计算 R2 分数的自定义评估指标类
   
   实现 CatBoost 自定义评估指标所需的接口:
   - get_final_error: 计算最终的评估分数
   - is_max_optimal: 指示是否更大的值更好
   - evaluate: 计算误差和权重
   """
   
   def get_final_error(self, error, weight):
       """
       计算最终的 R2 分数
       
       Args:
           error: evaluate() 返回的误差和
           weight: evaluate() 返回的权重和
           
       Returns:
           float: R2 分数
       """
       return 1 - error / (weight + 1e-38)

   def is_max_optimal(self):
       """
       指示是否更大的评估分数更好
       
       Returns:
           bool: True 表示更大的值更好
       """
       return True

   def evaluate(self, approxes, target, weight):
       """
       计算误差平方和与目标值平方和
       
       Args:
           approxes: 预测值列表的列表
           target: 真实标签值
           weight: 样本权重
           
       Returns:
           tuple: (error_sum, weight_sum)
           - error_sum: 加权误差平方和
           - weight_sum: 加权目标值平方和
           
       Note:
           CatBoost 的接口要求 approxes 是预测值的列表，
           但此实现只使用第一个预测值列表
       """
       assert len(approxes) == 1
       assert len(target) == len(approxes[0])

       approx = approxes[0]
       error_sum = 0.0  # 加权误差平方和
       weight_sum = 0.0  # 加权目标值平方和

       for i in range(len(approx)):
           w = 1.0 if weight is None else weight[i]
           weight_sum += w * (target[i] ** 2)
           error_sum += w * ((approx[i] - target[i]) ** 2)

       return error_sum, weight_sum

def reduce_mem_usage(df: pd.DataFrame, float16_as32: bool = True) -> pd.DataFrame:
    """
    通过调整数据类型来减少 DataFrame 的内存使用。
    
    Args:
        df: 输入的 DataFrame
        float16_as32: 是否将 float16 范围内的数据转换为 float32 以提高精度
        
    Returns:
        优化内存使用后的 DataFrame
        
    Note:
        数值类型转换范围：
        - int8:   -128 到 127
        - int16:  -32,768 到 32,767
        - int32:  -2,147,483,648 到 2,147,483,647
        - int64:  -9,223,372,036,854,775,808 到 9,223,372,036,854,775,807
        - float16: ±6.55e±4
        - float32: ±3.4e±38
        - float64: ±1.7e±308
    """
    # 计算初始内存使用量（MB）
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype
        
        # 跳过非数值类型的列（对象类型和类别类型）
        if col_type != object and str(col_type) != 'category':
            c_min, c_max = df[col].min(), df[col].max()
            
            # 处理整数类型
            if str(col_type)[:3] == 'int':
                # 根据数据范围选择最小的可用整数类型
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            
            # 处理浮点数类型
            else:
                # 根据数据范围选择最小的可用浮点类型
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    if float16_as32:
                        # 对精度敏感的数据使用 float32
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    # 计算优化后的内存使用量和节省比例
    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

def prepare_datasets(
    df: pd.DataFrame,
    feature_names: List[str],
    dates: np.ndarray,
    num_valid_dates: int
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    准备训练集和验证集的数据
    
    Args:
        df: 原始数据框
        feature_names: 特征列名列表
        dates: 唯一日期数组
        num_valid_dates: 验证集使用的日期数量
    
    Returns:
        训练集和验证集的字典，包含特征(X)、标签(y)和权重(w)
    """
    # 选择最后 num_valid_dates 个日期作为验证集, 其余作为训练集
    # 例如，如果 num_valid_dates=10，dates=[0, 1, 2, ..., 99]，则 valid_dates=[90, 91, ..., 99]
    valid_dates = dates[-num_valid_dates:]
    train_dates = dates[:-num_valid_dates]
    
    # 准备验证集，只包含 valid_dates 中的数据
    # X: 特征, y: 标签, w: 权重
    valid_mask = df['date_id'].isin(valid_dates)
    valid_data = {
        'X': df[feature_names][valid_mask],
        'y': df['responder_6'][valid_mask],
        'w': df['weight'][valid_mask]
    }
    
    # 准备基础训练集数据, 同上
    train_data = {
        'dates': train_dates,
        'X': df[feature_names],
        'y': df['responder_6'],
        'w': df['weight']
    }
    
    return train_data, valid_data

def get_fold_data(
    df: pd.DataFrame,
    feature_names: List[str],
    train_dates: List[int],
    fold_idx: int,
    n_folds: int
) -> Dict[str, pd.DataFrame]:
    """
    获取指定折的训练数据
    
    Args:
        df: 原始数据框
        train_dates: 训练集日期列表
        fold_idx: 当前折索引
        n_folds: 总折数
    
    Returns:
        当前折的训练数据字典
    """
    # 把 train_dates 分成 n_folds 份，取除了第 fold_idx 份之外的数据作为训练数据
    selected_dates = [date for i, date in enumerate(train_dates) if i % n_folds != fold_idx]
    mask = df['date_id'].isin(selected_dates)
    
    return {
        'X': df[feature_names][mask],
        'y': df['responder_6'][mask],
        'w': df['weight'][mask],
    }

def train_model(
    model: Union[lgb.LGBMRegressor, xgb.XGBRegressor, cbt.CatBoostRegressor],
    train_data: Dict[str, pd.DataFrame],
    valid_data: Dict[str, pd.DataFrame],
    model_type: str,
) -> Union[lgb.LGBMRegressor, xgb.XGBRegressor, cbt.CatBoostRegressor]:
    """
    训练模型
    
    Args:
        model: 模型实例
        train_data: 训练数据字典
        valid_data: 验证数据字典
        model_type: 模型类型 ('lgb', 'xgb', 或 'cbt')
    
    Returns:
        训练好的模型
    """
    if model_type == 'lgb':
        model.fit(
            train_data['X'], train_data['y'], 
            sample_weight=train_data['w'],
            eval_metric=[r2_lgb],
            eval_set=[(valid_data['X'], valid_data['y'], valid_data['w'])],
            callbacks=[
                lgb.early_stopping(100),
                lgb.log_evaluation(10)
            ]
        )
    elif model_type == 'cbt':
        evalset = cbt.Pool(valid_data['X'], valid_data['y'], weight=valid_data['w'])
        model.fit(
            train_data['X'], train_data['y'],
            sample_weight=train_data['w'],
            eval_set=[evalset],
            verbose=10,
        )
    elif model_type == 'xgb':  # xgb
        model.fit(
            train_data['X'], train_data['y'],
            sample_weight=train_data['w'],
            eval_set=[(valid_data['X'], valid_data['y'])],
            sample_weight_eval_set=[valid_data['w']],
            verbose=10,
        )
    else:
        model = None
        
    return model

def train_main():
    """
    主函数
    """
    # 配置参数
    input_path = "data"
    TRAINING = True
    N_fold = 5
    num_valid_dates = 10
    skip_dates = 100
    feature_nums = 79
    all_feature_names = [f"feature_{i:02d}" for i in range(feature_nums)]     # 记录所有特征的名字，从'feature_00' 到 'feature_79'
    models_toUse = ['xgb']     # 训练的模型类型
    
    #* 根据 features.csv，计算 correlation matrix, 筛除冗余特征
    features_tags = pd.read_csv(f"{input_path}/features.csv")
        
    correlation_matrix = features_tags[[ f"tag_{no}" for no in range(0,17,1) ]].T.corr()
    correlation_threshold = 0.8
    to_drop = set()
    fullset = set(range(feature_nums))
    # 遍历 correlation matrix，把相关性大于阈值的特征加入到 to_drop 中
    for i in range(feature_nums):
        for j in range(i+1, feature_nums):
            if correlation_matrix.iloc[i, j] > correlation_threshold:
                feature_to_drop = correlation_matrix.columns[j]
                to_drop.add(feature_to_drop)
    # 获取剩余特征的名字
    feature_names = list(fullset - to_drop)
    feature_names = [all_feature_names[i] for i in feature_names]
    
    # 记录所有使用的模型，以及每一个模型所采取的不同配置
    model_dict = {
        'lgb': lgb.LGBMRegressor(n_estimators=500, 
                                 device='cpu',  #! 对于苹果芯片，需要设置为 'cpu'
                                 gpu_use_dp=True, 
                                 objective='l2'),
        'xgb': xgb.XGBRegressor(n_estimators=2000,
                                learning_rate=0.02,
                                max_depth=10,
                                tree_method='hist',
                                device="cpu",   #! 对于苹果芯片，需要设置为 'cpu'
                                objective='reg:squarederror',   # 目标：均方误差
                                eval_metric=r2_xgb,
                                disable_default_eval_metric=True,
                                early_stopping_rounds=100,
                                callbacks=[
                                    xgb.callback.EvaluationMonitor(show_stdv=False)  # 只显示均值，不显示标准差
                                ]
                            ),
        'cbt': cbt.CatBoostRegressor(iterations=1000, 
                                     learning_rate=0.05, 
                                     task_type='CPU',   #! 对于苹果芯片，需要设置为 'CPU'，而不是 'GPU'
                                     loss_function='RMSE',
                                     verbose=True,       # 启用详细日志
                                     metric_period=1,
                                     early_stopping_rounds=100,
                                     eval_metric=r2_cbt(),   # 设置了 eval_metric 之后，只会打印这个指标
                                    )
    }
    
    if TRAINING:
        # 加载数据
        # df = pd.read_parquet(f'{input_path}/train.parquet')   # 加载全部数据
        df = pd.read_parquet(
            f'{input_path}/train.parquet', 
            filters=[('partition_id', 'in', [4, 5, 6])]     # 只加载 partition_id 为 4, 5, 6 的数据
        )
        df = reduce_mem_usage(df, False)
        df = df[df['date_id'] >= skip_dates].reset_index(drop=True)
        
        print("----------- Start to Load Dataset! -----------")
        # 准备数据集
        dates = df['date_id'].unique()
        train_data, valid_data = prepare_datasets(df, feature_names, dates, num_valid_dates)
        print("----------- Dataset Loaded! -----------")
        
        models = []
        for model_type in models_toUse:
            # 训练模型
            model = model_dict[model_type]
            trained_model = train_model(model, train_data, valid_data, model_type)
            models.append(trained_model)
            
            # 保存模型
            joblib.dump(trained_model, f'./models/{model_type}.model')
            
            # 清理内存
            # del train_data
            gc.collect()
        
        return models
        
        
        # 训练模型
        models = []
        for fold_idx in tqdm(range(N_fold), desc="Training: "):
            for model_type in models_toUse:
                # 获取当前折的训练数据
                fold_train_data = get_fold_data(df, feature_names, train_data['dates'], fold_idx, N_fold)
                
                # 获取模型实例
                model = model_dict[model_type]
                
                # 训练模型
                trained_model = train_model(model, fold_train_data, valid_data, model_type)
                models.append(trained_model)
                
                # 保存
                joblib.dump(trained_model, f'./models/{model_type}_{fold_idx}.model')
                
                # 清理内存
                # del fold_data
                gc.collect()
    else:
        # 加载预训练模型
        models = []
        model_ckpt_path = "./jsbaselinezyz/versions/1"
        for fold_idx in range(N_fold):
            for model_type in ['lgb', 'xgb', 'cbt']:
                model = joblib.load(f'{model_ckpt_path}/{model_type}_{fold_idx}.model')
                models.append(model)
    return models

if __name__ == "__main__":    
    # 运行主程序
    models = train_main()