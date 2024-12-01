import os
import joblib 

import pandas as pd
import polars as pl
import lightgbm as lgb
import xgboost as xgb
import catboost as cbt
import numpy as np 

from joblib import Parallel, delayed

# import kaggle_evaluation.jane_street_inference_server

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

