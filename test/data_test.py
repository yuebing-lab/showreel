import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import numpy as np
from skimage.transform import resize
from scipy import stats

lst_file = r'D:\paq\GEE\problem\lst\bj19_LST.tif'
with rasterio.open(lst_file) as src:
    lst_data = src.read()
    print(lst_data)

squeezed_matrix = np.squeeze(lst_data)  # 去除单维度变为(1943, 2511)
flatten_data = squeezed_matrix.flatten()
valid_data = flatten_data[~np.isnan(flatten_data)]

# 计算统计量
statistics = {
    "数据总量": len(valid_data),
    "空值数量": len(flatten_data) - len(valid_data),
    "最小值（0分位数）": np.min(valid_data),
    "10分位数": np.percentile(valid_data, 10),
    "中位数（50分位数）": np.median(valid_data),
    "90分位数": np.percentile(valid_data, 90),
    "平均数": np.mean(valid_data),
    "标准差": np.std(valid_data),
}


# 打印结果
print("统计特征报告：")
print("="*40)
for k, v in statistics.items():
    print(f"{k:.<20}: {v:.6f}" if isinstance(v, float) else f"{k:.<20}: {v}")

print("\n注意事项：")
print("- 众数计算基于四舍五入到4位小数的结果")
print("- 当多个值出现次数相同时，返回最先出现的最小值")
print("- 原始数据空值占比：{:.2%}".format(statistics["空值数量"]/len(flatten_data)))