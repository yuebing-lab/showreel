import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import numpy as np
from skimage.transform import resize

lst_file = r'D:\paq\GEE\problem\data\lst\bj19_LST.tif'
with rasterio.open(lst_file) as src:
    lst_data = src.read()

compressed_lst = resize(lst_data, (1,1943,2511), mode='reflect', anti_aliasing=True)

GAIA_file = r'D:\paq\GEE\problem\data\GAIA\2018year_bj.tif'
with rasterio.open(GAIA_file) as src:
    GAIA_data = src.read()
    print(GAIA_data.shape)
compressed_GAIA= resize(GAIA_data, (1,1943,2511), mode='reflect', anti_aliasing=True)

ndvi_file = r'D:\paq\GEE\problem\data\ndvi\mean_NDVI_2019.tif'
with rasterio.open(ndvi_file) as src:
    ndvi_data = src.read()
    print(ndvi_data.shape)
compressed_ndvi= resize(ndvi_data, (1,1943,2511), mode='reflect', anti_aliasing=True)

ndwi_file = r'D:\paq\GEE\problem\data\ndwi\mean_NDWI_2019.tif'
with rasterio.open(ndwi_file) as src:
    ndwi_data = src.read()
    print(ndwi_data.shape)
compressed_ndwi= resize(ndwi_data, (1,1943,2511), mode='reflect', anti_aliasing=True)

VIRS_file = r'D:\paq\GEE\problem\data\VIRS\Nighttime_Lights_2019_bj.tif'
with rasterio.open(VIRS_file) as src:
    VIRS_data = src.read()
    print(VIRS_data.shape)
compressed_VIRS= resize(VIRS_data, (1,1943,2511), mode='reflect', anti_aliasing=True)

pop_file = r'D:\paq\GEE\problem\data\wordpop\bj_19.tif'
with rasterio.open(pop_file) as src:
    pop_data = src.read()
    print(pop_data.shape)



# 合并矩阵
combined = np.concatenate([compressed_lst,
                           compressed_ndvi,compressed_ndwi,compressed_VIRS,
                           pop_data, compressed_GAIA], axis=0)

mask = ~np.isnan(combined[0]) & (~np.isnan(combined[1]) | ~np.isnan(combined[2]) & ~np.isnan(combined[3]) | ~np.isnan(combined[4]) | ~np.isnan(combined[5]))

# 获取有效位置的坐标
rows, cols = np.where(mask)

# 提取对应数值
values = combined[:, rows, cols].T  # 转置后每行对应一个坐标的两个值

# 创建DataFrame
df = pd.DataFrame({
    'row_index': rows,
    'col_index': cols,
    'LST': values[:, 0],
    'NDVI': values[:, 1],
    'NDWI':values[:,2],
    'VIRS':values[:,3],
    'POP':values[:,4],
    'GAIA':values[:,5],
})

# 保存为CSV（建议使用压缩格式）
df.to_csv(r'D:\paq\GEE\problem\bj19_2.csv')




