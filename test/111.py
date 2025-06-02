import rasterio
import numpy as np
import pandas as pd
from skimage.transform import resize

lst_file = r'D:\paq\GEE\problem\lst\bj19_LST.tif'
with rasterio.open(lst_file) as src:
    lst_data = src.read()

compressed = resize(lst_data, (1, 1943, 2511), mode='reflect', anti_aliasing=True)

pop_file = r'D:\paq\GEE\problem\人口\bj_19.tif'
with rasterio.open(pop_file) as src:
    pop_data = src.read()

# 合并矩阵
combined = np.concatenate([compressed, pop_data], axis=0)

mask = ~np.isnan(combined[0]) & ~np.isnan(combined[1])

# 获取有效位置的坐标
rows, cols = np.where(mask)

# 提取对应数值
values = combined[:, rows, cols].T  # 转置后每行对应一个坐标的两个值

# 创建DataFrame
df = pd.DataFrame({
    'row_index': rows,
    'col_index': cols,
    'wordpop': values[:, 1],
    'LST': values[:, 0],
})

# 删除人口密度数据中大于1或小于 -1 的值
df = df[(df['wordpop'] >= -1) & (df['wordpop'] <= 1)]

# 保存为CSV（建议使用压缩格式）
df.to_csv(r'D:\paq\GEE\problem\111.csv')
