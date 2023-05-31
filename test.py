import cv2
import numpy as np

# 生成13x13的随机数组
arr = np.random.rand(13, 13)
p = 3
p_class = 3
# 找到前3个最大值
max_values = np.sort(arr.flatten())[-p:]

# 将数组形状变为256x256
new_arr = cv2.resize(arr, (256, 256), interpolation=cv2.INTER_NEAREST)

# 找到最大值在数组中的位置
max_indices = np.where(new_arr == max_values[0])
#print("最大值在数组中的位置：", max_indices)
size_upsample = (256, 256)

mask = np.zeros(size_upsample)
# 如果有多个最大值，计算它们的平均位置
if len(max_indices[0]) > 1:
    center = np.mean(max_indices, axis=1).astype(int)
    print("最大值的中心位置：", tuple(center))
else:
    center = max_indices[0][0], max_indices[1][0]
    print("最大值的位置即为其中心位置：", max_indices[0][0], max_indices[1][0])
mask[center[0],center[1]] = 1

# 在数组中将最大值的位置处和周围的7x7个位置的值设为1
mask[max(center[0] - p_class, 0):min(center[0] + 1 + p_class, 256),
max(center[1] - p_class, 0):min(center[1] + 1 + p_class, 256)] = 1

print(mask)
# 打印结果
print("前三个最大值：", max_values)
print("前三个最大值在13x13数组中的位置：", max_indices)
#print("前三个最大值在256x256数组中的位置：", new_indices)