import numpy as  np
import time


data_size = 10 ** 7
py_list = list(range(data_size))
a = np.arange(data_size)
# a = np.array([10,20,30])
# print(a)
# print(type(a))
# 测试列表计算时间（循环+平方）
start_time = time.time()
squared_list = [x**2 for x in py_list]
list_time = time.time() - start_time

# 测试NumPy数组计算时间（向量化平方）
start_time = time.time()
squared_np = a**2
numpy_time = time.time() - start_time

# 输出结果
print(f"Python列表计算耗时: {list_time:.6f} 秒")
print(f"NumPy数组计算耗时: {numpy_time:.6f} 秒")
print(f"NumPy比列表快 {list_time/numpy_time:.1f} 倍")