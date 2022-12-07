import numpy as np
import pandas as pd
from math import log10

nums = [0]
for num in nums:
    print("num =",num)
    a = input("输入数字(空格间隔):")
    if a != '0': nums.extend(a.split(' '))
    print(nums)
