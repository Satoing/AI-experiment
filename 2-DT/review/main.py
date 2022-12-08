import pandas as pd
idx =  "hello the cruel world".split()
val = [1000, 201, 333, 104]
t = pd.Series(val, index = idx)
print (t.quantile())
print(t.median())
print (round(t.quantile(0.5), 2))
print (t.quantile(0.3))
print (t.quantile(0.75))