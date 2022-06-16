import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import linear_model

darea = np.array([600,3000,3200,3600,4000])
dprice = np.array([550000,565000,610000,680000,725000])

a = pd.DataFrame(darea,index='area')
b= pd.DataFrame(dprice,index='price')

reg = linear_model.LinearRegression()
reg.fit(a[['area']],b.price)


print(reg.predict(30000))