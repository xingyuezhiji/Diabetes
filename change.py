#coding=utf-8
import pandas as pd
import numpy as np
# from fancyimpute import KNN
from dateutil.parser import parse

def change(name):
    df = pd.read_csv(name,encoding='gb2312')
    index = ['尿素','*r-谷氨酰基转换酶','尿酸','*碱性磷酸酶','红细胞计数','血小板计数','血小板比积',
         '嗜酸细胞%','红细胞平均体积','红细胞体积分布宽度','红细胞平均血红蛋白浓度']
    for i in index:
        # index.remove(i)
        for j in index:
            if i == j:
                continue
            else:
                df['{}/{}'.format(i,j)] = df[i]/df[j]
    return df

df1 = change('d_train_20180102.csv')
print(df1.shape)
df1.to_csv('train_change.csv', index=False)
df2 = change('d_test_A_20180102.csv')
df2.to_csv('test_change.csv', index=False)
df3 = change('d_test_B_20180128.csv')
df3.to_csv('test_change_B.csv', index=False)
df4 = change('d_train_20180102_add.csv')
print(df4.shape)
df4.to_csv('train_change_add.csv', index=False)

# data = pd.read_csv('d_test_B_20180128.csv',encoding='gb2312')
# data['性别'] = data['性别'].map({'男': 1, '女': 0})
# data['体检日期'] = (pd.to_datetime(data['体检日期']) - parse('2017-10-09')).dt.days
# df = pd.DataFrame({'col1':[1, 2, np.nan, 4, 5, 6, 7, 8, 9, np.nan, 11],
#                     'col2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110.]})
# df_filled = KNN(k=3).complete(data)
# # print(df_filled)
# data_filled = pd.DataFrame(df_filled)
# data_filled.to_csv('testdata_B.csv')

