"""BaseX encoding"""

import pandas as pd
import numpy as np
import math
import re
from rdt.transformers import OneHotEncodingTransformer
from sklearn.preprocessing import OneHotEncoder


class MyOneHotEncoder():
    def __init__(self, cols, label=None):
        self.cols = cols
        self.label = label
        if label != None and label not in self.cols:
            self.cols.append(label)

    def fit(self, df):
        self.column_raw_dtypes = df.infer_objects().dtypes
        self.column_transform_info = {}
        for col in self.cols:
            raw_column_data = df[col].values
            ohe = OneHotEncodingTransformer()
            ohe.fit(raw_column_data)
            digit = len(ohe.dummies)
            self.column_transform_info[col] = (ohe, digit)

    def transform(self, df):
        cols = df.columns.values.tolist()
        self.columns = cols
        column_data_list = []
        for col in cols:
            raw_column_data = df[col].values
            ohe = self.column_transform_info.get(col)[0]
            encoded = ohe.transform(raw_column_data)
            column_data_list.append(encoded)
        return np.concatenate(column_data_list, axis=1).astype(float)

    def inverse_transform(self, x):
        st = 0
        recovered_column_data_list = []
        for col in self.columns:
            ohe, digit = self.column_transform_info.get(col)
            ed = st + digit
            raw_column_data = x[:, st:ed]
            recovered = ohe.reverse_transform(raw_column_data)
            recovered_column_data_list.append(recovered)
            st = ed
        recovered_data = np.column_stack(recovered_column_data_list)
        recovered_data = (pd.DataFrame(recovered_data, columns=self.columns)
                          .astype(self.column_raw_dtypes))
        return recovered_data

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)


if __name__ == '__main__':
    # compare customer
    # origin_path = '../datasets/tpch-0.1/customer.csv'
    # samples_path = '../output/keras_cvae_customer_c_nationkey_ld16_id64_bs16_ep20.csv'
    # sql = 'select sum(c_acctbal) from df group by c_nationkey'
    # encoder = MyOneHotEncoder(cols=['color', 'outcome'])
    encoder=OneHotEncoder(handle_unknown='ignore')
    df = pd.DataFrame({'color': ["a", "c", "a", "a", "b", "b"], 'outcome': ["1", "2", "0", "0", "3", "1"]})
    print(df)
    dfr = encoder.fit_transform(df)
    print(dfr)
    dfrr = encoder.inverse_transform(dfr)
    print(dfrr)
