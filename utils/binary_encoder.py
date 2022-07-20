"""BaseX encoding"""

import pandas as pd
import numpy as np
import math
import re


class BinaryEncoder():
    def __init__(self, cols, label=None):
        self.cols = cols
        self.origin_cols = [col for col in cols]
        self.base = 2
        self.label = label
        if label != None and label not in self.cols:
            self.cols.append(label)

    def calc_required_digits(self, values):
        # figure out how many digits we need to represent the classes present
        if self.base == 1:
            digits = len(values) + 1
        else:
            digits = int(np.ceil(math.log(len(values), self.base))) + 1
        return digits

    def fit_base_n_encoding(self, df):
        self.column_category_sizes = {}
        self.column_categories_map = {}
        self.column_digits = {}
        self.numeric_columns = [col for col in df.columns if col not in self.cols]
        mappings_out = {}  # []

        for col_name in self.cols:
            column = df[col_name].astype("category")
            self.column_categories_map[col_name] = dict(enumerate(column.cat.categories))
            self.column_category_sizes[col_name] = len(column.cat.categories)
            if self.label != None and self.label == col_name:
                self.label_value_mapping = self.column_categories_map[col_name]

        for col, values in self.column_categories_map.items():
            digits = self.calc_required_digits(values)
            self.column_digits[col] = digits
            X_unique = pd.DataFrame(index=values,
                                    columns=[str(col) + '_%d' % x for x in range(digits)],
                                    data=np.array([self.col_transform(x, digits) for x in range(0, len(values))]))
            mappings_out[col] = X_unique
            # mappings_out.append({'col': col, 'mapping': X_unique})
        return mappings_out

    @staticmethod
    def number_to_base(n, b, limit):
        if b == 1:
            return [0 if n != _ else 1 for _ in range(limit)]

        if n == 0:
            return [0 for _ in range(limit)]

        digits = []
        for _ in range(limit):
            digits.append(int(n % b))
            n, _ = divmod(n, b)

        return digits[::-1]

    def col_transform(self, col, digits):
        """
        The lambda body to transform the column values
        """

        if col is None or float(col) < 0.0:
            return None
        else:
            col = self.number_to_base(int(col), self.base, digits)
            if len(col) == digits:
                return col
            else:
                return [0 for _ in range(digits - len(col))] + col

    def fit(self, df):
        self.mapping = self.fit_base_n_encoding(df)

    def transform(self, X_in):
        X = X_in.copy(deep=True)
        for col_name in self.cols:
            X[col_name] = X[col_name].astype("category")
            X[col_name] = X[col_name].cat.codes
            # X[col_name] = X[col_name].astype("category")
        cols = X.columns.values.tolist()
        self.feature_names = cols

        for col in self.mapping:
            mod = self.mapping.get(col)

            base_df = mod.reindex(X[col])
            base_df.set_index(X.index, inplace=True)
            X = pd.concat([base_df, X], axis=1)

            old_column_index = cols.index(col)
            cols[old_column_index: old_column_index + 1] = mod.columns

        # for switch in self.mapping:
        #     col = switch.get('col')
        #     mod = switch.get('mapping')
        #
        #     base_df = mod.reindex(X[col])
        #     base_df.set_index(X.index, inplace=True)
        #     X = pd.concat([base_df, X], axis=1)
        #
        #     old_column_index = cols.index(col)
        #     cols[old_column_index: old_column_index + 1] = mod.columns
        if self.label != None and self.label not in self.origin_cols:
            self.feature_names = [col for col in cols if not col.startswith(self.label)]
        print("feature_names:",self.feature_names)
        return X.reindex(columns=cols)

    def basen_to_integer(self, X):
        out_cols = X.columns.values.tolist()
        for col in self.origin_cols:
            col_list = [col0 for col0 in out_cols if re.match(str(col) + '_\\d+', str(col0))]
            insert_at = out_cols.index(col_list[0])
            if self.base == 1:
                value_array = np.array([int(col0.split('_')[-1]) for col0 in col_list])
            else:
                len0 = len(col_list)
                value_array = np.array([self.base ** (len0 - 1 - i) for i in range(len0)])
            X.insert(insert_at, col, np.dot(X[col_list].values, value_array.T))
            X.drop(col_list, axis=1, inplace=True)
            out_cols = X.columns.values.tolist()
        return X

    def inverse_transform(self, X_in):
        if not isinstance(X_in, pd.DataFrame):
            X_in = pd.DataFrame(X_in, columns=self.feature_names)
        categorical_columns = [col for col in self.feature_names if col not in self.numeric_columns]
        print("categorical_columns:",categorical_columns)
        X_in[categorical_columns] = X_in[categorical_columns].applymap(lambda x: 1 if x >= 0.5 else 0)
        X_in2=X_in[categorical_columns]
        # numeric_df = X_in[self.numeric_columns]
        # X_in = X_in.applymap(lambda x: 1 if x >= 0.5 else 0)
        # X_in[self.feature_names] = numeric_df
        X = self.basen_to_integer(X_in2)
        for col, column_mapping in self.column_categories_map.items():
            if col in self.origin_cols:
                # column_mapping = self.column_categories_map[switch]
                inverse = pd.Series(data=column_mapping.values(), index=column_mapping.keys())
                X[col] = X[col].map(inverse)  # .astype(switch.get('data_type'))
        X.fillna(method='ffill')
        return X

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)


if __name__ == '__main__':
    # compare customer
    # origin_path = '../datasets/tpch-0.1/customer.csv'
    # samples_path = '../output/keras_cvae_customer_c_nationkey_ld16_id64_bs16_ep20.csv'
    # sql = 'select sum(c_acctbal) from df group by c_nationkey'
    encoder = BinaryEncoder(cols=['color', 'outcome'])
    df = pd.DataFrame({'color': ["a", "c", "a", "a", "b", "b"], 'outcome': ["1", "2", "0", "0", "3", "1"]})
    print(df)
    dfr = encoder.fit_transform(df)
    print(dfr)
    dfrr = encoder.inverse_transform(dfr)
    print(dfrr)
