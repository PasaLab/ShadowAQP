"""BaseX encoding"""

import pandas as pd
import numpy as np
import math
import re
from collections import namedtuple

from sklearn.mixture import BayesianGaussianMixture

GaussianModel = namedtuple(
    "GaussianModel", ["gm", "valid", "num_components"])


class GaussianEncoder():
    def __init__(self, cols, max_clusters=10, weight_threshold=0.001):
        self.cols = cols
        self.max_clusters = max_clusters
        self.weight_threshold = weight_threshold

    def fit(self, df):
        self.gms = {}
        self.total_digits=0
        for col in self.cols:
            column_data = df[col].values
            gm = BayesianGaussianMixture(
                n_components=self.max_clusters,
                weight_concentration_prior_type='dirichlet_process',
                weight_concentration_prior=0.001,
                n_init=1
            )
            gm.fit(column_data.reshape(-1, 1))
            valid_component_indicator = gm.weights_ > self.weight_threshold
            num_components = valid_component_indicator.sum()
            self.total_digits+=num_components+1
            self.gms[col] = GaussianModel(gm=gm, valid=valid_component_indicator, num_components=num_components)

    def transform(self, data):
        column_data_list = []
        column_names = []
        for col in self.cols:
            column_data = data[[col]].values
            gm = self.gms[col].gm
            valid_component_indicator = self.gms[col].valid
            num_components = valid_component_indicator.sum()
            means = gm.means_.reshape((1, self.max_clusters))
            stds = np.sqrt(gm.covariances_).reshape((1, self.max_clusters))
            normalized_values = ((column_data - means) / (4 * stds))[:, valid_component_indicator]
            component_probs = gm.predict_proba(column_data)[:, valid_component_indicator]
            selected_component = np.zeros(len(column_data), dtype='int')
            for i in range(len(column_data)):
                component_porb_t = component_probs[i] + 1e-6
                component_porb_t = component_porb_t / component_porb_t.sum()
                selected_component[i] = np.random.choice(
                    np.arange(num_components), p=component_porb_t)
            selected_normalized_value = normalized_values[
                np.arange(len(column_data)), selected_component].reshape([-1, 1])
            selected_normalized_value = np.clip(selected_normalized_value, -.99, .99)

            selected_component_onehot = np.zeros_like(component_probs)
            selected_component_onehot[np.arange(len(column_data)), selected_component] = 1
            column_data_list.append(selected_normalized_value)
            column_data_list.append(selected_component_onehot)
            column_names.append(col + "_norm")
            column_names += [col + "_" + str(i) for i in range(num_components)]
        encoded_data = np.concatenate(column_data_list, axis=1).astype(float)
        return pd.DataFrame(encoded_data, columns=column_names)

    def inverse_transform(self, data, sigmas=None):
        st = 0
        recovered_column_data_list = []
        column_names = []
        for col in self.cols:
            gm = self.gms[col].gm
            valid_component_indicator = self.gms[col].valid
            dim = self.gms[col].num_components + 1
            column_data = data[:, st:st + dim]
            selected_normalized_value = column_data[:, 0]
            selected_component_probs = column_data[:, 1:]
            if sigmas is not None:
                sig = sigmas[st]
                selected_normalized_value = np.random.normal(selected_normalized_value, sig)

            selected_normalized_value = np.clip(selected_normalized_value, -1, 1)#.astype(np.float32)
            component_probs = np.ones((len(column_data), self.max_clusters)) * -100
            component_probs[:, valid_component_indicator] = selected_component_probs

            means = gm.means_.reshape([-1])
            stds = np.sqrt(gm.covariances_).reshape([-1])
            selected_component = np.argmax(component_probs, axis=1)

            std_t = stds[selected_component]
            mean_t = means[selected_component]
            recovered_column_data = selected_normalized_value * 4 * std_t + mean_t

            recovered_column_data_list.append(recovered_column_data)
            column_names.append(col)
            st += dim

        recovered_data = np.column_stack(recovered_column_data_list)
        recovered_data = (pd.DataFrame(recovered_data, columns=column_names))

        return recovered_data

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)


if __name__ == '__main__':
    # df = pd.read_csv("../datasets/tpch-0.1/supplier.csv")
    # transformer = GaussianEncoder(cols=['s_acctbal','s_nationkey'],max_clusters=0)
    # transformer.fit(df)
    # tdf = transformer.transform(df)
    # print(transformer.gms)
    # vdf = transformer.inverse_transform(tdf.values)
    # print(vdf[:10])
    # vdf.to_csv("../datasets/vsupplier.csv",index=False)
    # print(df[['s_acctbal','s_nationkey']][:10])
    df = pd.read_csv("../datasets/adult.csv")
    transformer = GaussianEncoder(cols=['age'], max_clusters=1)
    transformer.fit(df)
    tdf = transformer.transform(df)
    print(transformer.gms)
    vdf = transformer.inverse_transform(tdf.values)
    print(vdf[:10])
    vdf.to_csv("../datasets/vsupplier.csv", index=False)
    print(df['age'][:10])
    print(vdf-df)