import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def bar_compare(origin_df, generated_df, column, figdir, figsize=(8, 6), rotation=0):
    plt.figure(figsize=figsize)
    plt.title(column)
    origin = origin_df.groupby(by=[column]).size()
    generated = generated_df.groupby(by=[column]).size()
    values = origin.index.tolist()
    x = range(len(values))
    df = pd.DataFrame({'origin': origin, 'generated': generated}).fillna(0)
    bar1 = plt.bar([i - 0.15 for i in x], df['origin'].values.tolist(), width=0.3, label='origin')
    bar2 = plt.bar([i + 0.15 for i in x], df['generated'].values.tolist(), width=0.3, label='samples')

    plt.xticks(x, values, rotation=rotation)
    plt.tight_layout()

    for rect in bar1:
        height = rect.get_height()  # 获得bar1的高度
        plt.text(rect.get_x() + rect.get_width() / 2, height + 3, str(height), ha="center", va="bottom")
    for rect in bar2:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height + 3, str(height), ha="center", va="bottom")
    plt.legend()

    plt.savefig(fname="{}/{}.png".format(figdir, column))
    plt.show()


def hist_compare(origin_df, generated_df, column, figdir, figsize=(8, 6), bound=(10, 100), binn=12):
    plt.figure(figsize=figsize)
    plt.title(column)
    bins = np.linspace(10, 100, 12)

    ax1 = plt.subplot(121)
    plt.hist(origin_df[column].values.tolist(), bins)
    ax2 = plt.subplot(122)
    plt.hist(generated_df[column].values.tolist(), bins)
    plt.tight_layout()

    plt.savefig(fname="{}/{}.png".format(figdir, column))
    plt.show()


def plot(samples, param):
    origin = pd.read_csv('./datasets/adult.csv')
    figdir = ""
    origin=origin[origin["marital-status"]=='Never-married']
    samples=samples[samples["marital-status"]=='Never-married']
    if param["model_type"] == "keras_vae":
        figdir = "./figures/{}_{}_ld{}_id{}_bs{}_ep{}.h5".format(param["model_type"], param["name"],
                                                                 param["latent_dim"],
                                                                 param["intermediate_dim"], param["batch_size"],
                                                                 param["epochs"])
    elif param["model_type"] == "keras_cvae":
        figdir = "./figures/{}_{}_{}_ld{}_id{}_bs{}_ep{}.h5".format(param["model_type"], param["name"],
                                                                    '_'.join(param["label_column"]),
                                                                    param["latent_dim"],
                                                                    param["intermediate_dim"], param["batch_size"],
                                                                    param["epochs"])

    try:
        os.mkdir(figdir)
    except:
        pass

    bar_compare(origin, samples, 'sex', figdir)
    bar_compare(origin, samples, 'income-bucket', figdir)
    bar_compare(origin, samples, 'marital-status', figdir, figsize=(10, 6), rotation=30)
    bar_compare(origin, samples, 'occupation', figdir, figsize=(10, 6), rotation=30)
    bar_compare(origin, samples, 'relationship', figdir, figsize=(10, 6), rotation=30)
    hist_compare(origin, samples, 'age', figdir)
    # hist_compare(origin, samples, 'fnlwgt')
