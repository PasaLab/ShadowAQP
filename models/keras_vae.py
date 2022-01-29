import math
import time
import logging
import numpy as np
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import losses
from keras.optimizers import Adam, RMSprop
from utils.model_utils import save_keras_model, load_keras_model
from keras.layers.merge import concatenate as concat
import tensorflow as tf
from keras.callbacks import EarlyStopping
import pandas as pd
from utils.dataset_utils import TabularDataset
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=K.shape(z_mean))
    return z_mean + K.exp(z_log_var / 2) * epsilon


def train_vae( param):
    # hyper parameters
    lr = param["lr"]
    optimizer_type = param["optimizer_type"]
    epochs = param["epochs"]
    batch_size = param["batch_size"]
    intermediate_dim = param["intermediate_dim"]
    latent_dim = param["latent_dim"]
    train_flag = param["train_flag"]

    loss_agg_type = param["loss_agg_type"]
    sample_rate = param["sample_rate"]

    # split train and test
    dataset = TabularDataset(param)
    data = dataset.data
    feature_info = dataset.feature_info
    n_row, n_col = data.shape
    origin_dim = n_col

    ## split the data into training data and test data
    # split_index = int(n_row * 0.7)
    # x_train = data.iloc[0:split_index, :].values
    # x_test = data.iloc[split_index:n_row, :].values

    ## use all the data as training data and 30% samples of the data as test data
    x_train = data.values
    x_test = data.values
    # x_test = data.sample(frac=0.3).values

    # build encoder layer
    x = Input(shape=(origin_dim,))
    encoder_h = Dense(intermediate_dim, activation='relu')(x)
    z_mean = Dense(latent_dim)(encoder_h)
    z_log_var = Dense(latent_dim)(encoder_h)
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    # build decoder layer
    decoder_h = Dense(intermediate_dim, activation='relu')
    h_decoded = decoder_h(z)

    if activate_type == "sigmoid":
        # use sigmoid function for all columns
        decoder_mean = Dense(origin_dim, activation='sigmoid')
        x_decoded_mean = decoder_mean(h_decoded)
    elif activate_type == "softmax":
        # use sigmoid function for all columns
        decoder_mean = Dense(origin_dim, activation='softmax')
        x_decoded_mean = decoder_mean(h_decoded)
    else:
        # use different activ function for different columns
        out_layer_list = [Dense(col_domain_size, activation='softmax') if col_type == "categorical"
                          else Dense(col_domain_size, activation='relu')
                          for col_name, col_type, col_domain_size in feature_info]
        out_list = []
        for out_layer in out_layer_list:
            out_list.append(out_layer(h_decoded))
        x_decoded_mean = concat(out_list)

    # build vae models
    vae = Model(x, x_decoded_mean)

    # define loss
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    if loss_agg_type == "sum":
        loss_agg_fun = K.sum
    else:
        loss_agg_fun = K.mean

    if loss_type == "bce":
        reconstruct_loss = K.sum(K.binary_crossentropy(x, x_decoded_mean), axis=-1)
        vae_loss = loss_agg_fun(reconstruct_loss + kl_loss)
    # elif loss_type == "dif":
    #     output_info = dataset.encoded_output_info
    #     softmax_loss = tf.zeros((batch_size,))
    #     tanh_loss = tf.zeros((batch_size,))
    #     st = 0
    #     for info in output_info:
    #         if info[1] == 'softmax':
    #             ed = st + info[0]
    #             softmax_loss += K.categorical_crossentropy(x_decoded_mean[:, st:ed], x[:, st:ed])
    #             st = ed
    #         else:
    #             ed = st + info[0]
    #             tanh_loss+=
    #             # tanh_loss += (x[:, st] - K.tanh(x_decoded_mean[:, st])) ** 2 / 2 / (0.1 ** 2)
    #             st = ed
    #     vae_loss = loss_agg_fun(softmax_loss + tanh_loss + kl_loss)
    else:
        numeric_column_len = len(dataset.numeric_columns)
        reconstruct_loss_num = losses.mean_squared_error(x[:, :numeric_column_len],
                                                         x_decoded_mean[:, :numeric_column_len])
        # reconstruct_loss_cat = K.categorical_crossentropy(x[:, numeric_column_len:], x_decoded_mean[:, numeric_column_len:])
        reconstruct_loss_cat = tf.zeros_like(reconstruct_loss_num)
        categorical_index = numeric_column_len
        for (col_name, col_type, col_domain_size) in feature_info:
            reconstruct_loss_cat += K.categorical_crossentropy(
                x[:, categorical_index:categorical_index + col_domain_size],
                x_decoded_mean[:, categorical_index:categorical_index + col_domain_size])
            categorical_index += col_domain_size
        vae_loss = loss_agg_fun(reconstruct_loss_num + reconstruct_loss_cat + kl_loss)

    vae.add_loss(vae_loss)
    optimizer = Adam(lr=lr)
    if optimizer_type == "rmsp":
        optimizer = RMSprop(lr=lr)
    logger.info("optimizer:{}".format(optimizer_type))
    vae.compile(optimizer=optimizer)
    vae.summary()

    # train models
    if train_flag == "train":
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=2)
        vae.fit(x_train,
                shuffle=False,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_test, None),
                callbacks=[early_stopping])
        save_keras_model(vae, param)
    else:
        load_keras_model(vae, param)

    model_weight = vae.get_weights()

    # encoder = Model(x, z_mean)
    decoder_input = Input(shape=(latent_dim,))
    _h_decoded = decoder_h(decoder_input)
    if activate_type == "sigmoid" or activate_type=="softmax":
        _x_decoded_mean = decoder_mean(_h_decoded)
    else:
        _out_list = []
        for out_layer in out_layer_list:
            _out_list.append(out_layer(_h_decoded))
        _x_decoded_mean = concat(_out_list)
    generator = Model(decoder_input, _x_decoded_mean)

    # generate samples
    z_decoded = generate_samples(generator, n_row, sample_rate, model_weight, latent_dim, batch_size)
    samples = dataset.decode_samples(z_decoded)
    save_samples(samples, param)
    return samples


def generate_samples(generator, total_num, sample_rate, model_weights, latent_dim, batch_size):
    start_time = time.perf_counter()
    z_mean_w = model_weights[3]
    z_std_w = [math.exp(logvar * 0.5) for logvar in model_weights[5]]
    z_samples = []
    total_samples = round(total_num * sample_rate)

    # for i in range(latent_dim):
    #     z_sample = np.random.normal(z_mean_w[i], z_std_w[i], total_samples)
    #     z_samples.append(z_sample)
    # z_samples = np.array(z_samples).T
    # z_decoded = generator.predict(z_samples, batch_size=batch_size)
    setps = total_samples // batch_size + 1
    decoded = []
    for _ in range(setps):
        z_samples = np.random.normal(0, 1, (batch_size, latent_dim))
        z_decoded = generator.predict(z_samples)
        decoded.append(z_decoded)
    result = np.concatenate(decoded)
    end_time = time.perf_counter()
    logger.info('[INFO]decode(decoder) samples time:{}'.format(end_time - start_time))
    return result


def save_samples(samples, param):
    samples_name = "{}_{}_ld{}_id{}_bs{}_ep{}.csv".format(param["model_type"], param["name"], param["latent_dim"],
                                                          param["intermediate_dim"], param["batch_size"],
                                                          param["epochs"])
    samples.to_csv("./output/{}".format(samples_name), index=False)

# def decode_samples(z_decoded, feature_info, categorical_columns, category_column_map, numeric_columns, scaler):
#     start_time = time.perf_counter()
#     column_index = 0
#     column_list = []
#     for (col_name, col_type, col_domain_size) in feature_info:
#         column_data = z_decoded[:, column_index:column_index + col_domain_size]
#         column_index += col_domain_size
#         if col_type == "categorical":
#             column_data = np.argmax(column_data, axis=1).reshape(-1, 1)
#         else:
#             column_data = column_data
#         column_list.append(column_data)
#     all_column = [info[0] for info in feature_info]
#     samples = np.concatenate(column_list, axis=1)
#     sample_df = pd.DataFrame(samples, columns=all_column)
#     for col_name in categorical_columns:
#         sample_df[col_name] = sample_df[col_name].map(category_column_map[col_name])
#
#     sample_df[numeric_columns] = scaler.inverse_transform(sample_df[numeric_columns])
#     end_time = time.perf_counter()
#     logger.info('[INFO]decode samples time:{}'.format(end_time - start_time))
#     return sample_df
