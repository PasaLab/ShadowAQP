import torch
import os.path
import time
import logging
logger = logging.getLogger(__name__)

def get_cvae_model_name(param):
    model_name = "model.h5"
    model_name = "{}_{}_{}_ld{}_id{}_bs{}_ep{}_{}_{}_{}".format(param["model_type"], param["name"],
                                                                 '_'.join(param["label_columns"]),
                                                                 param["latent_dim"],
                                                                 param["intermediate_dim"], param["batch_size"],
                                                                 param["epochs"], param['gpu_num'],param["categorical_encoding"],
                                                                 (param["numeric_encoding"] + str(
                                                                     param["max_clusters"])) if
                                                                 param["numeric_encoding"] == 'gaussian' else
                                                                 param["numeric_encoding"])
    return model_name

def get_model_name(param):
    model_name = "model.h5"
    label_columns=param["label_columns"].copy()
    bucket_columns=param["bucket_columns"].copy()
    if len(bucket_columns)>0:
        for col in bucket_columns:
            col_idx=label_columns.index(col)
            label_columns[col_idx]=label_columns[col_idx]+"_bucket"
    if param["model_type"] == "keras_vae" or param["model_type"] == "torch_vae":
        model_name = "{}_{}_ld{}_id{}_bs{}_ep{}_{}_{}".format(param["model_type"], param["name"], param["latent_dim"],
                                                              param["intermediate_dim"], param["batch_size"],
                                                              param["epochs"], param["categorical_encoding"],
                                                              param["numeric_encoding"] + str(param["max_clusters"]) if
                                                              param["numeric_encoding"] == 'gaussian' else
                                                              param["numeric_encoding"])
    elif param["model_type"] == "keras_cvae" or param["model_type"] == "torch_cvae":
        model_name = "{}_{}_{}_ld{}_id{}_bs{}_ep{}_{}_{}_{}".format(param["model_type"], param["name"],
                                                                 '_'.join(label_columns),
                                                                 param["latent_dim"],
                                                                 param["intermediate_dim"], param["batch_size"],
                                                                 param["epochs"], param['gpu_num'],param["categorical_encoding"],
                                                                 (param["numeric_encoding"] + str(
                                                                     param["max_clusters"])) if
                                                                 param["numeric_encoding"] == 'gaussian' else
                                                                 param["numeric_encoding"])
    return model_name


def save_keras_model(model, param):
    model_name = get_model_name(param)
    model.save_weights("./saved_models/{}.h5".format(model_name))


def load_keras_model(model, param):
    model_name = get_model_name(param)
    path = "./saved_models/{}.h5".format(model_name)
    if os.path.isfile(path):
        model.load_weights(path)
        return model
    return None


def save_torch_model(model, param,postfix=''):
    model_name = get_model_name(param)
    model_name+=postfix
    # torch.save(model, "./saved_models/{}.pkl".format(model_name))
    torch.save(model.state_dict(), "./saved_models/{}.pth".format(model_name))
    logger.info("save model successfully")


def load_torch_model(param, model,postfix=''):
    start_time = time.perf_counter()
    model_name = get_model_name(param)
    model_name += postfix
    # path = "./saved_models/{}.pkl".format(model_name)
    logger.info("load model name:{}".format(model_name))
    path = "./saved_models/{}.pth".format(model_name)
    if os.path.isfile(path):
        gpu_num = param['gpu_num']
        device_name="cuda:{}".format(gpu_num) if torch.cuda.is_available() else "cpu"
        device = torch.device(device_name)
        # model.load_state_dict(torch.load(path))
        # model = torch.load(path,map_location={"cuda:2":"cuda:1","cuda:3":"cuda:1"})
        model.load_state_dict(torch.load(path,map_location=device_name))
        model = model.to(device)
        end_time = time.perf_counter()
        logger.info("load torch model time elapsed:{}".format(end_time - start_time))
        return model

    return None
