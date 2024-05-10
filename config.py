from pathlib import Path

def get_config():
    return {
        "Batch_size":8,
        "num_epochs":75,
        "lr":10**-4,
        "seq_len":350,
        "d_model":512,
        "lang_src":"en",
        "lang_tgt":"it",
        "model_folder": "weights",
        "model_basename":"t_model_",
        "preload":'latest',
        "tokenizer_file":"tokenizer_{0}.json",
        "experiment_name":"runs/t_model",
        'data_set_name_from_HF':'Helsinki-NLP/opus_books', # Helsinki-NLP/opus-mt-en-ar
        'data_subset_ratio':0.7, # limiting the data set size -> range[0,1] , to use whole dataset None
        'num_workers':2,
        'pin_memory':True,
    }



def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['data_set_name_from_HF']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{config['data_set_name_from_HF']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
