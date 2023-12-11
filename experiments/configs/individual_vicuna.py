import os

os.sys.path.append("..")
from configs.template import get_config as default_config

def get_config():
    
    config = default_config()
    config.model_paths = [
        "/dlabdata1/drudi/models/vicuna-7b-v1.3",
    ]
    config.tokenizer_paths = [
        "/dlabdata1/drudi/models/vicuna-7b-v1.3",
    ]
    return config
