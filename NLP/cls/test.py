import os
import yaml
import datetime

from model import ConvClassifier
from dataset import DataModule
from train import Trainer
import logger


def parse_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config, save_path):
    with open(save_path, 'w') as f:
        yaml.dump(config, f)


def main():
    ckpt_path = "logs/w_bert/b256_t50_h128_lstm1/checkpoints/epoch_10.pth"
    config = parse_config(os.path.join(os.path.dirname(ckpt_path), '../config.yaml'))
    train_config, data_config, model_config = config['train'], config['data'], config['model']
    log_dir = os.path.join(train_config['log_dir'], "test_" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    train_config['log_dir'] = log_dir
    os.makedirs(log_dir, exist_ok=True)
    logger.default_logger = logger.get_logger(os.path.join(log_dir, 'test.log'))
    save_config(config, os.path.join(log_dir, 'config.yaml'))

    data_module = DataModule(data_config)
    model = ConvClassifier(**model_config)
    trainer = Trainer(model, data_module, train_config)
    trainer.load_model(ckpt_path)
    trainer.test()


if __name__ == '__main__':
    main()