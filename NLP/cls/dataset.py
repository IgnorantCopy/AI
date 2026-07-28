import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from gensim.models import KeyedVectors
from transformers import BertTokenizer, BertModel

from logger import default_logger


class GloVeEmbedder:
    def __init__(self, max_length):
        default_logger.info("Loading GloVe embeddings...")
        self.glove = KeyedVectors.load_word2vec_format(
            './data/wiki_giga_2024_100_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05.050_combined.txt',
            binary=False, no_header=True)
        default_logger.info("GloVe embeddings loaded.")

        self.max_length = max_length

    def __call__(self, sentence):
        embeddings = torch.from_numpy(np.stack([
            self.glove[word.lower()] if word.lower() in self.glove else np.zeros(self.glove.vector_size)
            for word in sentence
        ])).to(torch.float)
        seq_len, embed_dim = embeddings.shape
        if seq_len < self.max_length:
            padding = torch.zeros((self.max_length - seq_len, embed_dim))
            embeddings = torch.cat([embeddings, padding], dim=0)
        else:
            embeddings = embeddings[:self.max_length]
        return embeddings


class BertEmbedder:
    def __init__(self, max_length):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert.eval()
        self.bert.to('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_length = max_length

    @staticmethod
    def get_complete_sentence(sentence):
        result = ''
        for i, word in enumerate(sentence):
            if i > 0 and word[0].isalpha():
                result += " " + word
            else:
                result += word
        return result

    def __call__(self, sentence):
        _input = self.tokenizer(
            self.get_complete_sentence(sentence),
            padding='max_length', truncation=True,
            max_length=self.max_length, return_tensors='pt'
        )
        with torch.no_grad():
            output = self.bert(**{k: v.to(self.bert.device) for k, v in _input.items()})
        return output.last_hidden_state.squeeze(0).cpu()


class CsvDataset(Dataset):
    def __init__(self, filename, max_length, embedder):
        super().__init__()
        self.df = pd.read_csv(filename)
        self.max_length = max_length
        self.embedder = embedder

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sentence = eval(self.df.iloc[idx]['sentences'])
        label = self.df.iloc[idx]['label']
        return self.embedder(sentence), torch.tensor(label, dtype=torch.long)


class DataModule:
    def __init__(self, data_config):
        self.data_root = data_config['root']
        self.max_length = data_config['max_length']
        self.batch_size = data_config['batch_size']
        self.shuffle = data_config.get('shuffle', True)
        self.pin_memory = data_config.get('pin_memory', torch.cuda.is_available())
        self.use_bert = data_config['use_bert']
        self.embedder = BertEmbedder(self.max_length) if self.use_bert else GloVeEmbedder(self.max_length)

        train_dataset = CsvDataset(os.path.join(self.data_root, 'train.csv'), self.max_length, self.embedder)
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle, pin_memory=self.pin_memory)

        val_dataset = CsvDataset(os.path.join(self.data_root, 'dev.csv'), self.max_length, self.embedder)
        self.val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=self.pin_memory)

        test_dataset = CsvDataset(os.path.join(self.data_root, 'test.csv'), self.max_length, self.embedder)
        self.test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=self.pin_memory)