import torch

from torchtext.data import Field, BucketIterator
from torchtext import data

import spacy
import numpy as np
import pandas as pd
import spacy.cli 
import stanfordnlp
import indicnlp
from indicnlp.tokenize import indic_tokenize 

import random
import os
import glob
import re

def normalizeString(s):
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)
    #table = str.maketrans('', '', string.punctuation)
    #s = [word.translate(table) for word in s]
    s = re.sub(r"[.!?]+", r" ", s)
    return s


def HNTOENG(SRC, TRG, root = "data/HntoEng", data_type = "train", train_data = "train.txt", val_data = "val.txt"):
    fields = [('src', SRC),('trg', TRG)]
    
    def get_dataset(df):
        # example = [data.Example.fromlist(df[i]) for i in range(len(df))]
        #example = [data.Example.fromlist(df[i]) for i in range(len(df))]
        #example = open(df,encoding='utf-8').read().strip().split('\n')
        #print(df)
    # Split every line into pairs and normalize
        #pairs = [[s for s in l.split('\t')[:2]] for l in df]
        #print(pairs)
        example = [data.Example.fromlist([df.Hindi[i],df.English[i]], fields) for i in range(df.shape[0])]
        #print(example[1])
        dataset = data.Dataset(example, fields)
        #print(dataset[1])
        return dataset
    if data_type == "train":
        train_df = pd.read_csv(os.path.join(root, train_data), sep="\t")
        return get_dataset(train_df)
    elif data_type == "val":
        val_df = pd.read_csv(os.path.join(root, val_data), sep="\t")
        return get_dataset(val_df)
#en_ewt hi_hdtb
def dataloaders(device, data_name, batch_size = None, seed=42):
    #stanford_hi = stanfordnlp.download('hi')
    stanford_en = spacy.load('en_core_web_sm')
    def tokenize_hindi(text):
        """
        Tokenizes German text from a string into a list of strings (tokens) and reverses it
        """
        return [tok for tok in indic_tokenize.trivial_tokenize(text, lang='hi')]

    def tokenize_en(text):
        """
        Tokenizes English text from a string into a list of strings (tokens)
        """
        return [tok.text for tok in stanford_en.tokenizer(text)]
    
    SRC = Field(tokenize = tokenize_hindi, 
                init_token = '<sos>', 
                eos_token = '<eos>', 
                lower = True)

    TRG = Field(tokenize = tokenize_en, 
                init_token = '<sos>', 
                eos_token = '<eos>', 
                lower = True)
    
    if data_name == "HntoEng":
        train_data = HNTOENG(SRC, TRG, data_type = "train")
        # test_data = Multi30K(SRC, TRG, data_type = "test")
        valid_data = HNTOENG(SRC, TRG, data_type = "val")
    
    print("\n****************************************************************************\n")
    print("*****Data Details*****\n")
    print(f"Number of training examples: {len(train_data.examples)}")
    print(f"Number of validation examples: {len(valid_data.examples)}")
    # print(f"Number of testing examples: {len(test_data.examples)}")
    print("\n****************************************************************************\n")
    SRC.build_vocab(train_data, min_freq = 2)
    TRG.build_vocab(train_data, min_freq = 2)

    cuda = torch.cuda.is_available()

    torch.manual_seed(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True

    if cuda:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    
    batch_size = batch_size or (128 if cuda else 64)

    train_iterator, valid_iterator = BucketIterator.splits(
    (train_data, valid_data),
    sort_key = lambda x: len(x.src),
    batch_size = batch_size, 
    device = device)

    return train_data, valid_data, train_iterator, valid_iterator, SRC, TRG