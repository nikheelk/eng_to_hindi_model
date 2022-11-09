import torch

from utils import *
from models import *

import sys
import argparse
import random

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
    
model_name = "HntoEng_Attention"
data_name = "HntoEng"
model_type = "attention"

parser = argparse.ArgumentParser(description='Eng to Hindi Machine Translation PyTorch NLP Training')
parser.add_argument('--b', '-b', default=128, type=int, help='batch size')  
parser.add_argument('--e', '-e', default=5, type=int, help='no of epochs') 
parser.add_argument('--n', '-n', default=10, type=int, help='no of Examples') 
parser.add_argument('--name', default=model_name, type=str, help='Name of pth model file to be stored')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')   
parser.add_argument('--train', '-t', action='store_true',
                    help='training starts if True else only inference')         
parser.add_argument('--inference', '-i', action='store_true',
                    help='inference data if True else only training') 
args = parser.parse_args()


def main(model_name = args.name, batch_size = args.b, epochs = args.e, resume = args.resume, training = args.train, inference = args.inference, n = args.n):

    global data_name, model_type

    SEED = 1234

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data, valid_data, train_iterator, valid_iterator, SRC, TRG = dataloaders(device, data_name, batch_size = batch_size, seed = SEED)
    
    if not training and not inference:
        training = True
        inference = True

    if training:
        model = model_train(model_name, model_type, device, SRC, TRG, resume, train_iterator, valid_iterator, training=training, n_epochs = epochs )
    else:
        model = model_train(model_name, model_type, device, SRC, TRG, resume, train_iterator, valid_iterator, training=training, n_epochs = epochs )
    
    if inference:
        example_index = random.sample(range(0, len(valid_data.examples)), n)
        print("\n****************************************************************************\n")

        for i in range(n):
            
            print(f"***** Example {i} -- Index No {example_index[i]} *****\n")
            src = vars(valid_data.examples[example_index[i]])['src']
            trg = vars(valid_data.examples[example_index[i]])['trg']

            print(f'src = {src}')
            print(f'trg = {trg}')

            translation = translate_sentence(src, trg, SRC, TRG, model, device)


            print(f'predicted trg = {translation}')
            
        print("\n****************************************************************************\n")
if __name__ == "__main__":
    main()