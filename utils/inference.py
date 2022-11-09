import torch
import spacy
from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator
from indicnlp.tokenize import indic_tokenize
from indicnlp.tokenize import sentence_tokenize

def translate_sentence(sentence1, sentence2, src_field, trg_field, model, device, max_len = 50):
    
    model.eval()
        
    if isinstance(sentence1, str):
       nlp = spacy.load('en_core_web_sm')
       tokens = [token.text.lower() for token in nlp(sentence1)]
        #sentences=sentence_tokenize.sentence_split(sentence1, lang='hi')
        #UnicodeIndicTransliterator.transliterate(input_text,"hi","ta")
    else:
        tokens = [token.lower() for token in sentence1] 
    
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
     
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)

    if isinstance(sentence2, str):
       nlp = spacy.load('en_core_web_sm')
       tokens = [token.text.lower() for token in nlp(sentence2)]
    else:
        tokens = [token.lower() for token in sentence2]

    tokens = [trg_field.init_token] + tokens + [trg_field.eos_token]
        
    trg_indexes = [trg_field.vocab.stoi[token] for token in tokens]

    trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(1).to(device)

    with torch.no_grad():
        output = model(src_tensor, trg_tensor, 0)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        
        pred_token = output.argmax(1)
        
    trg_tokens = [trg_field.vocab.itos[i] for i in pred_token]
    
    return trg_tokens
