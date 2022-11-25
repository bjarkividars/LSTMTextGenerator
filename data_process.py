from sklearn.model_selection import train_test_split
from tokenizer.tokenizer import split_into_sentences #Tokenizer for icelandic
from torchtext.vocab import build_vocab_from_iterator
import re
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


def split(df):
    df.sample(frac=1)
    df = df.groupby('party').head(4800)
    df['speech'] = df['speech'].str.replace('<p>', '')

    return train_test_split(df, test_size=0.10, stratify=df.party)


def create_vocab(df):
    def yield_tokens(speech):
        for text in speech:
            for s in (split_into_sentences(re.sub("[\(\[].*?[\)\]]", "", text))):
                yield(s.split())

    vocab = build_vocab_from_iterator(yield_tokens(df.speech), specials = ['<unk>', '<bos>', '<eos>', '<pad>'], min_freq=10)
    vocab.set_default_index(vocab['<unk>'])
    
    return vocab


def text_pipeline(speech):
  r = []
  for s in (split_into_sentences(speech)):
      r.extend(s.split())
  return r


label_pipeline = lambda x: 0 if x == '(M)' else 1

def data_process(df, vocab):
  processed_data = []

  for index, row in df.iterrows():
    text = text = re.sub("[\(\[].*?[\)\]]", "", row[1])
    text_tensor = torch.tensor(vocab(text_pipeline(text)))
    party_tensor = torch.tensor(label_pipeline(row[0]))
    processed_data.append([text_tensor, party_tensor])

  return processed_data


def index_to_word(word, vocab):
    idx_to_word = vocab.get_itos()
    try:
        return idx_to_word[word]
    except KeyError:
        return '<unk>'


def generate_batch(data_batch, vocab):
    PAD_IDX = vocab['<pad>']
    BOS_IDX = vocab['<bos>']
    EOS_IDX = vocab['<eos>']
    text_list, label_list, party_list = [], [], []
    for text, party in data_batch:
        text_list.append(torch.cat((torch.tensor([BOS_IDX]), text), dim=0))
        label_list.append(torch.cat((text, torch.tensor([EOS_IDX])), dim=0))
        party_list.append(torch.tensor(party))
    text_list = pad_sequence(text_list, padding_value=PAD_IDX)
    label_list = pad_sequence(label_list, padding_value=PAD_IDX)
    return text_list, label_list, party_list