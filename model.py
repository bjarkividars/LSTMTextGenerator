import torch
import numpy as np
from torch import nn

class GenerationLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, PAD_IDX):
        super(GenerationLSTM, self).__init__()
        self.lstm_size = hidden_dim
        self.embedding_dim = embed_dim
        self.num_layers = n_layers

        n_vocab = vocab_size
        self.embedding = nn.Embedding(
            num_embeddings=n_vocab,
            embedding_dim=self.embedding_dim,
            padding_idx=PAD_IDX
        )
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.lstm_size,
            num_layers=self.num_layers,
            dropout=0.2,
        )
        self.fc = nn.Linear(self.lstm_size, n_vocab)

    def forward(self, x, prev_state):
        embed = self.embedding(x)

    
        output, state = self.lstm(embed, prev_state)
        logits = self.fc(output)

        return logits, state

    def init_state(self, batch_size, device): #used for pre-training
        h = np.zeros((self.num_layers, batch_size, self.lstm_size))
        c = np.zeros((self.num_layers, batch_size, self.lstm_size))
        h.fill(0.5)
        c.fill(0.5)
        return (torch.tensor(h, dtype=torch.float32).to(device),
                torch.tensor(c, dtype=torch.float32).to(device))

    def init_state_cat(self, cat, device): #used for categories
        h = np.zeros((self.num_layers, 1, self.lstm_size))
        c = np.zeros((self.num_layers, 1, self.lstm_size))
        if cat == 'M':
          h.fill(0)
          c.fill(0)
        else:
          h.fill(1)
          c.fill(1)
        return (torch.tensor(h, dtype=torch.float32).to(device),
                torch.tensor(c, dtype=torch.float32).to(device))
        
