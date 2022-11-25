import torch
from torch import nn

def predict(model, text, party, index_to_word, vocab, text_pipeline, device, next_words=50):
    text = text.lower()
    words = text.split(' ')
    model.eval()
    pred_words = []
    with torch.no_grad():
      state_h, state_c = model.init_state_half(party)
    
      i = 0
      while True:
          l = len(words)   
          x = torch.tensor(vocab(text_pipeline(words))).to(device)
          
          y_pred, (state_h, state_c) = model(x, (state_h, state_c))
          
          last_word_logits = nn.functional.softmax(y_pred[-1], dim=0)
      
          probs, maxk = torch.topk(last_word_logits, 3, dim=0)
          k = 0
          while k == 0 or k == 2:
            k = torch.multinomial(last_word_logits, 1)
            if i > next_words and k == 2:
              break

          word = index_to_word(k)
          words.append(word)
          pred_words.append(word)
          if k == 2:
            break
          i += 1

    return ' '.join(words)