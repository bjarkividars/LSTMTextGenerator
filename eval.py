import torch

def train(dataloader, model, criterion, device):
    losses = []
    with torch.no_grad():
      for idx, (text, label) in enumerate(dataloader):
        label = label.to(device)
        text = text.to(device)
        state_h, state_c = model.init_state(text.shape[1])
    
      
        y_pred, (state_h, state_c) = model(text, (state_h, state_c))

        loss = criterion(y_pred.transpose(1, 2), label)
        losses.append(loss.item())

    return sum(losses)/len(losses)