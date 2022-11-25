def train(dataloader, model, criterion, optimizer, device):
    model.train()
    losses = []
    for idx, (text, label) in enumerate(dataloader):
        label = label.to(device)
        text = text.to(device)

        state_h, state_c = model.init_state(text.shape[1], device)
        optimizer.zero_grad()
      
        y_pred, (state_h, state_c) = model(text, (state_h, state_c))
  
        loss = criterion(y_pred.transpose(1, 2), label)
        state_h = state_h.detach()
        state_c = state_c.detach()

        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return sum(losses)/len(losses)