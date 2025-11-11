import torch
from tqdm import tqdm


def train_fn(model, data_loader, optimizer, device):
    fin_loss = 0
    model.train()
    tk = tqdm(data_loader, total=len(data_loader))

    for data in tk:
        for k, v in data.items():
            data[k] = v.to(device)

        optimizer.zero_grad()
        _, loss = model(**data)
        loss.backward()
        optimizer.step()
        fin_loss += loss.item()

    return fin_loss / len(data_loader)


def evaluate(model, data_loader, device):
    model.eval()
    fin_loss = 0
    fin_preds = []

    with torch.no_grad():
        tk = tqdm(data_loader, total=len(data_loader))

        for data in tk:
            for k, v in data.items():
                data[k] = v.to(device)

            batch_preds, loss = model(**data)
            fin_loss += loss.item()
            batch_preds = batch_preds.cpu()
            fin_preds.append(batch_preds)

        return fin_preds, fin_loss / len(data_loader)


def predict(model, data_loader, device):
    model.eval()
    final_preds = []

    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader))

        for data in tk0:
            for key, value in data.items():
                data[key] = value.to(device)

            preds, _ = model(**data)
            preds = preds.cpu()
            final_preds.append(preds)

    return final_preds
