import torch
from torch.amp import autocast
from sklearn.metrics import f1_score

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0

    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

            # 🔥 Store for F1
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = correct / len(loader.dataset)
    avg_loss = total_loss / len(loader.dataset)

    #  Macro F1 
    f1 = f1_score(all_labels, all_preds, average="macro")

    return avg_loss, accuracy, f1