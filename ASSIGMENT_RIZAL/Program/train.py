import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from Utils.getData import getImageLabel
from torch.optim import Adam
from Model.CNN import SimpleCNN

def main():
    BATCH_SIZE = 32
    EPOCH = 25
    LEARNING_RATE = 0.001
    folds = [1, 2, 3, 4, 5]
    DEVICE = 'cuda'

    train_aug_loader = DataLoader(getImageLabel(augmented=f'C:/Bismillah/Data/Augmented Images/Augmented Images/FOLDS_AUG/', folds=folds, subdir=['Train']), batch_size=BATCH_SIZE, shuffle=True)
    train_ori_loader = DataLoader(getImageLabel(original=f'C:/Bismillah/Data/Original Images/Original Images/FOLDS/', folds=folds, subdir=['Train']), batch_size=BATCH_SIZE, shuffle=True)
    vali_loader = DataLoader(getImageLabel(original=f'C:/Bismillah/Data/Original Images/Original Images/FOLDS/', folds=folds, subdir=['Valid']), batch_size=BATCH_SIZE, shuffle=False)

    model = SimpleCNN(input_dim=32, input_c=3, output=6, device=DEVICE)
    model.to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss()

    loss_train_all, loss_vali_all = [], []
    for epoch in range(EPOCH):
        print(f"Epoch {epoch + 1}/{EPOCH}")
        train_loss = 0
        vali_loss = 0
        model.train()

        for batch, (src, trg) in enumerate(train_aug_loader):
            src = torch.permute(src, (0, 3, 1, 2)).to(DEVICE)
            trg = trg.to(DEVICE)
            pred = model(src)
            loss = loss_function(pred, trg)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        for batch, (src, trg) in enumerate(train_ori_loader):
            src = torch.permute(src, (0, 3, 1, 2)).to(DEVICE)
            trg = trg.to(DEVICE)
            pred = model(src)
            loss = loss_function(pred, trg)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            for batch, (src, trg) in enumerate(vali_loader):
                src = torch.permute(src, (0, 3, 1, 2)).to(DEVICE)
                trg = trg.to(DEVICE)
                pred = model(src)
                loss = loss_function(pred, trg)
                vali_loss += loss.item()

        train_loss_avg = train_loss / (len(train_aug_loader) + len(train_ori_loader))
        vali_loss_avg = vali_loss / len(vali_loader)
        loss_train_all.append(train_loss_avg)
        loss_vali_all.append(vali_loss_avg)

        print(f"Epoch {epoch + 1}, Train Loss: {train_loss_avg:.4f}, Validation Loss: {vali_loss_avg:.4f}")
        if (epoch + 1) % 25 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss_avg,
            }, f"./SimpleCNN_{epoch + 1}.pt")

    plt.plot(range(EPOCH), loss_train_all, color="red", label='Training')
    plt.plot(range(EPOCH), loss_vali_all, color="green", label='Validation')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("./training.png")

if __name__ == "__main__":
    main()
