import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader
from torch.optim import Adam
from Utils.getData import getImageLabel
from Model.CNN import SimpleCNN
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

def main():
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    folds = [1,2,3,4,5]
    DEVICE = 'cuda'

    test_loader = DataLoader(getImageLabel(original=f'C:/Bismillah/Data/Original Images/Original Images/FOLDS/', folds=folds, subdir=['Test']), batch_size=BATCH_SIZE, shuffle=True)

    model = SimpleCNN(input_dim=32, input_c=3, output=6, device=DEVICE)
    model.to(DEVICE)
    loss_function = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    checkpoint = torch.load('SimpleCNN_15.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss_batch = checkpoint['loss']

    prediction, ground_truth, probabilities = [], [], []
    with torch.no_grad():
        model.eval()
        for batch, (src, trg) in enumerate(test_loader):
            src = torch.permute(src, (0, 3, 1, 2))

            pred = model(src)
            probabilities.extend(pred.detach().numpy())
            prediction.extend(torch.argmax(pred,dim=1).detach().numpy())
            ground_truth.extend(torch.argmax(trg, dim=1).detach().numpy())

    classes = ('Chikenpox', 'Cowpox', 'Healty', 'HFMD', 'Measles', 'Monkeypox')

    cf_matrix = confusion_matrix(ground_truth, prediction)
    print(cf_matrix)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig('confusion_matrix.png')

    print("accuracy score = ", accuracy_score(ground_truth, prediction))
    print("precision score = ", precision_score(ground_truth, prediction, average='weighted'))
    print("recall score = ", recall_score(ground_truth, prediction, average='weighted'))
    print("f1 score score = ", f1_score(ground_truth, prediction, average='weighted'))

    ground_truth_bin = label_binarize(ground_truth, classes=list(range(len(classes))))
    probabilities = np.array(probabilities)

    plt.figure(figsize=(10, 8))
    colors = ['red', 'orange', 'blue', 'green', 'yellow', 'purple']
    for i, class_name in enumerate(classes):
        fpr, tpr, _ = roc_curve(ground_truth_bin[:, i], probabilities[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2, label=f'{class_name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Guess (AUC = 0.50)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig('roc_auc.png')

if __name__ == "__main__":
    main()