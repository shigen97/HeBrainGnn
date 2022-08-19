import torch
from torch import nn, optim
import numpy as np
from models import HeteroGraphCLF
import random
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import pickle
from graph_preprocessing import load_dataloader
from sklearn.metrics import log_loss
import argparse

def adjust_learning_rate(opt, lr):
    for param_group in opt.param_groups:
        param_group["lr"] = param_group["lr"] * lr
    return opt

def mask_prediction(x_train, y_train, x_test, y_test, x_val=None, y_val=None):
    model = LogisticRegression(solver='liblinear', C=1)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    if x_val is not None:
        val_prob = model.predict_proba(x_val)
        val_loss = log_loss(y_val, val_prob, labels=[0, 1])
    else:
        val_loss = None
    acc, f1 = accuracy_score(y_test, y_pred), f1_score(y_test, y_pred)
    return acc, f1, val_loss


def train_HeBrainGnn(args, train, test):
    seed = 123
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    lr = args.lr
    weight_decay = args.weight_decay
    K = args.K
    pre_epoch = args.pre_epoch

    train_loader, test_loader = load_dataloader(train, test, bacth_size=args.batch_size)
    print('finish data loading')
    etypes = ['l-r', 'l-l']

    meta_paths = {
        'l': [('lrl', ['l-r', 'r-l'])],
        'r': [('rlr', ['r-l', 'l-r'])]
    }
    n_epochs = args.epochs
    encode_dropout = args.dropout
    hidden_dims = args.hidden_dim
    device = torch.device('cuda:0')
    model = HeteroGraphCLF(in_dim=246, hidden_dims=hidden_dims, weighted=True, dropout=encode_dropout,
                           activation=nn.PReLU(),
                           residual=True, bias=False, etypes=etypes, meta_paths=meta_paths)

    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer2 = optim.Adam(model.parameters(), lr=2.5e-4, weight_decay=weight_decay)
    loss_func = nn.CrossEntropyLoss().to(device)
    loss_discriminator_func = nn.BCELoss().to(device)
    losses = []
    print('Training Start')
    epoch_accs, epoch_f1s = [], []

    for epoch in range(1, n_epochs+1):
        epoch_loss = 0.0
        model.train()
        for step, (bg, by) in enumerate(train_loader):
            bg = bg.to(device)
            by = by.to(device)
            prediction, pos_scores, neg_scores = model(bg, k=K)
            supervised_loss = loss_func(prediction, by)

            pos_labels, neg_labels = torch.ones(pos_scores.shape).to(device), torch.zeros(neg_scores.shape).to(device)
            discriminator_loss = (loss_discriminator_func(pos_scores, pos_labels) + \
                                  loss_discriminator_func(neg_scores, neg_labels)) / 2
            un_loss = discriminator_loss

            if epoch <= pre_epoch:
                loss = un_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                loss = supervised_loss
                optimizer2.zero_grad()
                loss.backward()
                optimizer2.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / (step + 1))
        
        if epoch >= 35 and epoch % 5 == 0:
            adjust_learning_rate(optimizer2, 0.25)

        if epoch <= pre_epoch:
            with torch.no_grad():
                model.eval()
                x_train, x_test = [], []
                y_train, y_test = [], []
                for step, (bg, by) in enumerate(train_loader):
                    bg = bg.to(device)
                    x = model.encode(bg, 'linear')
                    x_train.extend(x.tolist())
                    y_train.extend(by.numpy().tolist())

                for step, (bg, by) in enumerate(test_loader):
                    bg = bg.to(device)
                    x = model.encode(bg, 'linear')
                    x_test.extend(x.tolist())
                    y_test.extend(by.numpy().tolist())

                x_train, x_test = np.array(x_train), np.array(x_test)
                y_train, y_test = np.array(y_train), np.array(y_test)

                acc, f1, val_loss = mask_prediction(x_train, y_train, x_test, y_test, x_val=None, y_val=None)
                epoch_accs.append(acc)
                epoch_f1s.append(f1)
        else:
            with torch.no_grad():
                model.eval()
                y_pred, y_test = [], []
                for step, (bg, by) in enumerate(test_loader):
                    bg = bg.to(device)
                    prediction, _, _ = model(bg, k=K)
                    prediction = prediction.cpu()
                    pred = torch.argmax(prediction, 1).numpy()
                    y_test.extend(by.numpy().tolist())
                    y_pred.extend(pred.tolist())
                acc, f1 = accuracy_score(y_test, y_pred), f1_score(y_test, y_pred)
                epoch_accs.append(acc)
                epoch_f1s.append(f1)
        print(epoch, losses[-1], acc, f1)

    return epoch_accs[-1], epoch_f1s[-1]


parser = argparse.ArgumentParser()
parser.add_argument("--pre_epochs", type=int, default=20,
                    help="number of pre-training epochs")
parser.add_argument("--epochs", type=int, default=60,
                    help="number of total training epochs")
parser.add_argument("--hidden dimension", type=list, default=[64, 64],
                    help="number of hidden layers and hidden dimensions")
parser.add_argument("--dropout", type=float, default=.8,
                    help="dropout")
parser.add_argument("--lr", type=float, default=1e-4,
                    help="learning rate")
parser.add_argument('--weight-decay', type=float, default=1e-5,
                    help="weight decay")
parser.add_argument('--K', type=int, default=2,
                    help="the number of negative sampling")
parser.add_argument('--batch_size', type=int, default=128,
                    help="batch size")
args = parser.parse_args()
print(args)


skf = StratifiedShuffleSplit(n_splits=5, shuffle=True, random_state=123)
labels = pickle.load(open('labels.pkl', 'rb'))
accs, f1s = [], []
for fold, (train, test) in enumerate(skf.split(np.zeros(labels.shape), labels)):
    acc, f1 = train_HeBrainGnn(args, train, test)
    accs.append(acc)
    f1s.append(f1)

print('accuracy: ', np.mean(accs), np.std(accs), 'f1: ', np.mean(f1s), np.std(f1s))




