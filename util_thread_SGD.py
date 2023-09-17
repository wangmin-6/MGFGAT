import torch.nn.functional as F
from torch.optim import SGD
from torch_geometric.loader import DataLoader
import time
import torch
import pandas as pd
import numpy as np
from mypygdataset import PYGDataset
import datetime
from sklearn.metrics import roc_auc_score, accuracy_score


# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

def logger(info):
    fold, epoch = info['fold'] + 1, info['epoch']
    val_loss, test_acc = info['val_loss'], info['test_acc']
    train_loss = info['train_loss']
    val_acc = info['val_acc']
    train_acc = info['train_acc']
    test_sen = info['test_sen']
    test_spec = info['test_spec']
    test_prec = info['test_prec']
    test_recall = info['test_recall']
    test_F1 = info['test_F1']
    print(f'{fold:02d}/{epoch:03d}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
          f'Train Accuracy: {train_acc:.3f}, Val Accuracy: {val_acc:.3f}, Test Accuracy: {test_acc:.3f}')
    print(f'Test Sen: {test_sen:.3f}, Test Spec: {test_spec:.3f}, Test Prec: {test_prec:.3f},'
          f' Test Recall: {test_recall:.3f}, Test F1: {test_F1:.3f}')


def num_graphs(data):
    if hasattr(data, 'num_graphs'):
        return data.num_graphs
    else:
        return data.x.size(0)


def train(model, optimizer, loader, device):
    model.train()
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data)
        loss = F.nll_loss(out, data.y.view(-1))
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        optimizer.step()
    return total_loss / len(loader.dataset)


def eval_acc(model, loader, device):
    model.eval()
    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0
    correct = 0
    labels = []
    preds = []
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
        y = data.y.view(-1)
        for i in range(y.shape[0]):
            a = y[i].item()
            b = pred[i].item()
            labels.append(a)
            preds.append(b)
            if a == 1 and b == 1:
                true_positive = true_positive + 1
            if a == 1 and b == 0:
                false_negative = false_negative + 1
            if a == 0 and b == 0:
                true_negative = true_negative + 1
            if a == 0 and b == 1:
                false_positive = false_positive + 1

    auc = roc_auc_score(labels, preds)
    acc = accuracy_score(labels, preds)
    if (true_positive + false_negative) == 0:
        sensitivity = 0
    else:
        sensitivity = true_positive / (true_positive + false_negative)

    if (false_positive + true_negative) == 0:
        specificity = 0
    else:
        specificity = true_negative / (false_positive + true_negative)

    if (true_positive + false_positive) == 0:
        precision = 0
    else:
        precision = true_positive / (true_positive + false_positive)
    if (true_positive + false_negative) == 0:
        recall = 0
    else:
        recall = true_positive / (true_positive + false_negative)

    if (precision + recall) == 0:
        F1 = 0
    else:
        F1 = 2 * (precision * recall) / (precision + recall)

    evals = {'acc': correct / len(loader.dataset),
             'sensitivity': sensitivity,
             'specificity': specificity,
             'precision': precision,
             'recall': recall,
             'F1': F1,
             'auc': auc}
    return evals


def eval_loss(model, loader, device):
    model.eval()

    loss = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
        loss += F.nll_loss(out, data.y.view(-1), reduction='sum').item()
    return loss / len(loader.dataset)


def fun(Net, num_layers, hidden, args, class1, class2, resample_num, num_roi, log_root, data_root, map_root, device,
        file_name):
    print(f'--- {Net.__name__}' + " " + str(num_layers) + "_" + str(hidden), class1, class2, num_roi)
    # dataset = PYGDataset("data/fac/Philips/fusion/", class1, class2)
    test_accs = np.zeros(10)
    test_sens = np.zeros(10)
    test_specs = np.zeros(10)
    test_recalls = np.zeros(10)
    test_F1s = np.zeros(10)
    test_aucs = np.zeros(10)
    test_precisions = np.zeros(10)
    for fold in np.arange(10):
        log_path = log_root + str(num_roi) + "/" + class1 + "_" + class2 + " " + str(
            num_roi) + " " + Net.__name__ + " " + str(
            num_layers) + "_" + str(hidden) + " fold " + str(fold)
        model_path = log_root + str(num_roi) + "/" + "model/" + class1 + "_" + class2 + " " + str(
            num_roi) + " " + Net.__name__ + args.flag + " " + str(
            num_layers) + "_" + str(
            hidden) + " fold " + str(fold)

        train_dataset = PYGDataset(data_root, map_root, fold, "train", resample_num)
        val_dataset = PYGDataset(data_root, map_root, fold, "val", resample_num)
        test_dataset = PYGDataset(data_root, map_root, fold, "test", resample_num)
        train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)
        model = Net(train_dataset, num_layers, hidden)
        model.to(device).reset_parameters()
        optimizer = SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # if torch.cuda.is_available():
        #     print(torch.cuda)
        #     torch.cuda.synchronize()
        best_loss = 1e10
        best_acc = 0
        best_test_acc = 0
        log = []
        for epoch in range(0, args.epochs):
            train_loss = train(model, optimizer, train_loader, device)
            val_loss = eval_loss(model, val_loader, device)
            train_evals = eval_acc(model, train_loader, device)
            val_evals = eval_acc(model, val_loader, device)
            test_evals = eval_acc(model, test_loader, device)
            eval_info = {
                'fold': fold,
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_evals["acc"],
                'val_acc': val_evals["acc"],
                'test_acc': test_evals["acc"],
                'test_sen': test_evals["sensitivity"],
                'test_spec': test_evals["specificity"],
                'test_prec': test_evals["precision"],
                'test_recall': test_evals["recall"],
                'test_F1': test_evals["F1"],
            }
            log.append([fold, epoch, train_loss, val_loss, train_evals["acc"], val_evals["acc"],
                        test_evals["acc"], test_evals["sensitivity"], test_evals["specificity"],
                        test_evals["precision"], test_evals["recall"], test_evals["F1"], test_evals["auc"]])
            if args.is_lr_decay_factor and epoch % args.lr_decay_step_size == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.lr_decay_factor * param_group['lr']

            if test_evals["acc"] > best_test_acc:
                best_test_acc = test_evals["acc"]

            if best_loss > val_loss and epoch > 5:
                print("saving best model", "roi:", num_roi, "fold", fold,
                      "epoch:", epoch, "val_loss:", val_loss, "test_acc:", test_evals["acc"], "auc", test_evals["auc"])
                best_loss = val_loss
                test_accs[fold] = test_evals["acc"]
                test_sens[fold] = test_evals["sensitivity"]
                test_specs[fold] = test_evals["specificity"]
                test_recalls[fold] = test_evals["recall"]
                test_F1s[fold] = test_evals["F1"]
                test_precisions[fold] = test_evals["precision"]
                test_aucs[fold] = test_evals["auc"]
                torch.save(model, model_path + str(args.lr) + str(args.weight_decay) + ".pth")
                # best_model_wts = copy.deepcopy(model.state_dict())
        log_path = log_path + " " + args.flag + \
                   f' acc {test_accs[fold]:.3f} sen {test_sens[fold]:.3f} ' \
                   f'spec {test_specs[fold]:.3f} prec {test_precisions[fold]:.3f} ' \
                   f'recall {test_recalls[fold]:.3f} F1 {test_F1s[fold]:.3f} auc {test_aucs[fold]:.3f}.csv'
        # log_path = log_path + f'acc {best_test_acc:.3f} val_acc {best_loss_acc:.3f}.csv'
        print(log_path)
        dataFrame = pd.DataFrame(log)  # 你 ,w deng yi xia zai shang lai
        dataFrame.to_csv(log_path, index=False, sep=',')

    now = datetime.datetime.now()
    # print("mean", np.average(test_accs), "var", np.var(test_accs))
    folder_eval = [class1 + "_" + class2, num_roi, Net.__name__ + args.flag, num_layers, hidden,
                   now.strftime('%Y-%m-%d-%H'),
                   args.lr, args.weight_decay,
                   np.average(test_accs), np.var(test_accs),
                   np.average(test_sens), np.var(test_sens),
                   np.average(test_specs), np.var(test_specs),
                   np.average(test_precisions), np.var(test_precisions),
                   np.average(test_recalls), np.var(test_recalls),
                   np.average(test_F1s), np.var(test_F1s),
                   np.average(test_aucs), np.var(test_aucs)]
    print(num_roi, "AVG FOLDER", folder_eval)
    test = pd.DataFrame(data=[folder_eval])
    test.to_csv(log_root + file_name, index=False, encoding='utf-8', mode='a', header=False)
    return np.average(test_accs)
