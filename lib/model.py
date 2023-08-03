import os.path

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score

__all__ = ['compute_loss', 'get_result_score', 'train', 'test', 'SepsisLSTM']


def compute_loss():
    loss = nn.BCELoss()
    return loss


def get_result_score(loss, labels, pred, writer, step, state='train'):
    labels = torch.concat(labels).detach().cpu().numpy()
    pred = torch.concat(pred).detach().cpu().numpy()
    auc = roc_auc_score(labels, pred)
    ap = average_precision_score(labels, pred)
    accuracy = accuracy_score(labels, [1 if val >= 0.5 else 0 for val in pred])
    writer.add_scalar(tag=f'{state}/loss', scalar_value=loss, global_step=step)
    writer.add_scalar(tag=f'{state}/auc', scalar_value=auc, global_step=step)
    writer.add_scalar(tag=f'{state}/ap', scalar_value=ap, global_step=step)
    writer.add_pr_curve(tag=f'{state}/pr', labels=labels, predictions=pred, global_step=step)
    return auc, ap, accuracy, labels, pred


def train(args,
          model,
          optimizer,
          loss,
          trainloader,
          validloader,
          epoch,
          logger,
          writer,
          ckpt_path='./experiments/'):
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    auc_train_best = 0
    auc_valid_best = 0
    for i in range(epoch):
        model.train()
        error_train_all = 0
        train_labels = []
        train_pred = []
        for iid, tt, vals, masks, labels, sofa in trainloader:
            labels = labels.reshape(-1, 1)
            y_pred = model((vals, masks))
            optimizer.zero_grad()
            error = loss(y_pred, labels)
            error.backward()
            optimizer.step()
            error_train_all += error
            train_labels.append(labels)
            train_pred.append(y_pred)
        error_train_all = error_train_all / len(trainloader)
        train_auc, train_ap, train_accuracy, train_labels, train_pred = get_result_score(error_train_all,
                                                                                         train_labels,
                                                                                         train_pred,
                                                                                         writer=writer,
                                                                                         step=i,
                                                                                         state='train')
        auc_train_best = max(auc_train_best, train_auc)

        model.eval()
        error_valid_all = 0
        valid_labels = []
        valid_pred = []
        with torch.no_grad():
            for iid, tt, vals, masks, labels, sofa in validloader:
                labels = labels.reshape(-1, 1)
                y_pred = model((vals, masks))
                error = loss(y_pred, labels)
                error_valid_all += error
                valid_labels.append(labels)
                valid_pred.append(y_pred)
        error_valid_all = error_valid_all / len(validloader)
        valid_auc, valid_ap, valid_accuracy, valid_labels, valid_pred = get_result_score(error_valid_all,
                                                                                         valid_labels,
                                                                                         valid_pred,
                                                                                         writer=writer,
                                                                                         step=i,
                                                                                         state='valid')

        torch.save({'args': args,
                    'model_state_dict': model.state_dict(),
                    'optim_state_dict': optimizer.state_dict()
                    }, f'{ckpt_path}epoch_{i}_auc_{valid_auc:.3f}.ckpt')
        if auc_valid_best <= valid_auc:
            auc_valid_best = valid_auc
            # print(i, auc_best, valid_auc)
            torch.save({'args': args,
                        'epoch': i,
                        'auc': auc_valid_best,
                        'model_state_dict': model.state_dict(),
                        'optim_state_dict': optimizer.state_dict()
                        }, f'{ckpt_path}best_model.ckpt')
        logger.info(f'epoch: {i}\n'
                    f'train: loss: {error_train_all:.6f}\taccuracy_score: {train_accuracy:.6f}\t'
                    f'auc: {train_auc:.6f}\tap: {train_ap:.6f}\n'
                    f'valid: loss: {error_valid_all:.6f}\taccuracy_score: {valid_accuracy:.6f}\t'
                    f'auc: {valid_auc:.6f}\tap: {valid_ap:.6f}\n')
    return model, auc_train_best, auc_valid_best


def test(model,
         loss,
         testloader,
         logger,
         writer,
         ckpt_path,
         external_center='test',
         state='test'):
    best_model = torch.load(ckpt_path + 'best_model.ckpt')
    model.load_state_dict(best_model['model_state_dict'])
    model.eval()
    error_test_all = 0
    test_labels = []
    test_pred = []
    best_epoch = best_model['epoch']
    best_auc = best_model['auc']
    with torch.no_grad():
        for iid, tt, vals, masks, labels, sofa in testloader:
            labels = labels.reshape(-1, 1)
            y_pred = model((vals, masks))
            error = loss(y_pred, labels)
            error_test_all += error
            test_labels.append(labels)
            test_pred.append(y_pred)
    error_test_all = error_test_all / len(testloader)
    test_auc, test_ap, test_accuracy, test_labels, test_pred = get_result_score(error_test_all,
                                                                                test_labels,
                                                                                test_pred,
                                                                                writer=writer,
                                                                                step=best_model['epoch'],
                                                                                state=state)
    logger.info(f'test for data: {external_center}\n'
                f'test: loss: {error_test_all:.6f}\taccuracy_score: {test_accuracy:.6f}\t'
                f'auc: {test_auc:.6f}\tap: {test_ap:.6f}\n')
    return model, test_auc


class SepsisLSTM(nn.Module):

    def __init__(self, in_dim, hidden_dim, n_layer, n_classes):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.n_layer = n_layer
        self.n_classes = n_classes
        self.lstm_val = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True)
        self.lstm_mask = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True)
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, n_classes), nn.Sigmoid())

    def forward(self, x):
        out_val, (h_val, c_val) = self.lstm_val(x[0])
        out_mask, (h_mask, c_mask) = self.lstm_mask(x[1])
        y_val = h_val[-1, :, :]
        y_mask = h_mask[-1, :, :]
        y = self.classifier(y_val * y_mask)
        return y


if __name__ == '__main__':
    sepsis_model = SepsisLSTM(5, 2, 2, 1)
