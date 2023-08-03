import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm

__all__ = ['SepsisDataset', 'parse_datasets']


class PreDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


class SepsisDataset(Dataset):

    def __init__(self,
                 args,
                 read_path,
                 is_external=False,
                 data_min=None,
                 data_max=None,
                 normalization=True,
                 shuffle=True,
                 device='cpu'):
        self.read_path = read_path
        self.device = device
        self.data = torch.load(read_path, map_location=device)
        self.args = args
        self.shuffle = shuffle
        self.is_external = is_external
        self.data_min = data_min
        self.data_max = data_max

        if is_external:
            self.externalset = self.data
            if normalization:
                self.data_norm = PreDataset(self.data_normalization())
                self.external_loader = DataLoader(self.data_norm, shuffle=self.shuffle, batch_size=self.args.batch_size)
            else:
                self.external_loader = DataLoader(self.data)
        else:
            if normalization:
                if data_min is None or data_max is None:
                    self.data_min, self.data_max = self.get_data_min_max(self.data, device)
                self.data_norm = self.data_normalization()
                self.trainset, self.validset, self.testset = self.get_data_set(self.data_norm)
            else:
                self.trainset, self.validset, self.testset = self.get_data_set(self.data)
            self.trainloader, self.validloader, self.testloder = self.get_data_loader()

    @staticmethod
    def data_describe(dataset):
        label_all = []
        for _, _, _, _, labels, _ in dataset:
            label_all.append(labels)
        label_all = torch.tensor(label_all).detach().cpu().numpy()
        sepsis_count = sum(label_all == 1)
        nosepsis_count = sum(label_all == 0)
        return nosepsis_count, sepsis_count

    def data_normalization(self):
        data_norm = []
        for i in range(len(self.data)):
            record_id = self.data[i][0]
            tt = self.data[i][1]
            vals = (self.data[i][2] - self.data_min) / (self.data_max - self.data_min)
            masks = self.data[i][3]
            labels = self.data[i][4]
            sofas = self.data[i][5]
            data_norm.append((record_id, tt, vals, masks, labels, sofas))
        return data_norm

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def __repr__(self):
        if self.is_external:
            external_info = self.data_describe(self.data)
            info = f'externalset: {len(self.externalset):>6d}\t0: {external_info[0]:>6d}\t1: {external_info[1]:>6d}\n'
        else:
            train_info = self.data_describe(self.trainset)
            valid_info = self.data_describe(self.validset)
            test_info = self.data_describe(self.testset)
            info = f'trainset: {len(self.trainset):>6d}\t0: {train_info[0]:>6d}\t1: {train_info[1]:>6d}\n'\
                   f'validset: {len(self.validset):>6d}\t0: {valid_info[0]:>6d}\t1: {valid_info[1]:>6d}\n'\
                   f'testset : {len(self.testset ):>6d}\t0: {test_info[0] :>6d}\t1: {test_info[1] :>6d}\n'
        return info

    @ staticmethod
    def get_data_min_max(records, device='cpu'):
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        data_min, data_max = None, None
        inf = torch.Tensor([float("Inf")])[0].to(device)

        for b, (record_id, tt, vals, mask, labels, sofas) in enumerate(records):
            n_features = vals.size(-1)

            batch_min = []
            batch_max = []
            for i in range(n_features):
                non_missing_vals = vals[:, i][mask[:, i] == 1]
                if len(non_missing_vals) == 0:
                    batch_min.append(inf)
                    batch_max.append(-inf)
                else:
                    batch_min.append(torch.min(non_missing_vals))
                    batch_max.append(torch.max(non_missing_vals))

            batch_min = torch.stack(batch_min)
            batch_max = torch.stack(batch_max)

            if (data_min is None) and (data_max is None):
                data_min = batch_min
                data_max = batch_max
            else:
                data_min = torch.min(data_min, batch_min)
                data_max = torch.max(data_max, batch_max)

        return data_min, data_max

    def get_data_set(self, dataset):
        trainset, testset = train_test_split(dataset,
                                             test_size=self.args.test_size,
                                             random_state=self.args.random_seed)
        trainset, validset = train_test_split(trainset,
                                              test_size=self.args.valid_size / (1-self.args.test_size),
                                              random_state=self.args.random_seed)
        return trainset, validset, testset

    def get_data_loader(self):
        trainloader = DataLoader(self.trainset, shuffle=self.shuffle, batch_size=self.args.batch_size)
        validloader = DataLoader(self.validset, shuffle=False, batch_size=self.args.batch_size)
        testloader = DataLoader(self.testset, shuffle=False, batch_size=self.args.batch_size)
        return trainloader, validloader, testloader


def parse_datasets(args, read_path, device):
    dataset


if __name__ == '__main__':
    pass
