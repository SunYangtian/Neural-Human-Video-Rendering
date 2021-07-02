import torch.utils.data
from data.base_data_loader import BaseDataLoader
import numpy as np

def CreateDataset(opt):
    dataset = None
    from data.aligned_dataset import AlignedDataset
    dataset = AlignedDataset()

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)

        if opt.phase == "train":
            ### split train and validation
            self.ratio = opt.data_ratio
            dataset_size = len(self.dataset)
            indices = list(range(dataset_size))
            # np.random.shuffle(indices)
            split = int(self.ratio * dataset_size)
            train_indices, val_indices = indices[:split], indices[split:]
            train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
            if isinstance(opt.max_dataset_size, int):
                import random
                val_indices = random.sample(val_indices, int(opt.max_dataset_size * (1-self.ratio)))
            valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

            self.train_dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=opt.batchSize,
                sampler=train_sampler,
                num_workers=int(opt.nThreads))
            self.valid_dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=opt.batchSize,
                sampler=valid_sampler,
                num_workers=int(opt.nThreads))
        elif opt.phase == "test":
            # self.test_dataloader = torch.utils.data.DataLoader(
            #     self.dataset,
            #     batch_size=opt.batchSize,
            #     shuffle=False,
            #     num_workers=int(opt.nThreads))
            if opt.how_many < 0:
                indices = list(range(opt.start, len(self.dataset)))
            else:
                indices = list(range(opt.start, opt.start + opt.how_many))
            sampler = DirectSequentialSampler(indices)
            self.test_dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=opt.batchSize,
                sampler=sampler,
                num_workers=int(opt.nThreads))

            ### split train and validation
            # self.ratio = opt.data_ratio
            # dataset_size = len(self.dataset)
            # indices = list(range(dataset_size))
            # np.random.shuffle(indices)
            # split = int(self.ratio * dataset_size)
            # train_indices, val_indices = indices[:split], indices[split:]
            # train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
            # valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

            # self.test_dataloader = torch.utils.data.DataLoader(
            #     self.dataset,
            #     batch_size=opt.batchSize,
            #     sampler=valid_sampler,
            #     num_workers=int(opt.nThreads))

    def load_data(self):
        return self.train_dataloader, self.valid_dataloader

    def load_data_test(self):
        return self.test_dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)


class DirectSequentialSampler(torch.utils.data.Sampler):
    r"""Samples elements sequentially, always in the same order.
    Arguments:
        indices: indices of dataset to sample from
    """
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

class CustomDatasetDataLoader_new(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader_new'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.ratio = opt.data_ratio
        dataset_size = len(self.dataset)
        indices = list(range(dataset_size))
        split = int(self.ratio * dataset_size)
        train_indices, val_indices = indices[:split], indices[split:]
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
        valid_sampler = DirectSequentialSampler(val_indices)

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            sampler=valid_sampler,
            num_workers=int(opt.nThreads))
        self.dataset_size = len(val_indices)

    def load_data_test(self):
        return self.dataloader

    def __len__(self):
        return min(self.dataset_size , self.opt.max_dataset_size)