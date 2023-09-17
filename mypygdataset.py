import torch
from torch_geometric.data import InMemoryDataset


class PYGDataset(InMemoryDataset):
    def __init__(self, save_root,split_map_root,fold,train_val_test,resample_num,transform=None, pre_transform=None):
        super(PYGDataset, self).__init__(save_root, transform, pre_transform)
        pa_data = torch.load(save_root)
        split_map = torch.load(split_map_root)
        data_fold = split_map[fold]
        split = data_fold[train_val_test]
        dataset = []
        count = [0,0]

        for key in split:
            datas = pa_data[key]
            i = 0
            if datas[0].y.item() == 0:
                resample = resample_num[0]
                count[0] = count[0] + resample
            else:
                resample = resample_num[1]
                count[1] = count[1] + resample
            for data in datas:
                if i >= resample:
                    continue
                dataset.append(data)
                i = i+1
        self.data, self.slices = InMemoryDataset.collate(dataset)
