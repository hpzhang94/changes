import torch
from torch.utils.data import Dataset
import os
import numpy as np
from numpy.random import default_rng
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


def label_to_onehot(gts, sen_len):
    one_hot = np.zeros((sen_len))
    for gt in gts:
        one_hot[gt] = 1
    return one_hot


class StepwiseSciPaperDataset(Dataset):
    def __init__(self, data_root_path, split="newtest", dataset_name="pubmed", padding=True, subset=0, random_state=None, use_pos_feature=True, truncate=-300):
        assert split in ["train", "test", "val", "newtest", "newtrain"]
        assert dataset_name in ["pubmed", "arxiv"]
        self.dataset_name = dataset_name
        self.use_pos_feature = use_pos_feature
        self.random_state = random_state
        self.truncate = truncate

        if split == "val":
            doc_split = dataset_name
        else:
            doc_split = split
        self.doc_path = os.path.join(data_root_path, "filtered-{}/{}.npy".format(dataset_name, doc_split))
        self.docs = np.load(self.doc_path, allow_pickle=True)


        # self.data_path = os.path.join(data_root_path, "filtered-{}/Stepwise_Construction/features".format(dataset_name),
        #                               split)

        self.feature_path = os.path.join(data_root_path, dataset_name, "feature")
        self.label_path = os.path.join(data_root_path, "filtered-{}/Stepwise_Construction/labels".format(dataset_name),
                                       split)
        self.prior_path = os.path.join(data_root_path, "filtered-{}/Stepwise_Construction/priors".format(dataset_name),
                                       split)
        # print(self.data_path)
        # print(self.label_path)
        data_names = os.listdir(self.data_path)
        data_names.sort()

        label_names = os.listdir(self.label_path)
        label_names.sort()

        data_names = np.array(data_names)
        label_names = np.array(label_names)

        # assert len(self.docs) == len(data_names)
        if subset > 0:
            if random_state:
                rng = default_rng(random_state)
                idxs = rng.choice(range(0, len(data_names)), subset, replace=False)
                data_names = data_names[idxs]
                label_names = label_names[idxs]

            else:
                data_names = data_names[0:subset]
                label_names = label_names[0:subset]

        self.data_names = data_names
        self.label_names = label_names
        # assert len(data_names) == len(label_names)
        self.padding = padding

    def __len__(self):
        return len(self.data_names)

    def __getitem__(self, idx):
        name = self.data_names[idx]
        doc_idx = int(name.split(".")[0])
        abstract_text = self.docs[doc_idx]["abstract_text"]

        data_path = os.path.join(self.data_path, name)
        data = np.load(data_path, allow_pickle=True).item()

        # feature = data["features"]
        feature = np.load(os.path.join(self.feature_path, "{}.npz".format(doc_idx)))["feature"]
        if self.use_pos_feature:
            feature = feature
        else:
            feature = feature[:, 0:-4]

        sen_sec_mask = data["sen_sec_mask"]
        adj, sen_num, secs_mask = mask_to_adj(sen_sec_mask)

        label_path = os.path.join(self.label_path, name)
        label = np.load(label_path)

        label_one_hot = label_to_onehot(label, sen_num)

        prior_path = os.path.join(self.prior_path, name)
        if os.path.exists(prior_path):
            answer_prior = np.load(prior_path)
        else:
            answer_prior = np.zeros_like(label_one_hot)
        keep_num = 120
        x_mask, adj_mask = generate_teacher_mask(feature, adj, label, keep_num)
        truncate_idx = self.truncate

        ret = {"feature": torch.tensor(feature, dtype=torch.float32).squeeze(0)[truncate_idx:, :],
               "adj": torch.tensor(adj, dtype=torch.float32).squeeze(0)[truncate_idx:, truncate_idx:],
               "labels": torch.tensor(label, dtype=torch.int32),
               "labels_onehot": torch.tensor(label_one_hot, dtype=torch.float32).squeeze(0)[truncate_idx + 6:, ],
               "answer_prior": torch.tensor(answer_prior, dtype=torch.float32).squeeze(0)[truncate_idx + 6:, ],
               "batch_names": name,
               "abs_text": abstract_text,
               "x_mask": torch.tensor(x_mask, dtype=torch.int32).squeeze(0)[truncate_idx:, ],
               "adj_mask": torch.tensor(adj_mask, dtype=torch.int32).squeeze(0)[truncate_idx:, truncate_idx:],
               "secs_mask": torch.tensor(secs_mask, dtype=torch.int32).squeeze(0)
               }
        return ret


def generate_teacher_mask(x, adj, label, keep_sen_num=100):
    x_mask = np.ones_like(x)
    adj_mask = np.ones_like(adj)
    if x.shape[0] > keep_sen_num:
        del_num = x.shape[0] - keep_sen_num
        while del_num > 0:
            rng = default_rng()
            mask_pos = rng.choice(range(0, x.shape[0]))
            while mask_pos in label:
                mask_pos = rng.choice(range(0, x.shape[0]))
            x_mask[mask_pos, :] = 0
            adj_mask[mask_pos, :] = 0
            adj_mask[:, mask_pos] = 0
            del_num -= 1
    return x_mask[:, 0], adj_mask




def mask_to_adj(sen_sec_mask):
    sen_num = sen_sec_mask.shape[1]
    sec_num = sen_sec_mask.shape[0]
    adj = np.zeros((sen_num+sec_num, sen_num+sec_num))
    # section connection
    secs_mask = np.sum(sen_sec_mask, axis=1)
    secs_mask[secs_mask > 0] = 1
    secs_mask[-1] = 0
    # print(secs_mask)
    adj[-sec_num:, 0:-sec_num] = sen_sec_mask
    adj[-sec_num:, -sec_num:] = secs_mask
    # adj[-sec_num:, -sec_num:] = 0
    #document connection
    adj[-1, -sec_num:] = 1
    adj[-1, :-sen_num] = 1
    #build sentence connection
    start = 0


    for i in range(0, sec_num):
        sec_mask = sen_sec_mask[i]

        sec_sen_num = int(np.sum(sec_mask))
        adj_sec = np.zeros((sec_sen_num, sen_num + sec_num))
        adj_sec[:, :sen_num] = sec_mask
        # adj_sec[:, :sen_num] = 1
        # adj_sec[:, sen_num + i] = 1
        adj_sec[:, -sec_num:-1] = secs_mask[:-1]
        adj_sec[:, -1] = 0

        adj[start: start + sec_sen_num, :] = adj_sec
        start += sec_sen_num
    return adj, sen_num, secs_mask



