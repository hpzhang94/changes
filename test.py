from __future__ import division
from __future__ import print_function

import argparse
import os
import random
import sys

import GCL.losses as L
import numpy as np
import torch
import torch.optim as optim
from GCL.models import SingleBranchContrast, DualBranchContrast

from dataset import StepwiseSciPaperDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import ConcatDataset
from tqdm import tqdm

from layers import StepWiseGraphConvLayer
from models import BERT_Encoder, Contrast_Encoder, End2End_Encoder, Contrast_Filter

# os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"
from run import  train_e2e, val_e2e, test

torch.cuda.empty_cache()
torch.set_printoptions(threshold=np.inf)
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=20, help='Random seed.')
parser.add_argument('--hidden', type=int, default=2048, help='Number of hidden units.')
parser.add_argument("--doc_path", type=str, help="Document Text Path")
parser.add_argument("--model_save_path", type=str)
parser.add_argument("--data_path", type=str)
parser.add_argument("--data_name", type=str)
parser.add_argument("--model", type=str)
parser.add_argument("--subset_size", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--val_size", type=int, default=0)
parser.add_argument("--cweight_path", type=str, default=None)
parser.add_argument("--sweight_path", type=str, default=None)
parser.add_argument("--use_pos", action="store_true")
parser.add_argument("--pos_weight", type=int, default=26)
parser.add_argument("--prior_weight", type=float, default=0)

args = parser.parse_args()
print(args)
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
print(device)


random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

data_path = args.data_path
data_name = args.data_name
doc_path = args.doc_path
model_save_root_path = args.model_save_path
model_name = args.model
subset_size = args.subset_size

if args.pos_weight==0:
    pos_weight = None
else:
    pos_weight = args.pos_weight

# Load data
# adj, features, labels, idx_train, idx_val, idx_test = load_data()
torch.autograd.set_detect_anomaly(True)
print("Reading Dataset")
test_dataset = StepwiseSciPaperDataset(data_root_path=data_path, dataset_name=data_name, split="newtest", subset=args.subset_size , random_state=0,  use_pos_feature=False, truncate=-300)
test_data_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0)


docs = test_dataset.docs

print("Building Model")
c_graph_encoder = StepWiseGraphConvLayer(in_dim=768, out_dim=args.hidden, type_num=6, hid_dim=args.hidden, dropout_p=args.dropout, act=torch.nn.LeakyReLU(),fusion=False, graph=True, nheads=8, iter=1).to(device)
s_graph_encoder = StepWiseGraphConvLayer(in_dim=768, out_dim=args.hidden, type_num=6, hid_dim=args.hidden, dropout_p=args.dropout, act=torch.nn.LeakyReLU(), fusion=False, graph=True, nheads=8, iter=1).to(device)

bert_encoder = BERT_Encoder().to(device)
for p in bert_encoder.parameters():
    p.requires_grad = False

contrast_encoder = Contrast_Encoder(c_graph_encoder, bert_encoder, args.hidden, mode="train", dropout_p=args.dropout).to(device)
contrast_filter = Contrast_Filter(contrast_encoder, 768, args.hidden, args.dropout).to(device)

summarization_encoder = End2End_Encoder(s_graph_encoder, 768, args.hidden, args.dropout).to(device)
for p in summarization_encoder.parameters():
    p.requires_grad = False


if args.sweight_path is not None:
    print("loading summarization model from {}".format(args.sweight_path))
    summarization_encoder.load_state_dict(torch.load(args.sweight_path), strict=False)

if args.cweight_path is not None:
    print("loading contrast filter from {}".format(args.cweight_path))
    contrast_filter.load_state_dict(torch.load(args.cweight_path), strict=False)

model = [contrast_filter, summarization_encoder]
print("Start Testing")
rouge1_score, rouge2_score, rougel_score, loss, all_summaries, all_gt = test(test_data_loader, model, device, docs,
                                                                             prior_weight=args.prior_weight, dataset=data_name)

rouge1_score_avg = np.mean(rouge1_score)
rouge2_score_avg = np.mean(rouge2_score)
rougel_score_avg = np.mean(rougel_score)


print("Test Finished \n"
      "Test Rouge-2 Score is {} \n"
      "Test Rouge-1 Score is {}\n"
      "Test Rouge-L Score is {} \n"
      "Test Loss: {} \n"
      "Summary Example: {} \n"
      "GT is: {}".format(rouge2_score_avg, rouge1_score_avg, rougel_score_avg, loss, all_summaries[0], all_gt[0]))
