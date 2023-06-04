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
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import StepwiseSciPaperDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import ConcatDataset
from tqdm import tqdm

from layers import StepWiseGraphConvLayer
from models import BERT_Encoder, Contrast_Encoder,  End2End_Encoder, Contrast_Filter

# os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"
from run import train_e2e, val_e2e

torch.cuda.empty_cache()
torch.set_printoptions(threshold=np.inf)
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=20, help='Random seed.')
parser.add_argument('--sepochs', type=int, default=50, help='Number of epochs to train.')
parser.add_argument('--cepochs', type=int, default=50, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=5e-5, help='Initial learning rate.')
parser.add_argument('--hidden', type=int, default=2048, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument("--doc_path", type=str, help="Document Text Path")
parser.add_argument("--model_save_path", type=str)
parser.add_argument("--data_path", type=str)
parser.add_argument("--data_name", type=str)
parser.add_argument("--model", type=str)
parser.add_argument("--subset_size", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--val_size", type=int, default=0)
parser.add_argument("--cweight_path", type=str, default=None)
parser.add_argument("--sweight_path", type=str, default=None)
parser.add_argument("--use_pos", action="store_true")
parser.add_argument("--pos_weight", type=int, default=26)
parser.add_argument("--train_c", action="store_true")


writer = SummaryWriter()
args = parser.parse_args()
print(args)
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
print(device)

c_epochs = args.cepochs
s_epochs = args.sepochs
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
train_contra = args.train_c

if args.pos_weight==0:
    pos_weight = None
else:
    pos_weight = args.pos_weight

# Load data
print("Reading Dataset")
print("Train on {} Data".format(subset_size))
train_dataset = StepwiseSciPaperDataset(data_root_path=data_path, dataset_name=data_name, split="newtrain",
                                        subset=subset_size, random_state=args.seed, use_pos_feature=False, truncate=-300)
val_dataset = StepwiseSciPaperDataset(data_root_path=data_path, dataset_name=data_name, split="val",
                                      subset=args.val_size , random_state=0,  use_pos_feature=False, truncate=-300)



train_dataloader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, num_workers=0)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=0)

docs = val_dataset.docs

print("Building Model")
c_graph_encoder = StepWiseGraphConvLayer(in_dim=768, out_dim=args.hidden, type_num=6, hid_dim=args.hidden,
                                         dropout_p=args.dropout, act=torch.nn.LeakyReLU(),fusion=False, graph=True, nheads=8, iter=1).to(device)
s_graph_encoder = StepWiseGraphConvLayer(in_dim=768, out_dim=args.hidden, type_num=6, hid_dim=args.hidden,
                                         dropout_p=args.dropout, act=torch.nn.LeakyReLU(), fusion=False, graph=True, nheads=8, iter=1).to(device)

bert_encoder = BERT_Encoder().to(device)
for p in bert_encoder.parameters():
    p.requires_grad = False

contrast_encoder = Contrast_Encoder(c_graph_encoder, bert_encoder, args.hidden, mode="train", dropout_p=args.dropout).to(device)
contrast_filter = Contrast_Filter(contrast_encoder, 768, args.hidden, args.dropout).to(device)

summarization_encoder = End2End_Encoder(s_graph_encoder, 768, args.hidden, args.dropout).to(device)

if args.sweight_path is not None:
    print("loading summarization model from {}".format(args.sweight_path))
    summarization_encoder.load_state_dict(torch.load(args.sweight_path), strict=False)

if args.cweight_path is not None:
    print("loading contrast filter from {}".format(args.cweight_path))
    contrast_filter.load_state_dict(torch.load(args.cweight_path), strict=False)

print("Building Optimizer")
optimizer = optim.Adam([ {'params': summarization_encoder.parameters()},
                {'params': contrast_filter.parameters()}], lr=args.lr, weight_decay=1e-5)

best_r1 = 0
best_loss = 10000
train_loss = []
val_loss = []
step = 0
c_weight = 1
s_weight = 1
print("Start Training Summary Model")
c_patient = 30
s_patient = 30
train_s = True
train_c = args.train_c
for i in range(s_epochs):
    print("Epoch {}".format(i))
    if train_c and c_patient > 0:
        c_patient -= 1
    else:
        train_c = False
        for p in contrast_filter.parameters():
            p.requires_grad = False

        print("Stop Training Contrast")

    if s_patient > 0:
        train_s = True
        s_patient -= 1
    else:
        train_s = False
        c_patient = 1


    model = [contrast_filter, summarization_encoder]
    loss, step = train_e2e(train_dataloader,  model,  optimizer, device, step, writer)
    train_loss.append(loss)
    print("At Epoch {}, Train Loss: {}".format(i, loss))

    torch.cuda.empty_cache()

    print("Validating")
    rouge1_score, loss, c_loss, s_loss = val_e2e(val_dataloader, model, device, docs)
    # scheduler.step(rouge1_score)
    writer.add_scalar('Loss/val', loss, i)
    writer.add_scalar('Rouge 1/val', rouge1_score, i)
    torch.cuda.empty_cache()

    print("At Epoch {}, Val Loss: {}, Val CLoss: {}, Val SLoss: {},Val R1: {}".format(i, loss, c_loss, s_loss, rouge1_score))
    if rouge1_score > best_r1:
        model_save_path = os.path.join(model_save_root_path, "e_{}_{}.mdl".format(i, rouge1_score))
        torch.save(summarization_encoder.state_dict(), model_save_path)

        model_save_path = os.path.join(model_save_root_path, "c_{}_{}.mdl".format(i, rouge1_score))
        torch.save(contrast_filter.state_dict(), model_save_path)
        best_r1 = rouge1_score
        print("Epoch {} Has best R1 Score of {}, saved Model to {}".format(i, best_r1, model_save_path))

print("Finished")



