import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from nltk.corpus import stopwords
import datasets

from models import InfoNCE
from utils import evaluate_rouge, sigmoid_focal_loss

stop_words = stopwords.words('english') + ['!',',','.','?','-s','-ly','</s>','s', '(', ")"]



def train_e2e(train_dataloader, model, optimizer, device, step, writer):
    model[0].train()
    model[1].train()
    c_loss = 0
    s_loss = 0
    loss = 0
    batch_num = 0
    print_epo = 500

    for i, data in tqdm(enumerate(train_dataloader)):
        if i % print_epo == 0 and i > 0:
            batch_loss, bc_loss, bs_loss, batch_size = train_e2e_batch(data, model,  optimizer,  if_print=True)
        else:
            batch_loss, bc_loss, bs_loss, batch_size = train_e2e_batch(data, model, optimizer,  if_print=False)

        loss += batch_loss
        c_loss += bc_loss
        s_loss += bs_loss
        batch_num += 1

        if i % print_epo == 0 and i > 0:
            print("Batch {}, Loss: {}".format(i, loss / batch_num))
            print("Batch {}, C-Loss: {}".format(i, c_loss / batch_num))
            print("Batch {}, S-Loss: {}".format(i, s_loss / batch_num))
            sys.stdout.flush()
        writer.add_scalar('Loss/train', loss / batch_num, step)
        step += 1
    return loss / batch_num, step

def train_e2e_batch(data_batch, model,  optimizer,  if_print=True):
    c_model = model[0]
    s_model = model[1]

    optimizer.zero_grad()
    feature = data_batch["feature"]
    adj = data_batch["adj"]
    labels = data_batch["labels_onehot"]
    abs_text = "".join([sen[0].replace("<S>", "").replace("</S>", "") for sen in data_batch["abs_text"]])
    t_xmask = data_batch["x_mask"].unsqueeze(-1)
    t_adjmask = data_batch["adj_mask"]
    secs_mask = data_batch["secs_mask"]

    pg, ng, t, nt = c_model(feature.cuda(), adj.cuda(), None, None, abs_text, None)


    x, selected_idx, x_c, _ = s_model(feature.cuda(), pg, adj.cuda(), [5, 5, 30, 5, 5],
                                      t_xmask=t_xmask.cuda(), t_adjmask=t_adjmask.cuda())

    s_loss = F.binary_cross_entropy_with_logits(x.squeeze(-1), labels.cuda(), pos_weight=torch.tensor(20).cuda())
    pg = pg.squeeze(0)
    infonce = InfoNCE(tau=0.2)

    mask = torch.zeros(1, labels.shape[1] + 6)
    secs_mask[:, -1] = 1
    mask[:, :-6] = labels
    mask[:, -6:] = secs_mask

    neg_mask = 1 - mask
    c_loss = infonce(t, pg, mask.cuda(), neg_mask.cuda())

    loss = s_loss + 0.5 * c_loss
    loss.backward()
    optimizer.step()

    if if_print:
        print("selected_sen:{}".format(selected_idx[0]))
        print("select_prob:{}".format(x_c[0].squeeze(-1)))
        print("summary_sen_idx:{}".format(
            torch.argsort(x.squeeze(-1), dim=-1, descending=True)[0, 0:15]))
        print("output:{}".format(F.sigmoid(x[0].squeeze(-1))))
        print("labels: {}".format(np.argwhere(labels[0].detach() == 1)))

    return loss.data, c_loss.data, s_loss.data, x.shape[0]

def val_e2e(val_dataloader, model,  device, docs):
    rouge = datasets.load_metric('rouge')
    model[0].eval()
    model[1].eval()
    loss = 0
    c_loss = 0
    s_loss = 0

    batch_num = 0
    rouge1_score = []
    rouge2_score = []
    rougel_score = []
    data_num = 0

    all_summaries = []
    all_gt = []
    for i, data in enumerate(val_dataloader):
        cur_loss, c_loss_b, s_loss_b, scores, batch_size = val_e2e_batch(data, model, device)
        loss += cur_loss
        c_loss += c_loss_b
        s_loss += s_loss_b
        data_num += batch_size

        doc_names = data["batch_names"]
        abs_text = "".join([sen[0].replace("<S>", "").replace("</S>", "") for sen in data["abs_text"]])

        ranked_score_idxs = get_summary_ids(scores)


        for i, doc_name in enumerate(doc_names):

            ranked_score_idx = ranked_score_idxs[i]
            summary_text = get_summary_gt_text_w(ranked_score_idx, docs, doc_name, max_word_num=200)
            all_gt.append("\n".join(abs_text.split(".")))
            all_summaries.append("\n".join(summary_text))
            data_num += 1
        batch_num += 1

    rouge_results = rouge.compute(predictions=all_summaries, references=all_gt, use_stemmer=True)
    rouge1_score.append(rouge_results["rouge1"].mid.fmeasure)
    rouge2_score.append(rouge_results["rouge2"].mid.fmeasure)
    rougel_score.append(rouge_results["rougeLsum"].mid.fmeasure)

    rouge1_score = np.mean(rouge1_score)
    loss = loss / batch_num
    c_loss /= batch_num
    s_loss /= batch_num

    return rouge1_score, loss, c_loss, s_loss


def val_e2e_batch(data_batch, model,  device=None):
    c_model = model[0]
    s_model = model[1]
    feature = data_batch["feature"]
    adj = data_batch["adj"]
    labels = data_batch["labels_onehot"]
    abs_text = "".join([sen[0].replace("<S>", "").replace("</S>", "") for sen in data_batch["abs_text"]])
    data_num = feature.shape[0]

    pg, ng, t, nt = c_model(feature.cuda(), adj.cuda(), None, None, abs_text, None)
    x, selected_idx, x_c, x_mask = s_model(feature.cuda(), pg, adj.cuda(), [15, 30, 60, 30, 15])

    pg = pg.squeeze(0)
    infonce = InfoNCE(tau=0.2)

    mask = torch.zeros(1, labels.shape[1] + 6)
    mask[:, :-6] = labels
    mask[:, -6:] = torch.tensor([0, 0, 0, 0, 0, 1])

    neg_mask = 1 - mask
    c_loss = infonce(t, pg, mask.cuda(), neg_mask.cuda())
    s_loss = F.binary_cross_entropy_with_logits(x.squeeze(-1), labels.cuda(), pos_weight=torch.tensor(20).cuda()) #,

    loss = c_loss * 0.5 + s_loss


    scores = torch.sigmoid(x.squeeze(-1))
    return loss.data, c_loss.data, s_loss.data, scores, data_num


def test_e2e_batch(data_batch, model, device, prior_weight = 0, dataset="pubmed"):
    c_model = model[0]
    s_model = model[1]
    feature = data_batch["feature"]
    adj = data_batch["adj"]
    labels = data_batch["labels_onehot"]
    answer_prior = data_batch["answer_prior"]
    abs_text = "".join([sen[0].replace("<S>", "").replace("</S>", "") for sen in data_batch["abs_text"]])
    data_num = feature.shape[0]

    pg, ng, t, nt = c_model(feature.cuda(), adj.cuda(), None, None, abs_text, None)
    if dataset == "pubmed":
        x, selected_idx, x_c, x_mask = s_model(feature.cuda(), pg, adj.cuda(), [15, 30, 60, 30, 15])# pubmed [15, 30, 60, 30, 15] arxiv [10, 30, 100, 30, 10]
    else:
        x, selected_idx, x_c, x_mask = s_model(feature.cuda(), pg, adj.cuda(), [10, 30, 100, 30, 10])

    pg = pg.squeeze(0)
    infonce = InfoNCE(tau=0.2)

    mask = torch.zeros(1, labels.shape[1] + 6)
    mask[:, :-6] = labels
    mask[:, -6:] = torch.tensor([0, 0, 0, 0, 0, 1])

    neg_mask = 1 - mask
    c_loss = infonce(t, pg, mask.cuda(), neg_mask.cuda())
    s_loss = F.binary_cross_entropy_with_logits(x.squeeze(-1), labels.cuda(), pos_weight=torch.tensor(20).cuda())

    loss = c_loss * 0.5 + s_loss


    scores = torch.sigmoid(x.squeeze(-1)) * (1 - prior_weight) * 1  + prior_weight * answer_prior.cuda()


    return loss.data, c_loss.data, s_loss.data, scores, data_num



def test(test_dataloader, model, device, docs, prior_weight=0, dataset="pubmed"):
    model[0].eval()
    model[1].eval()
    rouge = datasets.load_metric('rouge')
    loss = 0
    batch_num = 0

    rouge1_score = []
    rouge2_score = []
    rougel_score = []
    data_num = 0

    all_summaries = []
    all_gt = []
    for j, data in tqdm(enumerate(test_dataloader)):
        cur_loss, c_loss_b, s_loss_b, scores, batch_size = test_e2e_batch(data, model,  device, prior_weight, dataset)
        # print(scores)
        loss += cur_loss
        data_num += batch_size
        doc_names = data["batch_names"]
        abs_text = "".join([sen[0].replace("<S>", "").replace("</S>", "") for sen in data["abs_text"]])
        ranked_score_idxs = get_summary_ids(scores)

        for i, doc_name in enumerate(doc_names):
            ranked_score_idx = ranked_score_idxs[i]
            if dataset == "arxiv":
                summary_text = get_summary_gt_text_w(ranked_score_idx, docs, doc_name, max_word_num=180) #pubmed 250 arxiv 180
            else:
                summary_text = get_summary_gt_text_w(ranked_score_idx, docs, doc_name, max_word_num=250)

            data_num += 1

            all_gt.append("\n".join(abs_text.split(".")))
            all_summaries.append("\n".join(summary_text))
        batch_num += 1


    rouge_results = rouge.compute(predictions=all_summaries, references=all_gt, use_stemmer=True)
    rouge1_score.append(rouge_results["rouge1"].mid.fmeasure)
    rouge2_score.append(rouge_results["rouge2"].mid.fmeasure)
    rougel_score.append(rouge_results["rougeLsum"].mid.fmeasure)

    loss = loss / batch_num

    return rouge1_score, rouge2_score, rougel_score, loss, all_summaries, all_gt


def get_summary_ids(scores):
    # scores : (batch_size, sen_len)
    return torch.argsort(scores, dim=1, descending=True)

def select_by_section(score, sec_adj, keep_num=None):
    if keep_num is None:
        keep_num = [2, 4, 3, 4, 2]
    sen_num = torch.sum(sec_adj, dim=-1)
    masked_score = score * sec_adj
    seleced_idx = []

    sec_order = [0, 1, 2, 3, 4]
    # sec_order = [2, 1, 0, 3, 4]
    for i in range(0, score.shape[0]):
        for j in sec_order:
            keep_idx = keep_num[j]
            if sen_num[i, j] == 0:
                continue
            elif sen_num[i, j] < keep_num[j]:
                keep_idx = sen_num[i, j].int()
            seleced_idx.append(torch.argsort(masked_score[i, j], dim=-1, descending=True)[:keep_idx].unsqueeze(0))
    return torch.concat(seleced_idx, dim=-1)


def get_summary_gt_text(ranked_score_idxs, docs, doc_name, max_sen_num=8,  mode="o", oracle=False, gt=None):
    doc_id = int(doc_name.split(".")[0])
    doc_text = docs[doc_id]["article_text"]
    # get gt
    gt_text = docs[doc_id]['abstract_text']
    for i, text in enumerate(gt_text):
        gt_text[i] = gt_text[i].replace("<S>", "")
        gt_text[i] = gt_text[i].replace("</S>", "")
    # get summary
    summary_text = []
    # summary_len = 0
    summ_trigrams = []

    if oracle:
        for idx in gt:
                summary_text.append(doc_text[idx])
    else:
        for i, idx in enumerate(ranked_score_idxs):
            if idx < len(doc_text):
                if mode == "tri":
                    candidate = doc_text[idx]
                    skip = False
                    c_trigrams = get_trigrams(candidate)
                    if len(summ_trigrams) > 0:
                        for tri in c_trigrams:
                            if tri in summ_trigrams:
                                skip = True
                        if skip:
                            continue
                    summ_trigrams += c_trigrams
                    summary_text.append(candidate)
                else:
                    summary_text.append(doc_text[idx])
            if len(summary_text) >= max_sen_num:
                break
    return summary_text

def get_summary_gt_text_w(ranked_score_idxs, docs, doc_name, max_word_num=200, mode="o"):
    # print("doc_name: {}".format(doc_name))
    doc_id = int(doc_name.split(".")[0])
    doc_text = docs[doc_id]["article_text"]
    # get summary
    summary_text = []
    summ_trigrams = []
    summary_word_num = 0
    for idx in ranked_score_idxs:
        if idx < len(doc_text):
            if mode == "tri":
                candidate = doc_text[idx]
                skip = False
                c_trigrams = get_trigrams(candidate)
                if len(summ_trigrams) > 0:
                    for tri in c_trigrams:
                        if tri in summ_trigrams:
                            skip = True
                    if skip:
                        continue

                summ_trigrams += c_trigrams
                summary_text.append(candidate)
                summary_word_num += len(candidate.split())

            else:
                summary_text.append(doc_text[idx])
                summary_word_num += len(doc_text[idx].split())
        # print(len(summary_text))
        if summary_word_num >= max_word_num:
            break
    return summary_text


def get_trigrams(sen):
    words = [w for w in sen.split() if w not in stop_words]
    tri_grams = []
    for i in range(0, len(words) - 2):
        tri_grams.append([words[i], words[i+1], words[i+2]])
    return tri_grams