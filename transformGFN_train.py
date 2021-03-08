#Albert Transform GFN train
import argparse
import math
import time
import torch.nn.functional as F
import dill as pickle
from torchtext.datasets import TranslationDataset
from torchtext.data import Field, Dataset, BucketIterator
import transpytorch.transformer.Constants as Constants
import os
import re
import json
from tqdm import tqdm, trange
import pdb
import random
from collections import namedtuple
import numpy as np
import copy
import torch
import traceback
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from pytorch_pretrained_bert.tokenization import whitespace_tokenize, BasicTokenizer, BertTokenizer
from pytorch_pretrained_bert.modeling import BertForQuestionAnswering
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.optimization import BertAdam
#from model import BertForMultiHopQuestionAnswering, CognitiveGNN
from transform2_model import BertForMultiHopQuestionAnswering, CognitiveGNN, Transformer
from transpytorch.transformer.Optim import ScheduledOptim
import torch.optim as optim
from utils import warmup_linear, find_start_end_after_tokenized, find_start_end_before_tokenized, bundle_part_to_batch, judge_question_type, fuzzy_retrieve, WindowMean, fuzz
from data import convert_question_to_samples_bundle, homebrew_data_loader
from pytorch_transformers import BertTokenizer,BertModel
from pytorch_pretrained_bert.modeling import (
    BertPreTrainedModel as PreTrainedBertModel, # The name was changed in the new versions of pytorch_pretrained_bert
    BertModel,
    BertLayerNorm,
    gelu,
    BertEncoder,
    BertPooler,

)
from pytorch_pretrained_bert.tokenization import (
    whitespace_tokenize,
    BasicTokenizer,
    BertTokenizer,
)
from transformers import AlbertModel, AlbertTokenizer, RobertaModel, RobertaConfig, RobertaTokenizer, XLNetTokenizer, XLNetModel

import argparse
from os.path import join
from tqdm import tqdm
from pytorch_pretrained_bert.modeling import BertModel
from DFGN.model.GFN import *
#from DFGN.utils import *
from DFGN.tools.data_iterator_pack import IGNORE_INDEX
import numpy as np
import queue
import threading
import time
import random
from DFGN.config import set_config
from DFGN.tools.data_helper import DataHelper
#from DFGN.text_to_tok_pack import *

BUF_SIZE = 5
data_queue = queue.Queue(BUF_SIZE)

def large_batch_encode(bert_model, batch, encoder_gpus, max_bert_bsz):
    doc_ids, doc_mask, segment_ids = batch['context_idxs'], batch['context_mask'], batch['segment_idxs']
    N = doc_ids.shape[0]

    doc_ids = doc_ids.cuda(encoder_gpus[0])
    doc_mask = doc_mask.cuda(encoder_gpus[0])
    segment_ids = segment_ids.cuda(encoder_gpus[0])
    doc_encoding = []

    ptr = 0
    while ptr < N:
        all_doc_encoder_layers = bert_model(input_ids=doc_ids[ptr:ptr+max_bert_bsz],
                                            token_type_ids=segment_ids[ptr:ptr+max_bert_bsz],
                                            attention_mask=doc_mask[ptr:ptr+max_bert_bsz],
                                            output_all_encoded_layers=False)
        tem_doc_encoding = all_doc_encoder_layers.detach()
        doc_encoding.append(tem_doc_encoding)
        ptr += max_bert_bsz
        del all_doc_encoder_layers

    doc_encoding = torch.cat(doc_encoding, dim=0)
    return doc_encoding


def dispatch(context_encoding, context_mask, batch, device):
    batch['context_encoding'] = context_encoding.cuda(device)
    batch['context_mask'] = context_mask.float().cuda(device)
    return batch


def cal_performance(pred, gold, trg_pad_idx, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, trg_pad_idx, smoothing=smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(trg_pad_idx)
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()

    return loss, n_correct, n_word


def cal_loss(pred, gold, trg_pad_idx, smoothing=False):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(trg_pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction='sum')
    return loss


def patch_src(src, pad_idx):
    src = src.transpose(0, 1)
    return src


def patch_trg(trg, pad_idx):
    trg = trg.transpose(0, 1)
    trg, gold = trg[:, :-1], trg[:, 1:].contiguous().view(-1)
    return trg, gold

def train_epoch(model, training_data, optimizer, opt, device, smoothing):
    ''' Epoch operation in training phase'''

    model.train()
    total_loss, n_word_total, n_word_correct = 0, 0, 0

    desc = '  - (Training)   '
    for batch in tqdm(training_data, mininterval=2, desc=desc, leave=False):

        # prepare data
        src_seq = patch_src(batch.src, opt.src_pad_idx).to(device)
        trg_seq, gold = map(lambda x: x.to(device), patch_trg(batch.trg, opt.trg_pad_idx))

        # forward
        optimizer.zero_grad()
        pred = model(src_seq, trg_seq)

        # backward and update parameters
        loss, n_correct, n_word = cal_performance(
            pred, gold, opt.trg_pad_idx, smoothing=smoothing)
        loss.backward()
        optimizer.step_and_update_lr()

        # note keeping
        n_word_total += n_word
        n_word_correct += n_correct
        total_loss += loss.item()

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy


def eval_epoch(model, validation_data, device, opt):
    ''' Epoch operation in evaluation phase '''

    model.eval()
    total_loss, n_word_total, n_word_correct = 0, 0, 0

    desc = '  - (Validation) '
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2, desc=desc, leave=False):

            # prepare data
            src_seq = patch_src(batch.src, opt.src_pad_idx).to(device)
            trg_seq, gold = map(lambda x: x.to(device), patch_trg(batch.trg, opt.trg_pad_idx))

            # forward
            pred = model(src_seq, trg_seq)
            loss, n_correct, n_word = cal_performance(
                pred, gold, opt.trg_pad_idx, smoothing=False)

            # note keeping
            n_word_total += n_word
            n_word_correct += n_correct
            total_loss += loss.item()

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy


def train(bundles, model1, device, mode, model2, batch_size, num_epoch, gradient_accumulation_steps, lr1, lr2, alpha, model3, training_data, validation_data, optimizer, trans_device, opt):
    '''Train Sys1 and Sys2 models.
    
    Train models by task #1(tensors) and task #2(bundle). 
    
    Args:
        bundles (list): List of bundles.
        model1 (BertForMultiHopQuestionAnswering): System 1 model.
        device (torch.device): The device which models and data are on.
        mode (str): Defaults to 'tensors'. Task identifier('tensors' or 'bundle').
        model2 (CognitiveGNN): System 2 model.
        batch_size (int): Defaults to 4.
        num_epoch (int): Defaults to 1.
        gradient_accumulation_steps (int): Defaults to 1. 
        lr1 (float): Defaults to 1e-4. Learning rate for Sys1.
        lr2 (float): Defaults to 1e-4. Learning rate for Sys2.
        alpha (float): Defaults to 0.2. Balance factor for loss of two systems.
    
    Returns:
        ([type], [type]): Trained models.
    '''

    # Prepare optimizer for Sys1
    param_optimizer = list(model1.named_parameters())
    # hack to remove pooler, which is not used.

    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    num_batch, dataloader = homebrew_data_loader(bundles, mode = mode, batch_size=batch_size)
    num_steps = num_batch * num_epoch
    global_step = 0
    opt1 = BertAdam(optimizer_grouped_parameters, lr = lr1, warmup = 0.1, t_total=num_steps)
    model1.to(device)
    model1.train()

    # Prepare optimizer for Sys2
    if mode == 'bundle':
        opt2 = Adam(model2.parameters(), lr=lr2)
        model2.to(device)
        model2.train()
        warmed = False # warmup for jointly training

    for epoch in trange(num_epoch, desc = 'Epoch'):
        ans_mean, hop_mean = WindowMean(), WindowMean()
        opt1.zero_grad()
        if mode == 'bundle':
            final_mean = WindowMean()
            opt2.zero_grad()
        tqdm_obj = tqdm(dataloader, total = num_batch)

        for step, batch in enumerate(tqdm_obj):
            try:
                if mode == 'tensors':
                    batch = tuple(t.to(device) for t in batch)
                    hop_loss, ans_loss, pooled_output = model1(*batch)
                    hop_loss, ans_loss = hop_loss.mean(), ans_loss.mean()
                    pooled_output.detach()
                    loss = ans_loss + hop_loss
                elif mode == 'bundle':
                    hop_loss, ans_loss, final_loss = model2(batch, model1, device)
                    hop_loss, ans_loss = hop_loss.mean(), ans_loss.mean()
                    loss = ans_loss + hop_loss + alpha * final_loss
                loss.backward()

                if (step + 1) % gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses. From BERT pytorch examples
                    lr_this_step = lr1 * warmup_linear(global_step/num_steps, warmup = 0.1)
                    for param_group in opt1.param_groups:
                        param_group['lr'] = lr_this_step
                    global_step += 1
                    if mode == 'bundle':
                        opt2.step()
                        opt2.zero_grad()
                        final_mean_loss = final_mean.update(final_loss.item())
                        tqdm_obj.set_description('ans_loss: {:.2f}, hop_loss: {:.2f}, final_loss: {:.2f}'.format(
                            ans_mean.update(ans_loss.item()), hop_mean.update(hop_loss.item()), final_mean_loss))
                        # During warming period, model1 is frozen and model2 is trained to normal weights
                        if final_mean_loss < 0.9 and step > 100: # ugly manual hyperparam
                            warmed = True
                        if warmed:
                            opt1.step()
                        opt1.zero_grad()
                    else:
                        opt1.step()
                        opt1.zero_grad()
                        tqdm_obj.set_description('ans_loss: {:.2f}, hop_loss: {:.2f}'.format(
                            ans_mean.update(ans_loss.item()), hop_mean.update(hop_loss.item())))
                    if step % 1000 == 0:
                        output_model_file = './models/bert-base-uncased.bin.tmp'
                        saved_dict = {'params1' : model1.module.state_dict()}
                        saved_dict['params2'] = model2.state_dict()
                        torch.save(saved_dict, output_model_file)
            except Exception as err:
                traceback.print_exc()
                if mode == 'bundle':
                    print(batch._id)
    #train  transform
    if mode == 'trans':
        if opt.use_tb:
            print("[Info] Use Tensorboard")
            from torch.utils.tensorboard import SummaryWriter
            tb_writer = SummaryWriter(log_dir=os.path.join(opt.output_dir, 'tensorboard'))

        log_train_file = os.path.join(opt.output_dir, 'train.log')
        log_valid_file = os.path.join(opt.output_dir, 'valid.log')

        print('[Info] Training performance will be written to file: {} and {}'.format(
            log_train_file, log_valid_file))

        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            log_tf.write('epoch,loss,ppl,accuracy\n')
            log_vf.write('epoch,loss,ppl,accuracy\n')

        def print_performances(header, ppl, accu, start_time, lr):
            print('  - {header:12} ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, lr: {lr:8.5f}, ' \
                  'elapse: {elapse:3.3f} min'.format(
                header=f"({header})", ppl=ppl,
                accu=100 * accu, elapse=(time.time() - start_time) / 60, lr=lr))

        # valid_accus = []
        valid_losses = []
        for epoch_i in range(opt.epoch):
            print('[ Epoch', epoch_i, ']')

            start = time.time()
            train_loss, train_accu = train_epoch(
                model3, training_data, optimizer, opt, device, smoothing=opt.label_smoothing)
            train_ppl = math.exp(min(train_loss, 100))
            # Current learning rate
            lr = optimizer._optimizer.param_groups[0]['lr']
            print_performances('Training', train_ppl, train_accu, start, lr)

            start = time.time()
            valid_loss, valid_accu = eval_epoch(model3, validation_data, device, opt)
            valid_ppl = math.exp(min(valid_loss, 100))
            print_performances('Validation', valid_ppl, valid_accu, start, lr)

            valid_losses += [valid_loss]

            checkpoint = {'epoch': epoch_i, 'settings': opt, 'model': model3.state_dict()}

            if opt.save_mode == 'all':
                model_name = 'model_accu_{accu:3.3f}.chkpt'.format(accu=100 * valid_accu)
                torch.save(checkpoint, model_name)
            elif opt.save_mode == 'best':
                model_name = 'model.chkpt'
                if valid_loss <= min(valid_losses):
                    torch.save(checkpoint, os.path.join(opt.output_dir, model_name))
                    print('    - [Info] The checkpoint file has been updated.')

            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=train_loss,
                    ppl=train_ppl, accu=100 * train_accu))
                log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=valid_loss,
                    ppl=valid_ppl, accu=100 * valid_accu))

            if opt.use_tb:
                tb_writer.add_scalars('ppl', {'train': train_ppl, 'val': valid_ppl}, epoch_i)
                tb_writer.add_scalars('accuracy', {'train': train_accu * 100, 'val': valid_accu * 100}, epoch_i)
                tb_writer.add_scalar('learning_rate', lr, epoch_i)
    return (model1, model3, model2)


def main(output_model_file = './models/pytorch_model.bin', output_dir = './models/trans.bin' , load = 3, mode = 'tensors', batch_size = 1,
            num_epoch = 3, gradient_accumulation_steps = 1, lr1 = 1e-5, lr2 = 1e-5, alpha = 0.2):
    
    # BERT_MODEL = 'bert-base-uncased' # bert-large is too large for ordinary GPU on task #2
    # tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, do_lower_case=True)

    # BERT_MODEL = BertModel.from_pretrained('bert-base-uncased')
    # print(BERT_MODEL)
    # tokenizer = BertTokenizer.from_pretrained("./albert_base")
    # BERT_MODEL = BertModel.from_pretrained("./albert_base")

    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2', do_lower_case=True)
    BERT_MODEL = 'albert-base-v2'

    # tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)
    # #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    # ROBERTA_MODEL = 'roberta-base'

    # tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True)
    # BERT_MODEL = 'xlnet-base-cased'

    # BERT_MODEL = '/home/shaoai/CogQA/uncased_L-2_H-128_A-2'
    # print(BERT_MODEL)
    # tokenizer = BertTokenizer.from_pretrained('/home/shaoai/CogQA/uncased_L-2_H-128_A-2')

    with open('./hotpot_train_v1.1_refined.json' ,'r') as fin:
        dataset = json.load(fin)
    bundles = []

    # for data in tqdm(dataset):
    #     try:
    #         bundles.append(convert_question_to_samples_bundle(tokenizer, data))
    #     except ValueError as err:
    #         pass
    #     except Exception as err:
    #         traceback.print_exc()
    #         pass


    data_example={
        "supporting_facts": [
            [
                "Arthur's Magazine",
                0,
                []
            ],
            [
                "First for Women",
                0,
                []
            ]
        ],
        "level": "medium",
        "question": "Which magazine was started first Arthur's Magazine or First for Women?",
        "context": [
            [
                "Radio City (Indian radio station)",
                [
                    "Radio City is India's first private FM radio station and was started on 3 July 2001.",
                    " It broadcasts on 91.1 (earlier 91.0 in most cities) megahertz from Mumbai (where it was started in 2004), Bengaluru (started first in 2001), Lucknow and New Delhi (since 2003).",
                    " It plays Hindi, English and regional songs.",
                    " It was launched in Hyderabad in March 2006, in Chennai on 7 July 2006 and in Visakhapatnam October 2007.",
                    " Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features.",
                    " The Radio station currently plays a mix of Hindi and Regional music.",
                    " Abraham Thomas is the CEO of the company."
                ]
            ],
            [
                "History of Albanian football",
                [
                    "Football in Albania existed before the Albanian Football Federation (FSHF) was created.",
                    " This was evidenced by the team's registration at the Balkan Cup tournament during 1929-1931, which started in 1929 (although Albania eventually had pressure from the teams because of competition, competition started first and was strong enough in the duels) .",
                    " Albanian National Team was founded on June 6, 1930, but Albania had to wait 16 years to play its first international match and then defeated Yugoslavia in 1946.",
                    " In 1932, Albania joined FIFA (during the 12\u201316 June convention ) And in 1954 she was one of the founding members of UEFA."
                ]
            ],
            [
                "Echosmith",
                [
                    "Echosmith is an American, Corporate indie pop band formed in February 2009 in Chino, California.",
                    " Originally formed as a quartet of siblings, the band currently consists of Sydney, Noah and Graham Sierota, following the departure of eldest sibling Jamie in late 2016.",
                    " Echosmith started first as \"Ready Set Go!\"",
                    " until they signed to Warner Bros.",
                    " Records in May 2012.",
                    " They are best known for their hit song \"Cool Kids\", which reached number 13 on the \"Billboard\" Hot 100 and was certified double platinum by the RIAA with over 1,200,000 sales in the United States and also double platinum by ARIA in Australia.",
                    " The song was Warner Bros.",
                    " Records' fifth-biggest-selling-digital song of 2014, with 1.3 million downloads sold.",
                    " The band's debut album, \"Talking Dreams\", was released on October 8, 2013."
                ]
            ],
            [
                "Women's colleges in the Southern United States",
                [
                    "Women's colleges in the Southern United States refers to undergraduate, bachelor's degree\u2013granting institutions, often liberal arts colleges, whose student populations consist exclusively or almost exclusively of women, located in the Southern United States.",
                    " Many started first as girls' seminaries or academies.",
                    " Salem College is the oldest female educational institution in the South and Wesleyan College is the first that was established specifically as a college for women.",
                    " Some schools, such as Mary Baldwin University and Salem College, offer coeducational courses at the graduate level."
                ]
            ],
            [
                "First Arthur County Courthouse and Jail",
                [
                    "The First Arthur County Courthouse and Jail, was perhaps the smallest court house in the United States, and serves now as a museum."
                ]
            ],
            [
                "Arthur's Magazine",
                [
                    "Arthur's Magazine (1844\u20131846) was an American literary periodical published in Philadelphia in the 19th century.",
                    " Edited by T.S. Arthur, it featured work by Edgar A. Poe, J.H. Ingraham, Sarah Josepha Hale, Thomas G. Spear, and others.",
                    " In May 1846 it was merged into \"Godey's Lady's Book\"."
                ]
            ],
            [
                "2014\u201315 Ukrainian Hockey Championship",
                [
                    "The 2014\u201315 Ukrainian Hockey Championship was the 23rd season of the Ukrainian Hockey Championship.",
                    " Only four teams participated in the league this season, because of the instability in Ukraine and that most of the clubs had economical issues.",
                    " Generals Kiev was the only team that participated in the league the previous season, and the season started first after the year-end of 2014.",
                    " The regular season included just 12 rounds, where all the teams went to the semifinals.",
                    " In the final, ATEK Kiev defeated the regular season winner HK Kremenchuk."
                ]
            ],
            [
                "First for Women",
                [
                    "First for Women is a woman's magazine published by Bauer Media Group in the USA.",
                    " The magazine was started in 1989.",
                    " It is based in Englewood Cliffs, New Jersey.",
                    " In 2011 the circulation of the magazine was 1,310,696 copies."
                ]
            ],
            [
                "Freeway Complex Fire",
                [
                    "The Freeway Complex Fire was a 2008 wildfire in the Santa Ana Canyon area of Orange County, California.",
                    " The fire started as two separate fires on November 15, 2008.",
                    " The \"Freeway Fire\" started first shortly after 9am with the \"Landfill Fire\" igniting approximately 2 hours later.",
                    " These two separate fires merged a day later and ultimately destroyed 314 residences in Anaheim Hills and Yorba Linda."
                ]
            ],
            [
                "William Rast",
                [
                    "William Rast is an American clothing line founded by Justin Timberlake and Trace Ayala.",
                    " It is most known for their premium jeans.",
                    " On October 17, 2006, Justin Timberlake and Trace Ayala put on their first fashion show to launch their new William Rast clothing line.",
                    " The label also produces other clothing items such as jackets and tops.",
                    " The company started first as a denim line, later evolving into a men\u2019s and women\u2019s clothing line."
                ]
            ]
        ],
        "answer": "Arthur's Magazine",
        "_id": "5a7a06935542990198eaf050",
        "type": "comparison",
        "Q_edge": [
            [
                "Arthur's Magazine",
                "Arthur's Magazine",
                33,
                50
            ],
            [
                "First for Women",
                "First for Women",
                54,
                69
            ]
        ]
    }
    bundles.append(convert_question_to_samples_bundle(tokenizer, data_example))


    device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_pkl', default='/home/shao/CogQA159/m30k_deen_shr.pkl')  # all-in-1 data pickle or bpe field

    parser.add_argument('-train_path', default=None)  # bpe encoded data
    parser.add_argument('-val_path', default=None)  # bpe encoded data

    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-b', '--batch_size', type=int, default=2048)

    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=2048)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)

    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-warmup', '--n_warmup_steps', type=int, default=4000)
    parser.add_argument('-lr_mul', type=float, default=2.0)
    parser.add_argument('-seed', type=int, default=None)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')
    parser.add_argument('-scale_emb_or_prj', type=str, default='prj')

    parser.add_argument('-output_dir', type=str, default='/home/shao/CogQA159/output')
    parser.add_argument('-use_tb', action='store_true')
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-label_smoothing', action='store_true')
    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_model

    # https://pytorch.org/docs/stable/notes/randomness.html
    # For reproducibility
    if opt.seed is not None:
        torch.manual_seed(opt.seed)
        torch.backends.cudnn.benchmark = False
        # torch.set_deterministic(True)
        np.random.seed(opt.seed)
        random.seed(opt.seed)

    if not opt.output_dir:
        print('No experiment result will be saved.')
        raise

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

    if opt.batch_size < 2048 and opt.n_warmup_steps <= 4000:
        print('[Warning] The warmup steps may be not enough.\n' \
              '(sz_b, warmup) = (2048, 4000) is the official setting.\n' \
              'Using smaller batch w/o longer warmup may cause ' \
              'the warmup stage ends with only little data trained.')

    trans_device = torch.device('cuda' if opt.cuda else 'cpu')

    # ========= Loading Dataset =========#

    if all((opt.train_path, opt.val_path)):
        training_data, validation_data = prepare_dataloaders_from_bpe_files(opt, device)
    elif opt.data_pkl:
        training_data, validation_data = prepare_dataloaders(opt, device)
    else:
        raise

    if load == 2:
        print('Loading model from {}'.format(output_model_file))
        model_state_dict = torch.load(output_model_file)
        model1 = BertForMultiHopQuestionAnswering.from_pretrained(BERT_MODEL, state_dict=model_state_dict['params1'])
        #model1 = RobertaForMultiHopQuestionAnswering.from_pretrained(ROBERTA_MODEL, cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(-1))
        model2 = Transformer(opt.src_vocab_size,
                             opt.trg_vocab_size,
                             src_pad_idx=opt.src_pad_idx,
                             trg_pad_idx=opt.trg_pad_idx,
                             trg_emb_prj_weight_sharing=opt.proj_share_weight,
                            emb_src_trg_weight_sharing=opt.embs_share_weight,
                            d_k=opt.d_k,
                            d_v=opt.d_v,
                            d_model=opt.d_model,
                            d_word_vec=opt.d_word_vec,
                            d_inner=opt.d_inner_hid,
                            n_layers=opt.n_layers,
                            n_head=opt.n_head,
                            dropout=opt.dropout,
                            scale_emb_or_prj=opt.scale_emb_or_prj).to(device)


        model2.load_state_dict(model_state_dict['params2'])
        parser = argparse.ArgumentParser()
        args = set_config()
        helper = DataHelper(gz=True, config=args)
        args.n_type = helper.n_type
        model3 = GraphFusionNet(config=args)

    if load == 1:
        model1 = BertForMultiHopQuestionAnswering.from_pretrained(BERT_MODEL, cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(-1))
        #model1 = RobertaForMultiHopQuestionAnswering.from_pretrained(ROBERTA_MODEL, cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(-1))
        model2 = Transformer(opt.src_vocab_size,
                             opt.trg_vocab_size,
                             src_pad_idx=opt.src_pad_idx,
                             trg_pad_idx=opt.trg_pad_idx,
                             trg_emb_prj_weight_sharing=opt.proj_share_weight,
                             emb_src_trg_weight_sharing=opt.embs_share_weight,
                             d_k=opt.d_k,
                             d_v=opt.d_v,
                             d_model=opt.d_model,
                             d_word_vec=opt.d_word_vec,
                             d_inner=opt.d_inner_hid,
                             n_layers=opt.n_layers,
                             n_head=opt.n_head,
                             dropout=opt.dropout,
                             scale_emb_or_prj=opt.scale_emb_or_prj).to(device)
        parser = argparse.ArgumentParser()
        args = set_config()
        helper = DataHelper(gz=True, config=args)
        args.n_type = helper.n_type
        model3 = GraphFusionNet(config=args)


    else:
        print('Loading model from {}'.format(output_model_file))
        model_state_dict = torch.load(output_model_file)
        model1 = BertForMultiHopQuestionAnswering.from_pretrained(BERT_MODEL,cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(-1))
        model2 = Transformer(opt.src_vocab_size,
                             opt.trg_vocab_size,
                             src_pad_idx=opt.src_pad_idx,
                             trg_pad_idx=opt.trg_pad_idx,
                             trg_emb_prj_weight_sharing=opt.proj_share_weight,
                             emb_src_trg_weight_sharing=opt.embs_share_weight,
                             d_k=opt.d_k,
                             d_v=opt.d_v,
                             d_model=opt.d_model,
                             d_word_vec=opt.d_word_vec,
                             d_inner=opt.d_inner_hid,
                             n_layers=opt.n_layers,
                             n_head=opt.n_head,
                             dropout=opt.dropout,
                             scale_emb_or_prj=opt.scale_emb_or_prj).to(device)
        model2.load_state_dict(model_state_dict['params2'])

        parser = argparse.ArgumentParser()
        args = set_config()
        helper = DataHelper(gz=True, config=args)
        args.n_type = helper.n_type
        model3 = GraphFusionNet(config=args)
        model3.load_state_dict(model_state_dict['params3'])

    optimizer = ScheduledOptim(
        optim.Adam(model2.parameters(), betas=(0.9, 0.98), eps=1e-09),
        opt.lr_mul, opt.d_model, opt.n_warmup_steps)
    print('Start Training... on {} GPUs'.format(torch.cuda.device_count()))
    model1 = torch.nn.DataParallel(model1, device_ids = range(torch.cuda.device_count()))

    model1, model2, model3 = train(bundles, model1=model1, device=device, mode=mode, model2=model3, # Then pass hyperparams
        batch_size=batch_size, num_epoch=num_epoch, gradient_accumulation_steps=gradient_accumulation_steps,lr1=lr1, lr2=lr2, alpha=alpha,
        model3=model2, training_data=training_data, validation_data=validation_data, optimizer=optimizer, trans_device=trans_device, opt=opt)

    print('Saving model to {}'.format(output_model_file))
    saved_dict = {'params1' : model1.module.state_dict()}
    saved_dict['params2'] = model2.state_dict()
    saved_dict['params3'] = model3.state_dict()
    torch.save(saved_dict, output_model_file)



def prepare_dataloaders_from_bpe_files(opt, device):
    batch_size = opt.batch_size
    MIN_FREQ = 2
    if not opt.embs_share_weight:
        raise

    data = pickle.load(open(opt.data_pkl, 'rb'))
    MAX_LEN = data['settings'].max_len
    field = data['vocab']
    fields = (field, field)

    def filter_examples_with_length(x):
        return len(vars(x)['src']) <= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN

    train = TranslationDataset(
        fields=fields,
        path=opt.train_path,
        exts=('.src', '.trg'),
        filter_pred=filter_examples_with_length)
    val = TranslationDataset(
        fields=fields,
        path=opt.val_path,
        exts=('.src', '.trg'),
        filter_pred=filter_examples_with_length)

    opt.max_token_seq_len = MAX_LEN + 2
    opt.src_pad_idx = opt.trg_pad_idx = field.vocab.stoi[Constants.PAD_WORD]
    opt.src_vocab_size = opt.trg_vocab_size = len(field.vocab)

    train_iterator = BucketIterator(train, batch_size=batch_size, device=device, train=True)
    val_iterator = BucketIterator(val, batch_size=batch_size, device=device)
    return train_iterator, val_iterator


def prepare_dataloaders(opt, device):
    batch_size = opt.batch_size
    data = pickle.load(open(opt.data_pkl, 'rb'))

    opt.max_token_seq_len = data['settings'].max_len
    opt.src_pad_idx = data['vocab']['src'].vocab.stoi[Constants.PAD_WORD]
    opt.trg_pad_idx = data['vocab']['trg'].vocab.stoi[Constants.PAD_WORD]

    opt.src_vocab_size = len(data['vocab']['src'].vocab)
    opt.trg_vocab_size = len(data['vocab']['trg'].vocab)

    #========= Preparing Model =========#
    if opt.embs_share_weight:
        assert data['vocab']['src'].vocab.stoi == data['vocab']['trg'].vocab.stoi, \
            'To sharing word embedding the src/trg word2idx table shall be the same.'

    fields = {'src': data['vocab']['src'], 'trg':data['vocab']['trg']}

    train = Dataset(examples=data['train'], fields=fields)
    val = Dataset(examples=data['valid'], fields=fields)

    train_iterator = BucketIterator(train, batch_size=batch_size, device=device, train=True)
    val_iterator = BucketIterator(val, batch_size=batch_size, device=device)

    return train_iterator, val_iterator


import fire
if __name__ == "__main__":
    fire.Fire(main)
    parser = argparse.ArgumentParser()
    args = set_config()

    # Allocate Models on GPU
    encoder_gpus = [int(i) for i in args.encoder_gpu.split(',')]
    model_gpu = 'cuda:{}'.format(args.model_gpu)

    encoder = BertModel.from_pretrained(args.bert_model)

    encoder.eval()

    helper = DataHelper(gz=True, config=args)
    args.n_type = helper.n_type

    # Set datasets

    # Set Model
    model2 = GraphFusionNet(config=args)
    model2.cuda(model_gpu)
    model2.train()