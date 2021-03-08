import json
from tqdm import tqdm, trange
import torch
import traceback
from torch.optim import Adam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.optimization import BertAdam
from model import BertForMultiHopQuestionAnswering, CognitiveGNN
from run_fail_store.roberta_model import CognitiveGNN
from utils import warmup_linear, WindowMean
from data import convert_question_to_samples_bundle, homebrew_data_loader
from transformers import AlbertTokenizer

import math
import logging
import time
import sys
import random
import argparse
import pickle
from pathlib import Path

import torch
import numpy as np

from tgn.model.tgn import TGN
from tgn.utils.utils import EarlyStopMonitor, get_neighbor_finder, MLP
from tgn.utils.data_processing import compute_time_statistics, get_data_node_classification
from tgn.evaluation.evaluation import eval_node_classification


def train(bundles, model1, device, mode, model2, batch_size, num_epoch, gradient_accumulation_steps, lr1, lr2, alpha):
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
    return (model1, model2)


def main(output_model_file = './models/pytorch_model.bin', load = True, mode = 'tensors', batch_size = 7,
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
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    ### Argument and global variables
    parser = argparse.ArgumentParser('TGN self-supervised training')
    parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                        default='wikipedia')
    parser.add_argument('--bs', type=int, default=100, help='Batch_size')
    parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
    parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
    parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
    parser.add_argument('--n_epoch', type=int, default=10, help='Number of epochs')
    parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
    parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
    parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
    parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
    parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
    parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to '
                                                                      'backprop')
    parser.add_argument('--use_memory', action='store_true',
                        help='Whether to augment the model with a node memory')
    parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=[
        "graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
    parser.add_argument('--message_function', type=str, default="identity", choices=[
        "mlp", "identity"], help='Type of message function')
    parser.add_argument('--aggregator', type=str, default="last", help='Type of message '
                                                                       'aggregator')
    parser.add_argument('--memory_update_at_end', action='store_true',
                        help='Whether to update memory at the end or at the start of the batch')
    parser.add_argument('--message_dim', type=int, default=100, help='Dimensions of the messages')
    parser.add_argument('--memory_dim', type=int, default=172, help='Dimensions of the memory for '
                                                                    'each user')
    parser.add_argument('--different_new_nodes', action='store_true',
                        help='Whether to use disjoint set of new nodes for train and val')
    parser.add_argument('--uniform', action='store_true',
                        help='take uniform sampling from temporal neighbors')
    parser.add_argument('--randomize_features', action='store_true',
                        help='Whether to randomize node features')
    parser.add_argument('--use_destination_embedding_in_message', action='store_true',
                        help='Whether to use the embedding of the destination node as part of the message')
    parser.add_argument('--use_source_embedding_in_message', action='store_true',
                        help='Whether to use the embedding of the source node as part of the message')
    parser.add_argument('--n_neg', type=int, default=1)
    parser.add_argument('--use_validation', action='store_true',
                        help='Whether to use a validation set')
    parser.add_argument('--new_node', action='store_true', help='model new node')

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    BATCH_SIZE = args.bs
    NUM_NEIGHBORS = args.n_degree
    NUM_NEG = 1
    NUM_EPOCH = args.n_epoch
    NUM_HEADS = args.n_head
    DROP_OUT = args.drop_out
    GPU = args.gpu
    UNIFORM = args.uniform
    NEW_NODE = args.new_node
    SEQ_LEN = NUM_NEIGHBORS
    DATA = args.data
    NUM_LAYER = args.n_layer
    LEARNING_RATE = args.lr
    NODE_LAYER = 1
    NODE_DIM = args.node_dim
    TIME_DIM = args.time_dim
    USE_MEMORY = args.use_memory
    MESSAGE_DIM = args.message_dim
    MEMORY_DIM = args.memory_dim

    Path("./saved_models/").mkdir(parents=True, exist_ok=True)
    Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)
    MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.data}' + '\
      node-classification.pth'
    get_checkpoint_path = lambda \
            epoch: f'./saved_checkpoints/{args.prefix}-{args.data}-{epoch}' + '\
      node-classification.pth'

    ### set up logger
    # logging.basicConfig(level=logging.INFO)
    # logger = logging.getLogger()
    # logger.setLevel(logging.DEBUG)
    # fh = logging.FileHandler('log/{}.log'.format(str(time.time())))
    # fh.setLevel(logging.DEBUG)
    # ch = logging.StreamHandler()
    # ch.setLevel(logging.WARN)
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # fh.setFormatter(formatter)
    # ch.setFormatter(formatter)
    # logger.addHandler(fh)
    # logger.addHandler(ch)
    # logger.info(args)

    full_data, node_features, edge_features, train_data, val_data, test_data = \
        get_data_node_classification(DATA, use_validation=args.use_validation)

    max_idx = max(full_data.unique_nodes)

    train_ngh_finder = get_neighbor_finder(train_data, uniform=UNIFORM, max_node_idx=max_idx)

    # Set device
    device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_string)

    # Compute time statistics
    mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
        compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)

    for i in range(args.n_runs):
        results_path = "results/{}_node_classification_{}.pkl".format(args.prefix,
                                                                      i) if i > 0 else "results/{}_node_classification.pkl".format(
            args.prefix)
        Path("results/").mkdir(parents=True, exist_ok=True)


    if load:
        print('Loading model from {}'.format(output_model_file))
        model_state_dict = torch.load(output_model_file)
        model1 = BertForMultiHopQuestionAnswering.from_pretrained(BERT_MODEL, state_dict=model_state_dict['params1'])
        #model1 = RobertaForMultiHopQuestionAnswering.from_pretrained(ROBERTA_MODEL, cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(-1))
        model2 = TGN(neighbor_finder=train_ngh_finder, node_features=node_features,
                  edge_features=edge_features, device=device,
                  n_layers=NUM_LAYER,
                  n_heads=NUM_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,
                  message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
                  memory_update_at_start=not args.memory_update_at_end,
                  embedding_module_type=args.embedding_module,
                  message_function=args.message_function,
                  aggregator_type=args.aggregator, n_neighbors=NUM_NEIGHBORS,
                  mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
                  mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst,
                  use_destination_embedding_in_message=args.use_destination_embedding_in_message,
                  use_source_embedding_in_message=args.use_source_embedding_in_message)

        model2 = model2.to(device)

        model2.load_state_dict(model_state_dict['params2'])

    else:
        model1 = BertForMultiHopQuestionAnswering.from_pretrained(BERT_MODEL, cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(-1))
        #model1 = RobertaForMultiHopQuestionAnswering.from_pretrained(ROBERTA_MODEL, cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(-1))
        model2 = TGN(neighbor_finder=train_ngh_finder, node_features=node_features,
                  edge_features=edge_features, device=device,
                  n_layers=NUM_LAYER,
                  n_heads=NUM_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,
                  message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
                  memory_update_at_start=not args.memory_update_at_end,
                  embedding_module_type=args.embedding_module,
                  message_function=args.message_function,
                  aggregator_type=args.aggregator, n_neighbors=NUM_NEIGHBORS,
                  mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
                  mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst,
                  use_destination_embedding_in_message=args.use_destination_embedding_in_message,
                  use_source_embedding_in_message=args.use_source_embedding_in_message)


    print('Start Training... on {} GPUs'.format(torch.cuda.device_count()))
    model1 = torch.nn.DataParallel(model1, device_ids = range(torch.cuda.device_count()))
    model1, model2 = train(bundles, model1=model1, device=device, mode=mode, model2=model2, # Then pass hyperparams
        batch_size=batch_size, num_epoch=num_epoch, gradient_accumulation_steps=gradient_accumulation_steps,lr1=lr1, lr2=lr2, alpha=alpha)
    
    print('Saving model to {}'.format(output_model_file))
    saved_dict = {'params1' : model1.module.state_dict()}
    saved_dict['params2'] = model2.state_dict()
    torch.save(saved_dict, output_model_file)

import fire
if __name__ == "__main__":
    fire.Fire(main)