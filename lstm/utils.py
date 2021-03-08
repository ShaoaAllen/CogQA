import sys
import re
from time import time
from os.path import isfile
from lstm.parameters import *
from collections import defaultdict

def normalize(x):
    # x = re.sub("[\uAC00-\uD7A3]+", "\uAC00", x) £ convert Hangeul to 가
    # x = re.sub("[\u3040-\u30FF]+", "\u3042", x) # convert Hiragana and Katakana to あ
    # x = re.sub("[\u4E00-\u9FFF]+", "\u6F22", x) # convert CJK unified ideographs to 漢
    x = re.sub("\s+", " ", x)
    x = re.sub("^ | $", "", x)
    x = x.lower()
    return x

def tokenize(x):
    if UNIT == "char":
        return list(re.sub(" ", "", x))
    if UNIT == "char+space":
        return [x.replace("_", "__").replace(" ", "_") for x in x]
    if UNIT in ("word", "sent"):
        return x.split(" ")

def save_data(filename, data):
    fo = open(filename, "w")
    for seq in data:
        fo.write((" ".join(seq[0]) + "\t" + " ".join(seq[1]) if seq else "") + "\n")
    fo.close()

def load_tkn_to_idx(filename):
    print("loading %s" % filename)
    tkn_to_idx = {}
    fo = open(filename)
    for line in fo:
        line = line[:-1]
        tkn_to_idx[line] = len(tkn_to_idx)
    fo.close()
    return tkn_to_idx

def load_idx_to_tkn(filename):
    print("loading %s" % filename)
    idx_to_tkn = []
    fo = open(filename)
    for line in fo:
        line = line[:-1]
        idx_to_tkn.append(line)
    fo.close()
    return idx_to_tkn

def save_tkn_to_idx(filename, tkn_to_idx):
    fo = open(filename, "w")
    for tkn, _ in sorted(tkn_to_idx.items(), key = lambda x: x[1]):
        fo.write("%s\n" % tkn)
    fo.close()

def load_checkpoint(filename, model = None):
    print("loading %s" % filename)
    checkpoint = torch.load(filename)
    if model:
        model.load_state_dict(checkpoint["state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    print("epoch = %d, loss = %f" % (checkpoint["epoch"], checkpoint["loss"]))
    return epoch

def save_checkpoint(filename, model, epoch, loss, time):
    print("epoch = %d, loss = %f, time = %f" % (epoch, loss, time))
    if filename and model:
        checkpoint = {}
        checkpoint["state_dict"] = model.state_dict()
        checkpoint["epoch"] = epoch
        checkpoint["loss"] = loss
        torch.save(checkpoint, filename + ".epoch%d" % epoch)
        print("saved %s" % filename)

def log_sum_exp(x):
    m = torch.max(x, -1)[0]
    return m + torch.log(torch.sum(torch.exp(x - m.unsqueeze(-1)), -1))

def tag_to_txt(xs, ys):
    _xs, _ys = [], []
    for x, y in zip(xs, ys):
        if UNIT == "char+space":
            if x == "_":
                y = "_"
            x = x.replace("__", "_")
        if len(_xs) and y in ("I", "E", "I-" + _ys[-1], "E-" + _ys[-1]):
            _xs[-1] += x
            continue
        if y[:2] in ("B-", "I-", "E-", "S-"):
            y = y[2:]
        _xs.append(x)
        _ys.append(y)
    if TASK == "pos-tagging":
        return " ".join(x + "/" + y for x, y in zip(_xs, _ys))
    if TASK == "word-segmentation":
        return " ".join("".join(x) for x in _xs)
    if TASK == "sentence-segmentation":
        return "\n".join(" ".join(x) for x in _xs)

def f1(p, r):
    return 2 * p * r / (p + r) if p + r else 0
