import argparse
import numpy as np
import os
import random
import subprocess
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
from transformers import BertModel, BertTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def format_time(start, end, total=None, finish=None):
    if total and finish:
        run_time = end - start
        total_time = run_time * total / finish
        todo_time = total_time - run_time
        run_h = run_time // 3600
        run_m = run_time % 3600 // 60
        run_s = run_time % 60
        todo_h = todo_time // 3600
        todo_m = todo_time % 3600 // 60
        todo_s = todo_time % 60
        ret = '%02d:%02d:%02d>%02d:%02d:%02d' % (todo_h, todo_m, todo_s, run_h, run_m, run_s)
    else:
        run_time = end - start
        run_h = run_time // 3600
        run_m = run_time % 3600 // 60
        run_s = run_time % 60
        ret = '%02d:%02d:%02d' % (run_h, run_m, run_s)
    return ret

