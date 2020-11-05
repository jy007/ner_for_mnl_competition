import os
import sys
import logging
import torch
import random
import numpy as np
from pathlib import Path
import time
from transformers import BertTokenizer

logger = logging.getLogger()

def init_logger(log_file=None,log_file_level= logging.NOTSET):
    if isinstance(log_file,Path):
        log_file = str(logging)
    log_format = logging.Formatter(fmt="%(asctime)s-%(levelname)s-%(name)s- %(message)s",
                                    datefmt="%m/%d/%Y %H:%M:%S"
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler  = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]
    if log_file and log_file != "":
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        logger.addHandler(file_handler)

    return logger


def seed_everyting(seed=2020):
    """
    设置整个开发环境seed
    
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True


def collate_fn(batch):
    """
    batch should be a list of (seq,target,length) tuples
    returns a padded tensor of sequences sorted from longest to shortest.
    """
    all_input_ids,all_attention_mask,all_token_type_ids,all_lens,all_labels = map(torch.stack,zip(*batch))
    max_len  = max(all_lens).item()
    all_input_ids = all_input_ids[:,:max_len]
    all_attention_mask = all_attention_mask[:,:max_len]
    all_token_type_ids = all_token_type_ids[:,:max_len]
    all_labels = all_labels[:,:max_len]

    return all_input_ids,all_attention_mask,all_token_type_ids,all_labels,all_lens



class ProgressBar(object):
    '''
    custom progress bar
    Example:
        >>> pbar = ProgressBar(n_total=30,desc='training')
        >>> step = 2
        >>> pbar(step=step)
    '''
    def __init__(self, n_total,width=30,desc='Training'):
        self.width = width
        self.n_total = n_total
        self.start_time = time.time()
        self.desc = desc

    def __call__(self, step, info={}):
        now = time.time()
        current = step + 1
        recv_per = current / self.n_total
        bar = f'[{self.desc}] {current}/{self.n_total} ['
        if recv_per >= 1:
            recv_per = 1
        prog_width = int(self.width * recv_per)
        if prog_width > 0:
            bar += '=' * (prog_width - 1)
            if current< self.n_total:
                bar += ">"
            else:
                bar += '='
        bar += '.' * (self.width - prog_width)
        bar += ']'
        show_bar = f"\r{bar}"
        time_per_unit = (now - self.start_time) / current
        if current < self.n_total:
            eta = time_per_unit * (self.n_total - current)
            if eta > 3600:
                eta_format = ('%d:%02d:%02d' %
                              (eta // 3600, (eta % 3600) // 60, eta % 60))
            elif eta > 60:
                eta_format = '%d:%02d' % (eta // 60, eta % 60)
            else:
                eta_format = '%ds' % eta
            time_info = f' - ETA: {eta_format}'
        else:
            if time_per_unit >= 1:
                time_info = f' {time_per_unit:.1f}s/step'
            elif time_per_unit >= 1e-3:
                time_info = f' {time_per_unit * 1e3:.1f}ms/step'
            else:
                time_info = f' {time_per_unit * 1e6:.1f}us/step'

        show_bar += time_info
        if len(info) != 0:
            show_info = f'{show_bar} ' + \
                        "-".join([f' {key}: {value:.4f} ' for key, value in info.items()])
            print(show_info, end='')
        else:
            print(show_bar, end='')


class CNerTokenizer(BertTokenizer):
    def __init__(self, vocab_file, do_lower_case=False):
        super().__init__(vocab_file=str(vocab_file), do_lower_case=do_lower_case)
        self.vocab_file = str(vocab_file)
        self.do_lower_case = do_lower_case

    def tokenize(self, text):
        _tokens = []
        for c in text:
            if self.do_lower_case:
                c = c.lower()
            if c in self.vocab:
                _tokens.append(c)
            else:
                _tokens.append('[UNK]')
        return _tokens
