from src.input_reader import Reader
from transformers import AutoTokenizer
from pprint import pprint
import src.utils as utils
import transformers
import torch


if __name__ == '__main__':
    print(torch.nn.Parameter(torch.Tensor(1, 2, 3)))
    # print(utils.get_device())
    # tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base', use_fast=False)
    # reader = Reader(tokenizer, 'data/VLSP_2016/vocab.json')
    # demo_dataset = reader.read('data/VLSP_2016/demo.json', 'VLSP_2016-demo')
    # pprint(demo_dataset[0])