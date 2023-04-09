import string
import random
from dataclasses import dataclass, field
from argparse_dataclass import ArgumentParser
from pathlib import Path
import json


ALL_CHARACTERS = string.ascii_letters + string.digits
DATA_FOLDER = Path(__file__).parent / '../data/'
DATA_FOLDER.mkdir(exist_ok=True)

@dataclass
class Args:
    number_vocab: int
    seq_length: int
    num_examples: int = 2000
    num_test_examples: int = 600
    type: str = 'associative_recall'


def associative_recall(number_vocab, seq_length, num_examples, seed=8):
    random.seed(seed)
    keys = random.sample(ALL_CHARACTERS, number_vocab)
    values = random.sample(keys, number_vocab)

    num_steps = seq_length // 2
    pairs = list(zip(keys, values))
    dataset = []
    for _ in range(num_examples):
        datapoint = []
        for _ in range(num_steps):
            datapoint.extend(random.choice(pairs))
        
        # datapoint.append('<EOS>')
        dataset.append(datapoint)

    vocab = list(set(keys)) # ['<EOS>'] +

    return vocab, dataset


def majority(number_vocab, seq_length, num_examples, seed=8):
    random.seed(seed)
    vocab = random.sample(string.ascii_letters, number_vocab)
    dataset = []
    for _ in range(num_examples):
        datapoint = []
        for _ in range(seq_length):
            char = random.choice(vocab)
            datapoint.append(char)

        major_char = max(datapoint, key = datapoint.count)
        datapoint.extend(["=>", major_char, "<EOS>"])
        dataset.append(datapoint)

    vocab = set(vocab) | set(["=>"])
    return vocab, dataset


def counting(number_vocab, seq_length, num_examples, seed=8):
    random.seed(seed)
    vocab = random.sample(string.ascii_letters, number_vocab)
    dataset = []
    for _ in range(num_examples):
        datapoint = []
        for _ in range(seq_length):
            char = random.choice(vocab)
            datapoint.append(char)

        major_char = max(datapoint, key = datapoint.count)
        max_count_value = datapoint.count(major_char)
        datapoint.extend(["=>", max_count_value, "<EOS>"])
        dataset.append(datapoint)

    vocab = set(vocab) | set(["=>"]) | set(string.digits)
    return vocab, dataset


def counting(number_vocab, seq_length, num_examples, seed=8):
    random.seed(seed)
    vocab = random.sample(string.ascii_letters, number_vocab)
    dataset = []

        

if __name__ == '__main__':
    parser: Args = ArgumentParser(Args).parse_args()
    vocab, dataset = associative_recall(
        parser.number_vocab,
        parser.seq_length,
        parser.num_examples + parser.num_test_examples
    )

    sub_folder = DATA_FOLDER / f"{parser.type}_{parser.number_vocab}_{parser.seq_length}"
    sub_folder.mkdir(exist_ok=True)
    with open(sub_folder / 'vocab.json', 'w') as fp:
        json.dump(list(vocab), fp)

    with open(sub_folder / 'train.json', 'w') as fp:
        json.dump([datapoint for datapoint in dataset[:parser.num_examples]], fp)

    with open(sub_folder / 'test.json', 'w') as fp:
        json.dump([datapoint for datapoint in dataset[parser.num_examples:]], fp)