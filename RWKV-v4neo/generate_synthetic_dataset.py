import numpy as np
from argparse import ArgumentParser
from pathlib import Path
from tqdm.auto import trange

ROOT_FOLDER = Path(__file__).parent / 'data'
ROOT_FOLDER.mkdir(exist_ok=True)

def generate_associative_recall(
    vocab_size,
    sequence_length,
    train_size,
    test_size
):  
    train_dataset, test_dataset = [], []
    for _ in trange(train_size, desc="Generating training dataset."):
        train_dataset.append(
            generate_associative_recall_datapoint(vocab_size=vocab_size, sequence_length=sequence_length)
        )
        
    train_dataset = np.stack(train_dataset)
        
    for _ in trange(test_size, desc="Generating test dataset."):
        test_dataset.append(
            generate_associative_recall_datapoint(vocab_size=vocab_size, sequence_length=sequence_length)
        )
        
    test_dataset = np.stack(test_dataset)
    
    np.save(ROOT_FOLDER / f"assoc_recall_{vocab_size}_{sequence_length}_{train_size}_train.npy", train_dataset)
    np.save(ROOT_FOLDER / f"assoc_recall_{vocab_size}_{sequence_length}_{test_size}_test.npy", test_dataset)
    
    print("Saved files:\n\t- {}\n\t- {}".format(
        (ROOT_FOLDER / f"assoc_recall_{vocab_size}_{sequence_length}_{train_size}_train.npy").as_posix(),
        (ROOT_FOLDER / f"assoc_recall_{vocab_size}_{sequence_length}_{test_size}_test.npy").as_posix()
    ))
    
def generate_associative_recall_datapoint(
    vocab_size,
    sequence_length
):
    mapping = {i: np.random.randint(vocab_size) for i in range(vocab_size)}
    sequence = np.random.randint(vocab_size, size=sequence_length // 2)
    mapped_sequence = np.vectorize(lambda x: mapping[x])(sequence)
    
    return np.stack([sequence, mapped_sequence]).T.ravel()


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--dataset-type",  type=str)  # full path, with .pth
    parser.add_argument("--vocab-size",  type=int)  # wandb project name. if "" then don't use wandb
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--train-size", type=int)
    parser.add_argument("--test-size", type=int)
    parser.add_argument("--random-seed", default=42, type=int)
    
    args = parser.parse_args()
    np.random.seed(args.random_seed)
    
    if args.dataset_type == "associative-recall":
        generate_associative_recall(
            vocab_size=args.vocab_size,
            sequence_length=args.sequence_length,
            train_size=args.train_size,
            test_size=args.test_size,
        )
    else:
        raise ValueError(f"{args.dataset_type} dataset type isn't supported")
    
    
    