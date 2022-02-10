import argparse

def parse_args():
    """Parse all arguments."""
    parser = argparse.ArgumentParser(description="Triaffine-BERT arguments", allow_abbrev=False)

    parser = _add_data_args(parser)
    parser = _add_model_args(parser)
    parser = _add_training_args(parser)

    return parser

def _add_data_args(parser):
    parser.add_argument('--vocab_file', type=str, default=None, help='Path to the vocab file.')
    parser.add_argument('--max_seq_length', type=int, default=None, help='Maximum sequence length to process')
    parser.add_argument('--train_data_path', type=str, default=None, help='Path to the training dataset')
    parser.add_argument('--val_data_path', type=str, default=None, help='Path to the validation dataset')

def _add_model_args(parser):
    None

def _add_training_args(parser):
    None