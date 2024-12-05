import os
from datasets import load_dataset


if __name__ == '__main__':
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root_dir, 'data')
    ds = load_dataset("wikimedia/wikipedia", "20231101.uk", cache_dir=data_dir)