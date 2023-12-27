import os

from src.prepare_data import prepare_data
from src.utils.args import parse_args_prepare


def main():
    args = parse_args_prepare()
    args.dataset = args.dataset.lower()

    root = os.path.join(args.root, args.dataset)
    prepare_data(dataset=args.dataset, root=root)


if __name__ == '__main__':
    main()
