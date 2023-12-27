import argparse


def parse_args_train():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--cfg', '-c', nargs='+', type=str, default=None,
        help='Paths to yaml files that overwrite the default config.'
    )
    parser.add_argument(
        '--device', '-d', type=int, default=None,
        help='GPU indices. If none provided, CPU will be used by default.'
    )
    parser.add_argument(
        '--opts', '-o', nargs=argparse.REMAINDER, default=None,
        help='Arguments to overwrite config.'
    )

    args = parser.parse_args()

    return args


def parse_args_prepare():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset', '-d', type=str, required=True)
    parser.add_argument(
        '--root', '-r', type=str, default='data')

    args = parser.parse_args()

    return args
