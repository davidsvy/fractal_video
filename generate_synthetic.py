from src.synthetic import (
    build_dead_leaves,
    build_fractal,
    build_octopus,
    build_perlin,
    parse_args,
    print_args,
)

from src.utils.other import set_seed


def main():
    args = parse_args()
    print_args(args)
    
    set_seed(args.seed)

    if args.type == 'dead_leaves':
        build_dead_leaves(args)
    elif args.type == 'fractal':
        build_fractal(args)
    elif args.type == 'octopus':
        build_octopus(args)
    elif args.type == 'perlin':
        build_perlin(args)
    else:
        raise ValueError(f'Unknown dataset type: {args.type}')


if __name__ == '__main__':
    main()
