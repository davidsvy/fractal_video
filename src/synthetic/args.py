import argparse
import datetime
from functools import partial

from .fractal.variations import var_idxs as var_available
from ..utils.data import find_files
from ..utils.other import size_to_str, time_to_secs


def positive(arg_name, dtype):
    def require_positive(value):
        number = dtype(value)
        if number <= 0:
            raise ValueError(f'Number {value} for arg "{arg_name}" must be positive.')
        
        return number

    return require_positive

pos_int = partial(positive, dtype=int)
pos_float = partial(positive, dtype=float)


def print_args(args):
    time_str = datetime.timedelta(seconds=int(args.time))
    print(f'Generating dataset with args:')
    print('#' * 50)
    blacklist = ['time', 'paths_param']
    print(f'\t--time: {time_str}')
    
    if hasattr(args, 'paths_param'):
        print(f'\t--n_classes: {len(args.paths_param)}')
    
    for attr, value in vars(args).items():
        if not attr in blacklist:
            print(f'\t--{attr}: {value}')
            

def print_status(args, n_created, time_run, counter_size):
    def secs_to_str(sec):
        return datetime.timedelta(seconds=int(sec))

    time_avg = time_run / n_created
    str_run = secs_to_str(time_run)

    print('#' * 60)
    if args.samples and not args.time:
        time_left = time_avg * max(0, args.samples - n_created)
        str_left = secs_to_str(time_left)
        print(f'Created {n_created}/{args.samples} samples.')
        print(
            f'Time -> So_far: {str_run}, Remaining: {str_left}, Avg: {time_avg:.1f}')

    elif args.time and not args.samples:
        time_left = max(0, args.time - time_run)
        str_left = secs_to_str(time_left)
        print(f'Created {n_created} samples.')
        print(
            f'Time -> So_far: {str_run}, Remaining: {str_left}, Avg: {time_avg:.1f}')

    else:
        time_left = max(0, min(time_avg * (args.samples - n_created), args.time - time_run))
        str_left = secs_to_str(time_left)
        print(f'Created {n_created}/{args.samples} samples.')
        print(
            f'Time -> So_far: {str_run}, Remaining: {str_left}, Avg: {time_avg:.1f}')

    size_total = size_to_str(counter_size.sum)
    size_avg = size_to_str(counter_size.avg)

    print(f'Space -> Total: {size_total}, Avg: {size_avg}')
            
            
def process_var(var):
    assert len(var), 'Provide variations'
    
    var_set = set(var)
    assert len(var_set) == len(var), f'Remove duplicate variations from {var}'
    
    if -1 in var_set:
        if len(var) == 1:
            return var
        
        else:
            raise ValueError('--var -1 cannot be provided with others.')
    

    var_unknown = sorted(list(var_set.difference(set(var_available))))

    if len(var_unknown):
        raise ValueError(f'Unkown variations: {var_unknown}. Available variations: {var_available}')
    
    var = sorted(var)
    
    return var
            
            
def process_args(args):
    if args.time:
        assert args.time > 0, 'Provide time > 0'
        
    if args.dir_in:
        paths_param = find_files(dir=args.dir_in, ext='npz')
        assert len(paths_param), f'No npz files found in {args.dir_in}'
        args.paths_param = paths_param
        
    if args.type != 'fractal':
        return args
    
    assert args.iter > args.iter_skip, 'Provide iter >> iter_skip'
    assert args.iter_b > args.iter_skip, 'Provide iter_b >> iter_skip'

    assert 0 < args.thres_b < 1

    args.var = process_var(args.var)

    return args
        

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--type', type=str, required=True, 
        choices=['fractal', 'perlin', 'octopus', 'dead_leaves'],
        help='Type of generated videos.')
    
    ##############################################################
    # Arguments for all types of videos
    ##############################################################
    
    parser.add_argument(
        '--samples', type=pos_int('samples'), default=0,
        help='Number of videos to generate.')
    parser.add_argument(
        '--time', type=time_to_secs, default=0,
        help='Time limit for dataset generation.')

    parser.add_argument(
        '--dir_out', type=str, default=None, required=True,
        help='Directory where the output will be stored.')
    parser.add_argument(
        '--dir_in', type=str, default=None,
        help='Directory that contains npz classes.')
    parser.add_argument(
        '--gen_param', type=pos_int('gen_param'), default=None,
        help='Generate parameters for classes.')
    
    parser.add_argument(
        '--res', type=pos_int('res'), default=256,
        help='Resolution of the created image.')
    parser.add_argument(
        '--fps', type=pos_int('fps'), default=12,
        help='fps of the generated videos.')
    parser.add_argument(
        '--seed', type=int, default=1,
        help='Seed for RNG.')
    parser.add_argument(
        '--print_every', type=pos_int('print_every'), default=50,
        help='How often to print status during dataset generation.')
    parser.add_argument(
        '--lib_save', type=str, default='ffmpeg', choices=['cv2', 'ffmpeg'],
        help='Library for saving videos as mp4 files.')
    
    ##############################################################
    # Arguments only for fractals
    ##############################################################
    
    parser.add_argument(
        '--img', action='store_true',
        help='Generate images instead of videos.')
    parser.add_argument(
        '--bs_point', type=pos_int('bs_point'), default=128,
        help='Number of points calculated at each iteration.')
    parser.add_argument(
        '--iter', type=pos_int('iter'), default=7500,
        help='Number of iterations for the chaos game algorithm.')
    parser.add_argument(
        '--iter_b', type=pos_int('iter_b'), default=3000,
        help='Number of iterations for border detection.')
    parser.add_argument(
        '--iter_skip', '-ns', type=pos_int('iter_skip'), default=200,
        help='Number of initial iterations that will be skipped.')
    parser.add_argument(
        '--thres_b', type=pos_float('thres_b'), default=0.9,
        help='Percentage of of area to keep while detecting borders.')
    parser.add_argument(
        '--res_b', type=pos_int('res_b'), default=128,
        help='Resolution for border detection.')
    parser.add_argument(
        '--var', nargs='+', type=int, default=[-1],
        help='Non linear functions. 0 means linear fractals. If -1 is provided, half the samples will be linear & the rest nonlinear.')
    parser.add_argument(
        '--linear_motion', action='store_true',
        help='Only linear interpolation will be used for videos.')

    args = parser.parse_args()
    args = process_args(args)

    return args