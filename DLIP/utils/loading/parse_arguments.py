import argparse
import logging


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config_files',
        '-cfg',
        type=str,
        help='Path to config file(s)'
    )
    parser.add_argument(
        '--result_dir',
        type=str,
        default="./results",
        help='Path to result_dir'
    )
    args, _ = parser.parse_known_args()
    if 'config_files' not in args:
        raise ValueError('config_files in parameters missing, aborting!')
    if 'result_dir' not in args:
        raise ValueError('result_dir in parameters missing, aborting!')
    logging.info(f"Reading parameter file(s) {args.config_files}")
    logging.info(f"Result directory: {args.result_dir}")
    return args.config_files, args.result_dir