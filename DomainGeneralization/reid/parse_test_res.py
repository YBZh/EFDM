"""
Goal
---
1. Read test results from train.log*** files
2. Compute mean and std across different folders (seeds)

Usage
---
Assume the output files are saved under output/my_experiment,
which contains results of different seeds, e.g.,

my_experiment/
    exp-1/
        train.log***
    exp-2/
        train.log***


Run the following command from the root directory:

$ python parse_test_res.py output/my_experiment/exp-1

Add --ci95 to the argument if you wanna get 95% confidence
interval instead of standard deviation:

$ python tools/parse_test_res.py output/my_experiment --ci95

If my_experiment/ has the following structure,

my_experiment/
    exp-1/
        train.log***
    exp-2/
        train.log***

Run

$ python parse_test_res.py output/my_experiment/ --multi-exp
"""
import re
import numpy as np
import os.path as osp
import os
import argparse
from collections import OrderedDict, defaultdict

from dassl.utils import check_isfile, listdir_nohidden
import ipdb

def compute_ci95(res):
    return 1.96 * np.std(res) / np.sqrt(len(res))


def parse_function(*metrics, directory='', args=None, end_signal=None):
    print(f'Parsing files in {directory}')
    # subdirs = listdir_nohidden(directory, sort=True)
    outputs = []
    file_list = os.listdir(directory)
    for file in file_list:
        if 'train.log' in file:
            fpath = osp.join(directory, file)
            with open(fpath, 'r') as f:
                lines = f.readlines()
                complete_flag = False
                for line in lines:
                    if 'Final' in line:
                        complete_flag = True
                        output = OrderedDict()
                    if complete_flag:
                        # print (line)
                        if 'mAP:' in line:
                            # ipdb.set_trace()
                            num = float(line.split(':')[1].split('%')[0])
                            output['mAP:'] = num
                        elif 'Rank-1 ' in line:
                            num = float(line.split(':')[1].split('%')[0])
                            output['Rank-1'] = num
                        elif 'Rank-5' in line:
                            num = float(line.split(':')[1].split('%')[0])
                            output['Rank-5'] = num
                        elif 'Rank-10' in line:
                            num = float(line.split(':')[1].split('%')[0])
                            output['Rank-10'] = num
                        else:
                            pass
            if complete_flag:
                outputs.append(output)
        else:
            pass

    metrics_results = defaultdict(list)

    for output in outputs:
        msg = ''
        for key, value in output.items():
            if isinstance(value, float):
                msg += f'{key}: {value:.2f}%. '
            else:
                msg += f'{key}: {value}. '
            if key != 'file':
                metrics_results[key].append(value)
        print(msg)

    output_results = OrderedDict()

    print('===')
    print(f'Summary of directory: {directory}')
    for key, values in metrics_results.items():
        avg = np.mean(values)
        std = compute_ci95(values) if args.ci95 else np.std(values)
        print(f'* {key}: {avg:.2f}% +- {std:.2f}%')
        output_results[key] = avg
    print('===')

    return output_results


def main(args, end_signal):
    metric1 = {
        'name': 'accuracy',
        'regex': re.compile(r'\* accuracy: ([\.\deE+-]+)%')
    }

    metric2 = {
        'name': 'error',
        'regex': re.compile(r'\* error: ([\.\deE+-]+)%')
    }

    if args.multi_exp:
        final_results = defaultdict(list)

        for directory in listdir_nohidden(args.directory, sort=True):
            directory = osp.join(args.directory, directory)
            results = parse_function(
                metric1,
                metric2,
                directory=directory,
                args=args,
                end_signal=end_signal
            )

            for key, value in results.items():
                final_results[key].append(value)

    else:
        parse_function(
            metric1,
            metric2,
            directory=args.directory,
            args=args,
            end_signal=end_signal
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=str, help='path to directory')
    parser.add_argument(
        '--ci95',
        action='store_true',
        help=r'compute 95\% confidence interval'
    )
    parser.add_argument(
        '--test-log', action='store_true', help='parse test-only logs'
    )
    parser.add_argument(
        '--multi-exp', action='store_true', help='parse multiple experiments'
    )
    args = parser.parse_args()

    end_signal = 'Finished training'
    if args.test_log:
        end_signal = '=> result'

    main(args, end_signal)
