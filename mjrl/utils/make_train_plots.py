import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy
import csv
from mjrl.utils.logger import DataLog
import argparse

def make_train_plots(log = None,
                     log_path = None,
                     keys = None,
                     save_loc = None,
                     sample_key = 'num_samples',
                     x_scale = 1.0,
                     y_scale = 1.0):
    if log is None and log_path is None:
        print("Need to provide either the log or path to a log file")
    if log is None:
        logger = DataLog()
        logger.read_log(log_path)
        log = logger.log
    # make plots for specified keys
    for key in keys:
        if key in log.keys():
            fig = plt.figure(figsize=(10,6))
            ax1 = fig.add_subplot(111)
            try:
                cum_samples = [np.sum(log[sample_key][:i]) * x_scale for i in range(len(log[sample_key]))]
                ax1.plot(cum_samples, [elem * y_scale for elem in log[key]])
                ax1.set_xlabel('samples')
                # mark iteration on the top axis
                ax2 = ax1.twiny() 
                ax2.set_xlabel('iterations', color=(.7,.7,.7))
                ax2.tick_params(axis='x', labelcolor=(.7,.7,.7))
                ax2.set_xlim([0, len(log[key])])
            except:
                ax1.plot(log[key])
                ax1.set_xlabel('iterations')
            ax1.set_title(key)
            plt.savefig(save_loc+'/'+key+'.png', dpi=100)
            plt.close()

# MAIN =========================================================
# Example: python make_train_plots.py --log_path logs/log.csv --keys eval_score rollout_score save_loc logs
def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-l', '--log_path', type=str, required=True, help='path file to log.csv')
    parser.add_argument(
        '-k', '--keys', type=str, action='append', nargs='+', required=True, help='keys to plot')
    parser.add_argument(
        '-s', '--save_loc', type=str, default='', help='Path for logs')
    args = parser.parse_args()

    make_train_plots(log_path=args.log_path, keys=args.keys[0], save_loc=args.save_loc)

if __name__ == '__main__':
    main()

