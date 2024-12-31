import _init_paths
import matplotlib.pyplot as plt
import argparse
plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist

parser = argparse.ArgumentParser(description='analysis results.')
parser.add_argument('--dataset_name', type=str, default="tnl2k_lang", help='Name of dataset (tnl2k_lang, lasot_lang).')
args = parser.parse_args()

trackers = []
dataset_name = args.dataset_name

trackers.extend(trackerlist(name='ctvlt', parameter_name='baseline', dataset_name=dataset_name,
                            run_ids=None, display_name='CTVLT'))

dataset = get_dataset(dataset_name)
print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'norm_prec', 'prec'))


