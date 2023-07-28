import config 
import argparse
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--grabe_exist_package", default=False, help="Must be True or False", action='store_true')
parser.add_argument("--merge_data_labels", default=False, help="Must be True or False", action='store_true')
parser.add_argument("--merge_training_dev", default=False, help="Must be True or False", action='store_true')
parser.add_argument("--merge_gold_soft_label", default=False, help="Must be True or False", action='store_true')
args = parser.parse_args()

if __name__ == "__main__":
    
    if args.grabe_exist_package:
        grabe_exist_package(
            config.PATH_SORCE_PACKAGE,
            config.PACKAGE_PATH)
    
    if args.merge_data_labels:
        merge_data_labels(
            config.PACKAGE_PATH, 
            config.LABEL_GOLD_PATH, 
            config.DATA_PATH, 
            config.DATA)
        
    if args.merge_training_dev:
        merge_training_dev(
            config.DATA_PATH, 
            config.DATA)

    if args.merge_gold_soft_label:
        merge_gold_soft_label(
            config.LABEL_GOLD_PATH, 
            config.DATA)