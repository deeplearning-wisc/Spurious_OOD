import numpy as np
from matplotlib import pyplot as plt
from utils import anom_utils
import torch
import os
from scipy.io import loadmat
import argparse



parser = argparse.ArgumentParser(description='OOD training for multi-label classification')
parser.add_argument('--name', '-n', required = True, type=str,
                    help='name of experiment')
parser.add_argument('--in-dataset', default="pascal", type=str, help='in-distribution dataset e.g. pascal')
parser.add_argument('--test_epochs', "-e", required = True, type=str,
                    help='# epoch to test performance')
parser.add_argument('--hist', default = False, type=bool,
                    help='if need to plot histogram')
parser.add_argument('--ntom', default =True, type=bool,
                    help='if load k + 1 model')
args = parser.parse_args()

def main():
    aurocs = {}
    fprs = {}
    for test_epoch in args.test_epochs.split():
        if args.ntom:
            save_dir =  f"./ntom_results/{args.name}"
            with open(os.path.join(save_dir, f'ntom_score_at_epoch_{test_epoch}.npy'), 'rb') as f:
                in_scores = np.load(f)
                out_scores = np.load(f)
            auroc, aupr, fpr = anom_utils.get_and_print_results(in_scores, out_scores,  "ImageNetOOD", f"K+1 score at epoch {test_epoch}")
            fprs[test_epoch] = fpr
            aurocs[test_epoch] = auroc
        else:
            save_dir =  f"./energy_results/{args.name}"
            with open(os.path.join(save_dir, f'energy_score_at_epoch_{test_epoch}.npy'), 'rb') as f:
                id_energy = np.load(f)
                id_sum_energy = np.load(f)
                ood_energy = np.load(f)
                ood_sum_energy = np.load(f)
            auroc, aupr, fpr = anom_utils.get_and_print_results(id_sum_energy, ood_sum_energy, "ImageNetOOD", f"Energy Sum at epoch {test_epoch}")
            fprs[test_epoch] = fpr
            aurocs[test_epoch] = auroc
            auroc, aupr, fpr = anom_utils.get_and_print_results( id_energy, ood_energy, "ImageNetOOD", f"Energy Max at epoch {test_epoch}")
        cp_dir = "checkpoints/{in_dataset}/{name}/".format(in_dataset=args.in_dataset, name=args.name)
        mAP = torch.load(os.path.join(cp_dir, "all_mAPs.data") )
        print(f"mAP: {mAP[int(test_epoch)-1]}")
    print("auroc for sum energy: ", aurocs)
    print("fpr for sum energy: ", fprs)

    if args.hist:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,24))

        _, bins, _ = ax1.hist(id_energy, bins = 20, density = True,alpha=0.5, label='id')
        ax1.hist(ood_energy, bins = bins,  density = True, alpha=0.5, label='ood')
        ax1.set_ylim(0, 0.65)
        ax1.legend(loc='upper right')
        ax1.set_title("Energy Max")

        ax2.hist(-1 * id_sum_energy, 20, density = True, alpha=0.5, label='id')
        ax2.hist(-1 * ood_sum_energy, 20,density = True, alpha=0.5, label='ood')
        ax2.set_ylim(0, 0.65)
        ax2.legend(loc='upper right')
        ax2.set_title("Energy Sum")

        plt.savefig("hist_fine_tune.png")


if __name__ == '__main__':
    # root='./datasets/pascal/'
    # split="voc12-val"
    # filePath = root + split + '.mat'
    # datafile = loadmat(filePath)
    # GT = datafile['labels']
    # Imglist = datafile['Imlist']
    # print("hi")


    # directory = "/u/a/l/alvinming/ood/Atom/informative-outlier-mining/checkpoints/pascal/debug_energy_blr_ratio_1_random"
    # mAP = torch.load(os.path.join(directory, "all_mAPs.data") ) 
    # print(f"mAP: {mAP[int('3')-1]}")

    main()