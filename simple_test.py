import numpy as np
import torch
import argparse
import json
import warnings
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import torch.utils.data as Data
import models.densenet as dn
import models.simplenet as sn
from neural_linear_simple import NeuralLinearSimple
import torch.nn as nn

def generate_batch_training_data(num = 20):
    # np.random.seed(0)
    # noise = np.random.normal(mu, sigma, num)
    x = np.random.random_sample((num,))*6 -3 # Return random floats in the half-open interval [0.0, 1.0).
    y = np.array([np.sin(item) for item in x])
    return x, y

def generate_test_data(num = 200):
    x = np.arange(-6, 6, 12/num)
    y = np.array([np.sin(item) for item in x])
    return x, y

def get_y_from_x(x , mu = 0, sigma = 0.1):
    return np.array([np.sin(i) + np.random.normal(mu, sigma) for i in x])

if __name__ == "__main__":
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    parser = argparse.ArgumentParser(description=None)

    # total episode
    parser.add_argument('--train-episodes', type=int, default=100, metavar='NS', help='episode number of testing the trained cem')
    # nn model direction
    parser.add_argument('--load-cost-model-dir', default=None, metavar='LMD', help='folder to load trained models from')
    parser.add_argument('--load-dx-model-dir', default=None, metavar='LMD', help='folder to load trained models from')
    # no use
    parser.add_argument('--trueenv', dest='trueenv', action='store_true', help='use true env in collecting trajectories')
    # NN
    parser.add_argument('--hidden-dim', type=int, default=50, metavar='NS',
                        help='hidden dimention of fc network[10 for cartpole]')
    #parser.add_argument('--max-collect-eps', type=int, default=300, metavar='NS',
                        #help='number of iterating the distribution params')
    parser.add_argument('--collect-step', type=int, default=200, metavar='NS',
                        help='episode number of testing the trained cem')
    # parser.add_argument('--gpu-ids', type=int, default=None, nargs='+', help='GPUs to use [-1 CPU only] (default: -1)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='T', help='learning rate in training nn.')
    parser.add_argument('--training-iter', type=int, default=100, metavar='NS', help='iterations in training nn')
    parser.add_argument('--batch-size', type=int, default=32, metavar='NS', help='batch size in training nn')
    # parser.add_argument('--data-type', default='random', metavar='M', help='collect data to train nn [random, cem]')
    parser.add_argument('--log-dir', default='logs/', metavar='LG', help='folder to save logs')
    # parser.add_argument('--gp-models-dir', default='gp-models/', metavar='LG', help='folder to save logs')
    parser.add_argument('--optimize', dest='optimize', action='store_true', help='if true, use gp optimization')
    # parser.add_argument('--save-data-dir', default=None, metavar='LG', help='folder to save logs')

    #neural_bayes param
    parser.add_argument('--a0', type=float, default=6.0, metavar='T', help='a0')
    parser.add_argument('--b0', type=float, default=6.0, metavar='T', help='b0')
    parser.add_argument('--lambda_prior', type=float, default=0.25, metavar='T', help='lambda_prior')
    parser.add_argument('--var_scale', type=float, default=4.0, metavar='T', help='var_scale')
    parser.add_argument('--sigma', type=float, default= 10, help='control var for weights')
    parser.add_argument('--sigma_n', type=float, default=0.1, help='control var for noise')

    parser.add_argument('--if_save', type=bool, default=False, metavar='T', help='save model params or not')

    parser.add_argument('--num-samples', type=int, default=31, metavar='NS', help='number of sampling from params distribution')

    args = parser.parse_args()

    model = sn.SimpleNet(num_classes = 1)
    model = model.cuda()
    bayes_nn = NeuralLinearSimple(args, model, repr_dim = 20, output_dim = 1)

    eps = 10
    sample_num = 5
    num = 100
    test_num = 200

    test_x, test_y = generate_test_data(test_num)
    #print("before cov: {}".format(my_cost.cov_w[:, :3, :3]))
    #print("my dx cov: {}".format(my_cost.cov_w[:, :5, :5]))

    # criterion = nn.CrossEntropyLoss().cuda()
    # optimizer = torch.optim.SGD(model.parameters(), 0.1,
    #                             momentum=0.9,
    #                             nesterov=True,
    #                             weight_decay=0.0001)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    # train
    for i in range(eps):
        train_x, train_y = generate_batch_training_data(num = 100)
        plt.scatter(train_x, train_y, c = 'black', s = 1)
        train_x = torch.from_numpy(train_x.reshape([num,1])).float()
        train_y = torch.from_numpy(train_y.reshape([num,1])).float()
        bayes_nn.push_to_buffer(new_x=train_x, new_y=train_y)
        torch_dataset = Data.TensorDataset(train_x, train_y)
        loader = Data.DataLoader(dataset=torch_dataset, batch_size= 20, shuffle=True)
        bayes_nn.sample_BDQN()
        bayes_nn.train_blr_in(loader, model, criterion, 1,  optimizer, i)
        bayes_nn.update_representation()
        bayes_nn.update_bays_reg_BDQN()
        #DEBUG
        print("cov and mean checking...")
        print(f"cov: {bayes_nn.cov_w[0,:6,:6]}")
        print(f"mu: {bayes_nn.mu_w[0][:10  ]}")
        # print("my cost cov: {}".format(my_cost.cov_w[:, :5, :5]))

    # test
    y_all = np.empty([test_num, 0])
    for i in range(sample_num - 1):
        bayes_nn.sample_BDQN()
        pred_y = bayes_nn.predict(torch.from_numpy(test_x.reshape([test_num,1])).cuda().float() )
        y_all = np.hstack((y_all, pred_y.cpu().numpy()))
    y_mean = np.mean(y_all, axis = 1)
    y_std = np.std(y_all, axis = 1)
    plt.plot(test_x, y_mean)
    plt.fill_between(test_x, y_mean-y_std, y_mean+y_std, alpha = 0.5)
    plt.savefig("test.png")