import json
import shutil
import os
from os.path import join, exists, splitext, basename
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import torchvision
from torchvision import datasets
from torchvision import transforms as T
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

from models import *
from utils import *

parser = argparse.ArgumentParser(description='test GDN using another dataset')
parser.add_argument('--epochs', default=300, type=int,
                    help='number of total epochs to run for feature combine')
parser.add_argument('--dataset', default="cifar10", type=str,
                    help='choose dataset name (cifar10, cifar100, svhn)')
parser.add_argument('--num_class', default=10, type=int,
                    help='number of class (default: 10)')
parser.add_argument('--layers', default=100, type=int,
                    help='total number of layers (default: 100)')
parser.add_argument('--depth', default=28, type=int,
                    help='depth of wideresnet (default: 28)')
parser.add_argument('--widen_factor', default=10, type=int,
                    help='widen factor of wideresnet (default: 10)')
parser.add_argument('--z_dim', default=84, type=int,
                    help='dimension of penultimate layer (default: 84)')
parser.add_argument('--growth', default=12, type=int,
                    help='number of new channels per layer (default: 12)')
parser.add_argument('--droprate', default=0, type=float,
                    help='dropout probability (default: 0.0)')
parser.add_argument('--reduce', default=0.5, type=float,
                    help='compression rate in transition stage (default: 0.5)')
parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                    help='To not use bottleneck block')
parser.add_argument('--num_exp', default=1, type=int,
                    help='number of experiment')
parser.add_argument('--model', default="dense", type=str,
                    help='choose model name (dense, resnet, lenet)')
parser.add_argument('--num_valid', default=100, type=int,
                    help='number of validation data splitted from OOD data (default: 100)')
parser.add_argument('--num_valid_train', default=50, type=int,
                    help='number of validation train data splitted from validation OOD data (default: 50)')
parser.add_argument('--k', default=5, type=int,
                    help='save every k-th epoch (default: 5)')
tp = lambda x: list(map(int, x.split(',')))
parser.add_argument('--rep', default="1,1,1", type=tp,
                    help='add reparameterization (default: 1,1,1)')
parser.set_defaults(bottleneck=True)


def main():
    global args, weights_dir
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    result_dir = "./results"
    if not exists(result_dir):
        os.makedirs(result_dir)

    # Data loading code
    if args.dataset == 'svhn':
        normalize = T.Normalize(mean=[x / 255.0 for x in[109.9, 109.7, 113.8]],
                                         std=[x / 255.0 for x in [50.1, 50.6, 50.8]])
    else:
        normalize = T.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    transform_test = T.Compose([
        T.ToTensor(),
        normalize
        ])

    kwargs = {'num_workers': 1, 'pin_memory': True}
    if args.dataset == "cifar10":
        validationset = datasets.CIFAR10('./data', train=True, download=True, transform=transform_test)
        testset = datasets.CIFAR10('./data', train=False, download=True, transform=transform_test)
    elif args.dataset == "cifar100":
        validationset = datasets.CIFAR100('./data', train=True, download=True, transform=transform_test)
        testset = datasets.CIFAR100('./data', train=False, download=True, transform=transform_test)
    elif args.dataset == "svhn":
        validationset = datasets.SVHN('./data', split="train", download=True, transform=transform_test)
        testset = datasets.SVHN('./data', split="test", download=True, transform=transform_test)

    # in-distribution validation dataset
    valid_num = 1000
    validationset = Subset(validationset, np.arange(valid_num))
    validloader = DataLoader(validationset, batch_size=100, shuffle=True, **kwargs)

    valid_train_num = 800
    validationset_train = Subset(validationset, np.arange(valid_train_num))
    validationset_test = Subset(validationset, np.arange(valid_train_num, valid_num))

    validloader_train = DataLoader(validationset_train, batch_size=100, shuffle=True, **kwargs)
    validloader_test = DataLoader(validationset_test, batch_size=100, shuffle=False, **kwargs)
    testloader = DataLoader(testset, batch_size=100, shuffle=True, **kwargs)

    np.random.seed(42)
    testsetouts = [
        datasets.ImageFolder("./data/Imagenet_resize", transform=transform_test),
        datasets.ImageFolder("./data/LSUN_resize", transform=transform_test),
        datasets.ImageFolder("./data/iSUN", transform=transform_test),
        GaussianNoise(size=(3, 32, 32), n_samples=10000, mean=0.5, variance=1.0),
        UniformNoise(size=(3, 32, 32), n_samples=10000, low=0., high=1.)
    ]
    testsetouts_name = ["TIM", "LSUN", "iSUN", "Gaussian", "Uniform"]

    # create model
    rep_ind = "".join(np.array(args.rep).astype("str"))
    if args.model == "dense":
        model_name = "DenseNet_BC_{}_{}_{}".format(rep_ind, args.layers, args.growth)
        f_x = DenseNet3(args.layers, args.num_class, args.growth, reduction=args.reduce,
                             bottleneck=args.bottleneck, dropRate=args.droprate, rep=args.rep).to(device)
    elif args.model == "resnet":
        model_name = "WideResnet_{}_{}_{}".format(rep_ind, args.depth, args.widen_factor)
        f_x = WideResNet(args.depth, args.num_class, args.widen_factor, args.droprate, rep=args.rep).to(device)
    elif args.model == "lenet":
        model_name = "Lenet_{}_{}".format(rep_ind, args.z_dim)
        f_x = LeNet5(args.z_dim, args.num_class, rep=args.rep).to(device)
    weights_dir = "./weights/{}_{}_{}_{}".format(
        model_name, args.num_class, args.num_exp, args.dataset)

    y_dim = args.num_class
    ID_list = range(y_dim)
    exp_num = args.num_exp

    # the number of validation set of OOD
    valid_num = args.num_valid
    valid_train_num = args.num_valid_train

    print("In-dataset: {}".format(args.dataset))
    print("weight dir: {}".format(weights_dir))
    f_x.load_state_dict(torch.load(join(weights_dir, "model_best.pth.tar"))["state_dict"])
    f_x.eval()

    for i, (testsetout, name_out) in enumerate(zip(testsetouts, testsetouts_name)):

        validsetOut_train = Subset(testsetout, np.arange(valid_train_num))
        validsetOut_test = Subset(testsetout, np.arange(valid_train_num, valid_num))
        validsetOut = Subset(testsetout, np.arange(valid_num))
        testsetOut = Subset(testsetout, np.arange(valid_num, len(testsetout)))

        validloaderOut_train = DataLoader(validsetOut_train, batch_size=100, shuffle=True)
        validloaderOut_test = DataLoader(validsetOut_test, batch_size=100, shuffle=False)
        validloaderOut = DataLoader(validsetOut, batch_size=100, shuffle=False)
        testloaderOut = DataLoader(testsetout, batch_size=100, shuffle=False)

        metric_in_list = []
        metric_out_list = []
        accs = dict()
        stypes = []
        print("dataset out: {}".format(name_out))
        num_rep = np.array(args.rep).sum()
        if num_rep:
            if i == 0:
                features, acc = return_metrics(f_x, testloader, ID_list, rep=args.rep)
                features_v, _ = return_metrics(f_x, validloader, device="cuda", rep=args.rep)

            features_out_v, _ = return_metrics(f_x, validloaderOut, device="cuda", rep=args.rep)
            w = return_LR_weights(features_v, features_out_v)
            features_out, _ = return_metrics(f_x, testloaderOut, rep=args.rep)

            in_feature = w[0] + w[1].dot(features)
            out_feature = w[0] + w[1].dot(features_out)

            stype = "UFEL (LR)"
            accs[stype] = acc
            stypes.append(stype)


            if args.model == "lenet":
                g_x = Feature_Ensemble_CNN_for_Lenet(args.num_class).to(device)
            elif args.model == "dense":
                g_x = Feature_Ensemble_CNN_for_Dense(args.num_class).to(device)
            elif args.model == "resnet":
                g_x = Feature_Ensemble_CNN_for_resnet(args.num_class).to(device)

            optimizer = optim.Adam(g_x.parameters(), lr=1e-3)
            criterion_BCE = nn.BCEWithLogitsLoss()

            if args.model == "lenet":
                train_lenet(f_x, g_x, validloader_train, validloaderOut_train, validloader_test, validloaderOut_test,
                            criterion_BCE, optimizer, device="cuda")
                g_x.load_state_dict(torch.load(join(weights_dir, "g_x_best.pth.tar"))["state_dict"])
                g_x.eval()

                pred_i, pred_o, acc = test_lenet(f_x, g_x, testloader, testloaderOut, device="cuda")

            else:
                train_dense(f_x, g_x, validloader_train, validloaderOut_train, validloader_test, validloaderOut_test,
                            criterion_BCE, optimizer, device="cuda")
                g_x.load_state_dict(torch.load(join(weights_dir, "g_x_best.pth.tar"))["state_dict"])
                g_x.eval()

                pred_i, pred_o, acc = test_dense(f_x, g_x, testloader, testloaderOut, device="cuda")

            stype = "UFEL (CNN)"
            accs[stype] = acc
            stypes.append(stype)

            metric_in_list = np.stack([in_feature, pred_i])
            metric_out_list = np.stack([out_feature, pred_o])

        else:
            if i == 0:
                softmax, acc = return_metrics(f_x, testloader, ID_list, rep=args.rep)

            temperatures = [1, 10, 100, 1000]
            magnitudes = [0, 0.001, 0.005, 0.01, 0.05, 0.1]
            df = pd.DataFrame(columns=temperatures, index=magnitudes)
            for temperature in temperatures:
                for magnitude in magnitudes:
                    softmax_v = odin(f_x, validloader, temperature, magnitude, device="cuda")
                    softmax_out_v = odin(f_x, validloaderOut, temperature, magnitude, device="cuda")
                    result = metric([softmax_v], [softmax_out_v],
                                    stypes=["odin"], accs={"odin": acc}, verbose=False)
                    df[temperature][magnitude] = result["odin"]["AUROC"]
            magnitude_m = df.max(1).idxmax()
            temperature_m = df.max(0).idxmax()
            auroc_m = df.max().max()
            print("max mag: {}, temp: {}, auroc: {}".format(magnitude_m, temperature_m, auroc_m))

            softmax_odin = odin(f_x, testloader, temperature_m, magnitude_m, device="cuda")

            softmax_out, _ = return_metrics(f_x, testloaderOut, rep=args.rep)
            softmax_out_odin = odin(f_x, testloaderOut, temperature_m, magnitude_m, device="cuda")

            stype = "Baseline"
            metric_in_list.append(softmax)
            metric_out_list.append(softmax_out)
            accs[stype] = acc
            stypes.append(stype)

            stype = "ODIN"
            metric_in_list.append(softmax_odin)
            metric_out_list.append(softmax_out_odin)
            accs[stype] = acc
            stypes.append(stype)

        # write results
        results = metric(metric_in_list, metric_out_list, accs, stypes)
        filename = "{}_ydim{}_expnum{}_{}_{}_{}_{}.json".format(
            model_name, y_dim, exp_num, args.dataset, name_out, args.num_valid, args.num_valid_train)
        fw = open(join(result_dir, filename), "w")
        json.dump(results, fw, indent=4)

def train_dense(f_x, g_x, validloader_train, validloaderOut_train, validloader_test, validloaderOut_test,
                criterion_BCE, optimizer, device="cuda"):
    f_x.eval()
    for epoch in tqdm(range(args.epochs)):
        for (x_i, _), (x_o, _) in zip(validloader_train, validloaderOut_train):
            with torch.no_grad():
                output_i = f_x(x_i.to(device))
                logvar1_i = f_x.logvar1
                logvar2_i = f_x.logvar2
                logvar3_i = f_x.logvar3

                output_o = f_x(x_o.to(device))
                logvar1_o = f_x.logvar1
                logvar2_o = f_x.logvar2
                logvar3_o = f_x.logvar3

            x_1 = torch.cat([logvar1_i, logvar1_o])
            x_2 = torch.cat([logvar2_i, logvar2_o])
            x_3 = torch.cat([logvar3_i, logvar3_o])
            x_4 = torch.cat([output_i.max(1)[0], output_o.max(1)[0]])[:, None]

            y = np.concatenate([np.repeat(1, len(x_i)), np.repeat(0, len(x_o))])
            y = torch.from_numpy(y[:, None]).float()

            index = np.arange(len(y))
            np.random.shuffle(index)
            x_1, x_2, x_3, x_4, y = x_1[index], x_2[index], x_3[index], x_4[index], y[index]

            g_x.train()
            optimizer.zero_grad()
            output = g_x(x_1, x_2, x_3, x_4)
            loss = criterion_BCE(output, y.to(device))
            loss.backward()
            optimizer.step()

        best_prec1 = 0
        if (epoch+1) % args.k == 0:
            pred_i, pred_o, acc = test_dense(f_x, g_x, validloader_test, validloaderOut_test, device="cuda")
            stype = "UFEL (CNN)"
            accs = {stype: acc}
            result = metric([pred_i], [pred_o], stypes=[stype], accs=accs, verbose=False)
            prec1 = result[stype]["AUROC"]

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': g_x.state_dict(),
                'best_prec1': best_prec1,
            }, is_best)

def test_dense(f_x, g_x, testloader, testloaderOut, device="cuda"):
    g_x.eval()
    pred = []
    test = []
    pred_i = []
    for x_i, y in testloader:
        with torch.no_grad():
            output = f_x(x_i.to(device))
            argmax = output.max(1)[1]
            output_i = output.max(1)[0][:, None]
            logvar1_i = f_x.logvar1
            logvar2_i = f_x.logvar2
            logvar3_i = f_x.logvar3
            pred_i.extend(g_x(logvar1_i, logvar2_i, logvar3_i, output_i).detach().cpu().numpy())
            pred.extend(argmax.detach().cpu().numpy())
            test.extend(y.numpy())

    pred_o = []
    for x_o, _ in testloaderOut:
        with torch.no_grad():
            output_o = f_x(x_o.to(device)).max(1)[0][:, None]
            logvar1_o = f_x.logvar1
            logvar2_o = f_x.logvar2
            logvar3_o = f_x.logvar3
            pred_o.extend(g_x(logvar1_o, logvar2_o, logvar3_o, output_o).detach().cpu().numpy())

    pred_i = np.array(pred_i).flatten()
    pred_o = np.array(pred_o).flatten()
    acc = accuracy_score(test, pred)
    return pred_i, pred_o, acc

def train_lenet(f_x, g_x, validloader_train, validloaderOut_train, validloader_test, validloaderOut_test,
                criterion_BCE, optimizer, device="cuda"):
    f_x.eval()
    for epoch in tqdm(range(args.epochs)):
        for (x_i, _), (x_o, _) in zip(validloader_train, validloaderOut_train):
            with torch.no_grad():
                output_i = f_x(x_i.to(device))
                logvar2_i = f_x.logvar2
                logvar3_i = f_x.logvar3

                output_o = f_x(x_o.to(device))
                logvar2_o = f_x.logvar2
                logvar3_o = f_x.logvar3

            x_2 = torch.cat([logvar2_i, logvar2_o])
            x_3 = torch.cat([logvar3_i, logvar3_o])
            x_4 = torch.cat([output_i.max(1)[0], output_o.max(1)[0]])[:, None]

            y = np.concatenate([np.repeat(1, len(x_i)), np.repeat(0, len(x_o))])
            y = torch.from_numpy(y[:, None]).float()

            index = np.arange(len(y))
            np.random.shuffle(index)
            x_2, x_3, x_4, y = x_2[index], x_3[index], x_4[index], y[index]

            g_x.train()
            optimizer.zero_grad()
            output = g_x(x_2, x_3, x_4)
            loss = criterion_BCE(output, y.to(device))
            loss.backward()
            optimizer.step()

        best_prec1 = 0
        if (epoch+1) % args.k == 0:
            pred_i, pred_o, acc = test_lenet(f_x, g_x, validloader_test, validloaderOut_test, device="cuda")
            stype = "UFEL (CNN)"
            accs = {stype: acc}
            result = metric([pred_i], [pred_o], stypes=[stype], accs=accs, verbose=False)
            prec1 = result[stype]["AUROC"]

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': g_x.state_dict(),
                'best_prec1': best_prec1,
            }, is_best)

def test_lenet(f_x, g_x, testloader, testloaderOut, device="cuda"):
    g_x.eval()
    pred = []
    test = []
    pred_i = []
    for x_i, y in testloader:
        with torch.no_grad():
            output = f_x(x_i.to(device))
            argmax = output.max(1)[1]
            output_i = output.max(1)[0][:, None]
            logvar2_i = f_x.logvar2
            logvar3_i = f_x.logvar3
            pred_i.extend(g_x(logvar2_i, logvar3_i, output_i).detach().cpu().numpy())
            pred.extend(argmax.detach().cpu().numpy())
            test.extend(y.numpy())

    pred_o = []
    for x_o, _ in testloaderOut:
        with torch.no_grad():
            output_o = f_x(x_o.to(device)).max(1)[0][:, None]
            logvar2_o = f_x.logvar2
            logvar3_o = f_x.logvar3
            pred_o.extend(g_x(logvar2_o, logvar3_o, output_o).detach().cpu().numpy())

    pred_i = np.array(pred_i).flatten()
    pred_o = np.array(pred_o).flatten()
    acc = accuracy_score(test, pred)
    return pred_i, pred_o, acc

def save_checkpoint(state, is_best, filename='log.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "%s/" % (weights_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '%s/' % (weights_dir) + 'g_x_best.pth.tar')

if __name__ == '__main__':
    main()
