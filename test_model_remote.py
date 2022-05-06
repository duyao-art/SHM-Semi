from __future__ import print_function
import argparse
import os
import shutil
import string
import time
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision import models as models
from PIL import Image
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
import math
from sklearn.metrics import confusion_matrix
import string
from sklearn.metrics import classification_report


# -------------------- Definition of global variables used in the training  process --------------------

parser = argparse.ArgumentParser(description='PyTorch MixMatch Training of Data Anomaly Detection')
parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', default=160, type=int, metavar='N', help='train batch size')
parser.add_argument('--lr', default=0.001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--manualSeed', default=0, type=int, help='manual seed')
parser.add_argument('--gpu', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--n-labeled', default=1400, type=int, help='Number of labeled data')
parser.add_argument('--train_iteration', default=160, type=int, help='Number of iteration per epoch')
parser.add_argument('--out', default='anomaly@1400-timehistory-semisupervised', help='Directory to save the result')
parser.add_argument('--alpha', default=0.75, type=float, help='hyper-parameter of mixmatch')
parser.add_argument('--lambda_u', default=75, type=float, help='hyper-parameter of mixmatch')
parser.add_argument('--T', default=0.5, type=float, help='hyper-parameter of mixmatch')
parser.add_argument('--ema_decay', default=0.999, type=float, help='hyper-parameter of mixmatch')
parser.add_argument('--gamma', default=2, type=float, help='parameter of focal loss')
parser.add_argument('--number_class', default=7, type=int, help='number of class')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()

# Path of dataset for training and test
path_01 = '../time_history_01_120_100/'
path_02 = '../time_history_02_120_100/'
path_01_label = './201201.txt'
path_02_label = './201202.txt'
path_02_label_part = './201202fold.txt'

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
np.random.seed(args.manualSeed)

best_acc = 0


# -------------------- Definition of main, training, validation functions --------------------

# definition of the whole training process
def main():
    torch.cuda.empty_cache()
    since = time.time()
    global best_acc
    if not os.path.isdir(args.out):
        mkdir_p(args.out)

    # dataset preprocessing
    print("==> Data preprocessing before training")
    gray_mean = 0.1307
    gray_std = 0.3081
    transform_train = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor(),
                                          transforms.Normalize((gray_mean,), (gray_std,))])
    transform_val = transforms.Compose([transforms.ToTensor(), transforms.Normalize((gray_mean,), (gray_std,))])

    train_labeled_set, train_unlabeled_set, val_set, test_set = get_anomaly(args.n_labeled,
                                                                            transform_train=transform_train,
                                                                            transform_val=transform_val)

    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

    # Model preparation
    print("==> creating WRN-28-2")

    def create_model(ema=False):
        model = WideResNet()
        model = model.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    test_model(model, test_loader,test_set)
    time_end = time.time() - since
    print('Total Test Time: {:.0f}m {:.0f}s'.format(time_end // 60, time_end % 60))


def test_model(model, test_loader,test_set):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name_classes = ['normal', 'missing', 'minor', 'outlier', 'square', 'trend', 'drift']
    checkpoint = torch.load('./anomaly@1400-timehistory-semisupervised/model_best.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])

    class_correct = list(0. for i in range(args.number_class))
    class_total = list(0. for i in range(args.number_class))

    label_validation = []
    pred_validation = []
    score_validation = []

    model.eval()

    with open('result/probability_all.txt', 'w') as f_prob:
        with torch.no_grad():
            for data in test_loader:
                imgs, labels = data
                imgs = imgs.to(device)
                labels = labels.to(device)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                outputs = outputs.view(-1,outputs.size(-1))
                preds_softmax = F.softmax(outputs,dim=1)
                lines_softmax = preds_softmax.cpu().numpy()
                for i in range(len(lines_softmax)):
                    str_pred = ' '.join([str(x) for x in lines_softmax[i]])
                    f_prob.write(str_pred)
                    f_prob.write('\n')

                array_softmax = preds_softmax.gather(1,labels.view(-1,1)).t()
                label_validation.extend(labels.cpu().numpy())
                pred_validation.extend(preds.cpu().numpy())
                score_validation.extend(array_softmax.cpu().numpy()[0])

                list_correct = (preds == labels.data).squeeze()
                for i in range(len(labels)):
                    label = labels[i]
                    class_correct[label] += list_correct[i].item()
                    class_total[label] += 1
    f_prob.close()

    str_label = ' '.join([str(x) for x in label_validation])
    str_pred = ' '.join([str(x) for x in pred_validation])
    str_score = ' '.join([str(x) for x in score_validation])

    with open('result/label_validation.txt','w') as f_label:
        f_label.write(str_label)
    with open('result/pred_validation.txt','w') as f_pred:
        f_pred.write(str_pred)
    with open('result/score_validation.txt','w') as f_score:
        f_score.write(str_score)

    label_validation = []
    f = open('result/label_validation.txt', "r")  # 打开文件以便写入
    lines = f.readlines()
    for word in lines[0].split():
        word = word.strip(string.whitespace)
        label_validation.append(int(word))
    f.close()

    pred_validation = []
    f_pred = open('result/pred_validation.txt', "r")
    lines_pred = f_pred.readlines()
    for word in lines_pred[0].split():
        word = word.strip(string.whitespace)
        pred_validation.append(int(word))
    f_pred.close()

    score_validation = []
    f_score = open('result/score_validation.txt', "r")
    lines_score = f_score.readlines()
    for word in lines_score[0].split():
        word = word.strip(string.whitespace)
        score_validation.append(float(word))
    f_score.close()

    pred_validation = np.array(pred_validation)
    label_validation = np.array(label_validation)

    print(classification_report(label_validation,pred_validation,labels=[0,1,2,3,4,5,6],
                                target_names=['normal','missing','minor','outlier','square','trend','drift']))

# --------------- Definition of self-constructed dataset class for labelled and unlabeled data ---------------


# dataset class for labelled data
class Mydataset(Dataset):
    def __init__(self, data, transform, path, indexs=None):
        super().__init__()
        if indexs is not None:
            new_data = []
            for i in range(len(indexs)):
                flag = indexs[i]
                new_data.append(data[flag])
            self.data = new_data
            self.path = path
            self.transform = transform
        else:
            self.data = data
            self.path = path
            self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        image_path = os.path.join(self.path, self.data[item][0])
        img = Image.open(image_path).convert('L')
        img = self.transform(img)
        label = self.data[item][1]
        return img, label


# dataset class for unlabelled data
class MyDatasetUnlabelled(Dataset):

    def __init__(self, data, transform, path, indexs=None):
        super().__init__()
        if indexs is not None:
            new_data = []
            for i in range(len(indexs)):
                flag = indexs[i]
                new_data.append(data[flag])
            self.data = new_data
            self.path = path
            self.transform = transform
        else:
            self.data = data
            self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        image_path = os.path.join(self.path, self.data[item][0])
        img = Image.open(image_path).convert('L')
        img = self.transform(img)
        label = np.array([-1 for i in range(len(img))])
        return img, label


# Definition of obtaining Training/Validation/Test data index (path + label)
def get_anomaly(n_labeled, transform_train=None, transform_val=None):
    num_classes = args.number_class

    f_dataset_train = open(path_01_label, 'r')
    dataset = f_dataset_train.readlines()
    dataset_train = []
    for line in dataset:
        line = line.rstrip(' ')
        words = line.split(' ')
        if len(line) > 1:
            if (int(words[1]) >= 0) and (int(words[1]) < num_classes):
                dataset_train.append((words[0], int(words[1])))
            else:
                pass

    f_dataset_val = open(path_02_label_part, 'r')
    dataset = f_dataset_val.readlines()
    dataset_val = []
    for line in dataset:
        line = line.rstrip(' ')
        words = line.split(' ')
        if len(line) > 1:
            if (int(words[1]) >= 0) and (int(words[1]) < num_classes):
                dataset_val.append((words[0], int(words[1])))
            else:
                pass

    f_dataset_test = open(path_02_label, 'r')
    dataset = f_dataset_test.readlines()
    dataset_test = []
    for line in dataset:
        line = line.rstrip(' ')
        words = line.split(' ')
        if len(line) > 1:
            if (int(words[1]) >= 0) and (int(words[1]) < num_classes):
                dataset_test.append((words[0], int(words[1])))
            else:
                pass

    train_labeled_idxs, train_unlabeled_idxs = train_val_split_anomaly(int(n_labeled / 7))

    train_labeled_dataset = Mydataset(dataset_train, transform=transform_train, path=path_01,
                                      indexs=train_labeled_idxs, )
    train_unlabeled_dataset = MyDatasetUnlabelled(dataset_train, transform=TransformTwice(transform_train),
                                                  path=path_01, indexs=train_unlabeled_idxs)
    val_dataset = Mydataset(dataset_val, transform=transform_val, path=path_02)
    test_dataset = Mydataset(dataset_test, transform=transform_val, path=path_02)

    print(f"#Labeled: {len(train_labeled_idxs)} #Unlabeled: {len(train_unlabeled_idxs)} #Val: {len(val_dataset)} "
          f"#Test: {len(test_dataset)}")
    return train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset


# Splitting training data index based on the allowed labelled training data
def train_val_split_anomaly(n_labeled_per_class):
    num_classes = args.number_class
    f_dataset = open(path_01_label, 'r')
    dataset = f_dataset.readlines()
    dataset_label = []
    for line in dataset:
        line = line.rstrip(' ')
        words = line.split(' ')
        if len(line) > 1:
            if (int(words[1]) >= 0) and (int(words[1]) < num_classes):
                dataset_label.append(int(words[1]))
            else:
                pass
    labels = np.array(dataset_label)

    train_labeled_idxs = []
    train_unlabeled_idxs = []

    for i in range(0, 7):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        train_labeled_idxs.extend(idxs[:n_labeled_per_class])
        train_unlabeled_idxs.extend(idxs[n_labeled_per_class:])

    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)

    return train_labeled_idxs, train_unlabeled_idxs


# -------------------- Definition of Network Model used for training --------------------

# Spatial Pyramid Pooling Module for random input image size
class SPPLayer(nn.Module):

    def __init__(self, num_levels, pool_type='maxpool'):
        super(SPPLayer, self).__init__()
        self.num_levels = num_levels
        self.pooltype = pool_type

    def forward(self, x):
        num, channel, height, width = x.size()
        for i in range(self.num_levels):
            level = i + 1
            kernel_size = (math.ceil(height / level), math.ceil(width / level))
            stride = (math.ceil(height / level), math.ceil(width / level))
            padding = (
                math.floor((kernel_size[0] * level - height + 1) / 2),
                math.floor((kernel_size[1] * level - width + 1) / 2))

            if self.pooltype == 'maxpool':
                tensor = F.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding).view(num, -1)
            else:
                tensor = F.avg_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding).view(num, -1)

            if i == 0:
                x_flatten = tensor.view(num, -1)
            else:
                x_flatten = torch.cat((x_flatten, tensor.view(num, -1)), 1)

        return x_flatten


# The definition of WideResNet
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, activate_before_residual=False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None
        self.activate_before_residual = activate_before_residual

    def forward(self, x):
        if not self.equalInOut and self.activate_before_residual == True:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, activate_before_residual=False):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate,
                                      activate_before_residual)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, activate_before_residual):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate,
                                activate_before_residual))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, number_class=args.number_class, depth=28, widen_factor=2, dropRate=0.0, spplevel=3):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        self.conv1 = nn.Conv2d(1, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, activate_before_residual=True)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        self.bn1 = nn.BatchNorm2d(nChannels[3], momentum=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.spp = SPPLayer(spplevel)
        self.fc = nn.Linear(1792, number_class)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = self.spp(out)
        out = self.fc(out)
        return out


# Generation of EMA Model
class WeightEMA(object):

    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * args.lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param.dtype == torch.float32:
                ema_param.mul_(self.alpha)
                ema_param.add_(param * one_minus_alpha)
                param.mul_(1 - self.wd)


# -------------------- Definition of loss functions for training --------------------

# The original SemiLoss function of MixMatch in Pytorch version
class SemiLoss(object):

    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch):
        probs_u = torch.softmax(outputs_u, dim=1)
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u) ** 2)
        return Lx, Lu, args.lambda_u * linear_rampup(epoch)


# Focal loss module to consider class imbalance in the training process
class FocalLoss(nn.Module):

    def __init__(self, gamma=args.gamma, num_classes=args.number_class,
                 size_average=True):
        super(FocalLoss, self).__init__()
        self.size_average = size_average
        self.gamma = gamma

    def forward(self, preds, labels):
        preds = preds.view(-1, preds.size(-1))
        preds_softmax = F.softmax(preds, dim=1)
        preds_logsoft = torch.log(preds_softmax)
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma), preds_logsoft)
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


# The revision of SemiLoss function of MixＭatch by combining focal loss module to consider class imbalance
class SemiLossFocal(object):

    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch):

        # loss for labelled data, class imbalance is considered
        loss_x = FocalLoss()
        Lx = loss_x(outputs_x,targets_x)

        # loss for unlabelled data
        probs_u = torch.softmax(outputs_u, dim=1)
        Lu = torch.mean((probs_u - targets_u) ** 2)
        return Lx, Lu, args.lambda_u * linear_rampup(epoch)


# -------------------- Basic functions used --------------------

def linear_rampup(current, rampup_length=args.epochs):
    if rampup_length == 0:
        return 1.0
    else:
        # np.clip将对应数组中的元素限制在参数所列出的最大最小值之间,当超出这个范围,将超出的值自动替换为对应的最大最小值.
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)


def interleave_offsets(batch, nu):
    # //为向下取整
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]


# Two transformations for data augmentation of unlabelled training data
class TransformTwice:

    def __init__(self, transform):
        self.transform = transform

    # 这里对应 unlabelled sample 要经过两次随机变换

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2


# checkpoint (model and training parameter save)
def save_checkpoint(state, is_best, checkpoint=args.out, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


# -------------------- Main Function Run --------------------
if __name__ == '__main__':
    main()
