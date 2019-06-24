'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import time

from models import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='cifar10')
    parser.add_argument('--GPU',help="whether use GPU")
    parser.add_argument('--model',default="resnet" , help="choose which model to build")
    parser.add_argument('--epoch',default="200" , help="epoch number")
    parser.add_argument('--learningRate',default="0.1" , help="epoch number")
    parser.add_argument('--evaluation',default="False" , help="evaluation or not")


    args=parser.parse_args()
    
    epoch_number = int(args.epoch)
    learning_rate = float(args.learningRate)

    if args.GPU == 'True' and torch.cuda.is_available():
        use_GPU = True
    else:
        use_GPU = False

    print("USE GPU:", use_GPU)

    if args.evaluation == 'False':
        evaluation = False
        print("Dont evaluation")
    else:
        evaluation = True
        print("Evaluation during training and testing after training")

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    print('==> Building model..')
    if args.model == 'resnet':
        net = ResNet18()
    elif args.model == 'vgg':
        net = VGG('VGG19')
    elif args.model == 'googlenet':
        net = GoogLeNet()
    elif args.model == 'densenet':
        net = DenseNet121()
    elif net == 'mobilenet':
        net = MobileNet()
    elif net == 'mobilenet2':
        net = MobileNetV2()
    else:
        raise ValueError("must input a right model number, for example:resnet, vgg, googlenet, densenet,mobilenet,mobilenet2")

    # net = PreActResNet18()
    # net = ResNeXt29_2x64d()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    # net = ShuffleNetV2(1)
    # net = EfficientNetB0()

    # print(net)
    if use_GPU:
        net.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)


    print("==> Start training..")
    t_start = time.time()
    print("Start Training time:{}".format(time.asctime(time.localtime(time.time()))))

    for epoch in range(epoch_number):
        print("epoch:{}, time: {}".format(epoch, time.asctime(time.localtime(time.time()))))
        net.train()
        for step, (b_x, b_y) in enumerate(trainloader):
            if use_GPU:
                b_x = b_x.cuda()
                b_y = b_y.cuda()

            output = net(b_x)
            loss = criterion(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if evaluation:
            if epoch % 10 == 0:
                net.eval()
                correct = 0
                total = 0
                for step, (b_x, b_y) in enumerate(testloader):
                    if use_GPU:
                        b_x = b_x.cuda()
                        b_y = b_y.cuda()
                    testoutput = net(b_x)

                    if use_GPU:
                        pre_y = torch.max(testoutput, 1)[1].cuda().data.squeeze()
                    else:
                        pre_y = torch.max(testoutput, 1)[1].data.squeeze()

                    right = torch.sum(pre_y==b_y).type(torch.FloatTensor)
                    total += b_y.size()[0]
                    correct += right
                
                print("Epoch {}, Accuracy:{}".format(epoch, correct/total))
    t_end = time.time()
    print("Finish Training")
    print("Time consuming: {} seconds".format(t_end - t_start))
    