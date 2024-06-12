import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import argparse
import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import os
from collections import OrderedDict
from copy import deepcopy
from torch.autograd import Variable

from tqdm import tqdm
from .method import FSCLIPmethod
from .utils import *

def calculate_lr_alpha(features, clip_weights):
    # lr_alpha
    ftT = features @ clip_weights
    temp = torch.sum(torch.pow(ftT, 2),dim = 0)
    max_sum = max(temp)
    lr_alpha = features.shape[0] / (max_sum * 4)
    return lr_alpha

def calculate_init_alpha(features, labels, shots, clip_weights):
    # init_alpha
    alpha_tilde = compute_centroids_alpha((features @ clip_weights).unsqueeze(0), labels.unsqueeze(0))[0]
    alpha_tilde = alpha_tilde.double() * shots
    alpha_init = 250 / shots * alpha_tilde
    final_init_alpha_mean = torch.mean(alpha_init)
    return final_init_alpha_mean

def calculate_lr_w(features):
    # lr_w
    ff_t = features.T @ features
    ff_t_np = ff_t.cpu().numpy()
    w, v = eigh(ff_t_np)
    max_eigen = max(w) # check the iters of power iteration
    lr_w =  (4 * features.shape[0]) / max_eigen
    return lr_w


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def weight_average(alltheweights):
    K = len(alltheweights)
    avg_state_dict = OrderedDict()
    for param_name, param in alltheweights[0].items():
        avg_param = sum(sd[param_name] for sd in alltheweights)
        avg_param /= K
        avg_state_dict[param_name] = avg_param
    return avg_state_dict


class LinearProbe_P2(FSCLIPmethod):
    '''
    LP++: Linear Probe method with learning weight term alpha by gradient descent
    '''

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.epoch = args['train_epoch']
        self.shot = args['shots']
        self.dataset = args['dataset']
        self.output_dir = args['output_dir']

    def forward(self,
                train_loader: torch.utils.data.DataLoader,
                val_loader: torch.utils.data.DataLoader,
                test_features: torch.tensor,
                test_labels: torch.tensor,
                text_weights: torch.tensor,
                model: nn.Module,
                classnames):
        """
        inputs:
            train_loader : torch.utils.data.DataLoader
            test_features : torch.Tensor of shape [test_data_size, 1024]
            test_labels : torch.Tensor of shape [test_data_size]
            text_weights : torch.Tensor of shape [num_shot*num_classes, 1024]
        """

        # Feature Extraction for Training
        print("\nExtracting visual features and labels from train set.")
        features, labels = [], []
        with torch.no_grad():
            for i, (images, target) in enumerate(tqdm(train_loader)):
                images, target = images.cuda(), target.cuda()
                image_features = model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                features.append(image_features)
                labels.append(target)  
        features, labels = torch.cat(features), torch.cat(labels)
        
        # Feature Extraction for Validation
        print("\nExtracting visual features and labels from val set.")
        val_features, val_labels = [], []
        with torch.no_grad():
            for i, (images, target) in enumerate(tqdm(val_loader)):
                images, target = images.cuda(), target.cuda()
                image_features = model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                val_features.append(image_features)
                val_labels.append(target)
        val_features, val_labels = torch.cat(val_features), torch.cat(val_labels)
        

        centroids = compute_centroids(features.unsqueeze(0), labels.unsqueeze(0))  # [batch, num_class, d]
        
        classifier = nn.Linear(features.shape[1], int(features.shape[0]/self.shot),bias=True).to(model.dtype).cuda()
        classifier.weight.data = centroids[0]


        print('Running LP++')
        # lr_w
        lr_temp = calculate_lr_w(features)

        # init_alpha
        if len(text_weights.shape)==3:
            alpha_vec = []
            self.lr_alpha = []
            for i in range(len(text_weights)):
                final_init_alpha_mean = calculate_init_alpha(features, labels, self.shot, text_weights[i])
                alpha_vec.append(Variable(
                    final_init_alpha_mean * torch.ones(1, int(features.shape[0] / self.shot)).to(model.dtype).cuda(),
                    requires_grad=True))

                # lr_alpha
                self.lr_alpha.append(calculate_lr_alpha(features, text_weights[i]))
        else:
            final_init_alpha_mean= calculate_init_alpha(features, labels, self.shot, text_weights)

            alpha_vec = Variable(final_init_alpha_mean * torch.ones(1, int(features.shape[0]/self.shot)).to(model.dtype).cuda(), requires_grad=True)

            # lr_alpha
            self.lr_alpha = calculate_lr_alpha(features, text_weights)

        optimizer = torch.optim.SGD(classifier.parameters(), lr_temp, momentum=0.9)
 
 
        # Train
        print('\nStart Training procedure!')

        if len(text_weights.shape)==3:
            model_state, optimizer_state = copy_model_and_optimizer(model, optimizer)
            best_acc, best_epoch = 0.0, 0
            for epoch in range(self.epoch//2):
                alltheweights = []
                for i in range(len(text_weights)):
                    if epoch == 0:
                        load_model_and_optimizer(model, optimizer, model_state, optimizer_state)
                    else:
                        model.load_state_dict(avg_state_dict, strict=False)
                    # print('Running model for epoch: {}'.format(epoch))
                    for k in range(2):
                        classifier.train()
                        vision_logits = classifier(features)
                        text_logits = features @ text_weights[i]
                        logits = vision_logits + torch.ones(features.shape[0], 1).to(
                            model.dtype).cuda() @ alpha_vec[i] * text_logits
                        loss = F.cross_entropy(logits, labels)
                        acc = np.mean(logits.argmax(dim=1).cpu().numpy() == labels.cpu().numpy()) * 100.0

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        # # update for alpha
                        if (epoch + 1) % 10 == 0:
                            alpha_vec[i].data -= self.lr_alpha[i] * alpha_vec[i].grad.data

                        classifier.eval()
                        vision_logits_val = classifier(val_features)
                        text_logits_val = val_features.detach() @ text_weights[i]
                        logits_val = vision_logits_val + torch.ones(val_features.shape[0], 1).to(
                            model.dtype).cuda() @ alpha_vec[i] * text_logits_val
                        acc_val = np.mean(logits_val.argmax(dim=1).cpu().numpy() == val_labels.cpu().numpy()) * 100.0
                        # print('The accuracy for val data is ',acc_val)

                        if acc_val >= best_acc:
                            best_acc = acc_val
                            best_epoch = epoch
                            vision_logits_test = classifier(test_features)
                            text_logits_test = test_features.detach() @ text_weights[i]
                            logits_test = vision_logits_test + torch.ones(test_features.shape[0], 1).to(
                                model.dtype).cuda() @ alpha_vec[i] * text_logits_test
                            acc_test = np.mean(logits_test.argmax(dim=1).cpu().numpy() == test_labels.cpu().numpy()) * 100.0
                    weights = {}
                    for name, module in model.named_modules():
                        if isinstance(module, nn.LayerNorm):
                            for nparam, p in module.named_parameters():
                                if nparam in ['weight', 'bias']:
                                    weights[f"{name}.{nparam}"] = deepcopy(p)
                    alltheweights.append(weights)
                avg_state_dict = weight_average(alltheweights)
                model.load_state_dict(avg_state_dict, strict=False)
        else:
            best_acc, best_epoch = 0.0, 0
            for epoch in range(self.epoch):

                # print('Running model for epoch: {}'.format(epoch))
                classifier.train()
                vision_logits = classifier(features)
                text_logits = features @ text_weights
                logits = vision_logits + torch.ones(features.shape[0],1).to(model.dtype).cuda() @ alpha_vec * text_logits
                loss = F.cross_entropy(logits, labels)
                acc = np.mean(logits.argmax(dim=1).cpu().numpy() ==  labels.cpu().numpy()) * 100.0

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # # update for alpha
                if (epoch + 1) % 10 == 0:
                    alpha_vec.data -= self.lr_alpha * alpha_vec.grad.data

                classifier.eval()
                vision_logits_val = classifier(val_features)
                text_logits_val = val_features.detach() @ text_weights
                logits_val = vision_logits_val + torch.ones(val_features.shape[0], 1).to(model.dtype).cuda() @ alpha_vec * text_logits_val
                acc_val = np.mean(logits_val.argmax(dim=1).cpu().numpy() ==  val_labels.cpu().numpy()) * 100.0
                # print('The accuracy for val data is ',acc_val)

                if acc_val >= best_acc:
                    best_acc = acc_val
                    best_epoch = epoch
                    vision_logits_test = classifier(test_features)
                    text_logits_test = test_features.detach() @ text_weights
                    logits_test = vision_logits_test + torch.ones(test_features.shape[0], 1).to(model.dtype).cuda() @ alpha_vec * text_logits_test
                    acc_test = np.mean(logits_test.argmax(dim=1).cpu().numpy() ==  test_labels.cpu().numpy()) * 100.0
                    # print('The accuracy for test data is ',acc_test)
                    # torch.save(classifier, self.output_dir + "/best_lp_model_" + str(self.shot) + "shots.pt")

                    # Evaluation

            # classifier = torch.load(self.output_dir + "/best_lp_model_" + str(self.shot) + "shots.pt")
            # vision_logits_test = classifier(test_features)
            # text_logits_test = test_features.detach() @ text_weights
            # logits_test = vision_logits_test + torch.ones(test_features.shape[0], 1).to(model.dtype).cuda() @ alpha_vec * text_logits_test
            # acc_test = np.mean(logits_test.argmax(dim=1).cpu().numpy() ==  test_labels.cpu().numpy()) * 100.0
            # print('The accuracy for test data is ',acc_test)
                
        return loss.item(), acc_test

