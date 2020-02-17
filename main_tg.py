import sys
import os
import torch
import random
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import math
from network_tg import GUNet
from mlp_dropout import MLPClassifier
from sklearn import metrics
from util import cmd_args, sep_tg_data
import os.path as osp

from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torchsummary import summary

sys.path.append(
    '%s/pytorch_structure2vec-master/s2v_lib' % os.path.dirname(
        os.path.realpath(__file__)))


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        model = GUNet

        # print("latent dim is ", cmd_args.latent_dim)

        self.s2v = model(
            latent_dim=cmd_args.latent_dim,
            output_dim=cmd_args.out_dim,
            num_node_feats=cmd_args.feat_dim,
            num_edge_feats=0,
            k=cmd_args.sortpooling_k)

        # print("num_node_feats: ", cmd_args.feat_dim)
        out_dim = cmd_args.out_dim
        if out_dim == 0:
            out_dim = self.s2v.dense_dim

        # print("out dim is ", out_dim)
        self.mlp = MLPClassifier(
            input_size=out_dim, hidden_size=cmd_args.hidden,
            num_class=cmd_args.num_class, with_dropout=cmd_args.dropout)

    def forward(self, data):
        # node_feat, labels = self.PrepareFeatureLabel(batch_graph)
        labels = data.y

        embed = self.s2v(data)
        return self.mlp(embed, labels)

    def output_features(self, data):
        embed = self.s2v(data)
        labels = data.y

        return embed, labels


def loop_dataset(dataloader, classifier, optimizer=None, device=torch.device('cpu')):
    total_loss = []
    # total_iters = (len(sample_idxes) + (bsize - 1) * (optimizer is None)) // bsize # noqa
    total_iters = len(dataloader)
    pbar = tqdm(range(total_iters), unit='batch')
    all_targets = []
    all_scores = []

    n_samples = 0
    dataloader_iterator = iter(dataloader)
    for pos in pbar:
        data = next(dataloader_iterator)

        # Deal with the data with no node attributes
        if 'x' not in data.keys:
            data.x = torch.ones(data.num_nodes, 1)

        data = data.to(device)
        num_selected = data.batch.max().item() + 1
        targets = data.y
        all_targets += targets.tolist()
        logits, loss, acc = classifier(data)

        # print("Preds: ")
        # print(logits)
        # print("Targets: ")
        # print(targets)
        all_scores.append(logits[:, 1].detach())  # for binary classification

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = loss.data.cpu().numpy()
        pbar.set_description('loss: %0.5f acc: %0.5f' % (loss, acc))

        total_loss.append(np.array([loss, acc]) * num_selected)

        n_samples += num_selected

    total_loss = np.array(total_loss)
    avg_loss = np.sum(total_loss, 0) / n_samples
    all_scores = torch.cat(all_scores).cpu().numpy()

    # np.savetxt('test_scores.txt', all_scores)  # output test predictions

    all_targets = np.array(all_targets)
    fpr, tpr, _ = metrics.roc_curve(all_targets, all_scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    avg_loss = np.concatenate((avg_loss, [auc]))

    return avg_loss

def count_parameters(model):
    total_param = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_param = np.prod(param.size())
            if param.dim() > 1:
                print(name, ':', 'x'.join(str(x) for x in list(param.size())), '=', num_param)
            else:
                print(name, ':', num_param)
            total_param += num_param
    return total_param


if __name__ == '__main__':
    print(cmd_args)
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # device = torch.device('cpu')

    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', cmd_args.data)
    dataset = TUDataset(path, name=cmd_args.data)
    dataset = dataset.shuffle()

    if cmd_args.sortpooling_k <= 1:
        num_nodes_list = sorted([
            g.num_nodes for g in dataset])
        cmd_args.sortpooling_k = num_nodes_list[
            int(math.ceil(cmd_args.sortpooling_k * len(num_nodes_list))) - 1]
        cmd_args.sortpooling_k = max(10, cmd_args.sortpooling_k)
        print('k used in SortPooling is: ' + str(cmd_args.sortpooling_k))

    # Ten Folds validation
    train_dataset, test_dataset = sep_tg_data(dataset, cmd_args.fold)
    print('# train: %d, # test: %d' % (len(train_dataset), len(test_dataset)))
    print('# num of classes: ', dataset.num_classes)

    test_loader = DataLoader(test_dataset, batch_size=cmd_args.batch_size)
    train_loader = DataLoader(train_dataset, batch_size=cmd_args.batch_size)
    cmd_args.feat_dim = dataset.num_node_features
    cmd_args.num_class = dataset.num_classes
    if cmd_args.feat_dim == 0:
        cmd_args.feat_dim = 1

    classifier = Classifier().to(device)
    print("Number of Model Parameters: ", count_parameters(classifier))

    optimizer = optim.Adam(
        classifier.parameters(), lr=cmd_args.learning_rate, amsgrad=True,
        weight_decay=0.0008)

    # train_idxes = list(range(len(train_graphs)))
    best_loss = None
    max_acc = 0.0
    for epoch in range(cmd_args.num_epochs):
        # random.shuffle(train_idxes)
        classifier.train()
        avg_loss = loop_dataset(train_loader, classifier, optimizer=optimizer, device=device)
        if not cmd_args.printAUC:
            avg_loss[2] = 0.0
        print('\033[92maverage training of epoch %d: loss %.5f acc %.5f auc %.5f\033[0m'
              % (epoch, avg_loss[0], avg_loss[1], avg_loss[2])) # noqa

        classifier.eval()
        test_loss = loop_dataset(test_loader, classifier, device=device)
        if not cmd_args.printAUC:
            test_loss[2] = 0.0
        print('\033[93maverage test of epoch %d: loss %.5f acc %.5f auc %.5f\033[0m'
              % (epoch, test_loss[0], test_loss[1], test_loss[2])) # noqa
        max_acc = max(max_acc, test_loss[1])

    with open('acc_result_tg_%s.txt' % cmd_args.data, 'a+') as f:
        # f.write(str(test_loss[1]) + '\n')
        f.write(str(max_acc) + '\n')

    if cmd_args.printAUC:
        with open('auc_results_tg.txt', 'a+') as f:
            f.write(str(test_loss[2]) + '\n')

    # if cmd_args.extract_features:
    #     features, labels = classifier.output_features(train_graphs)
    #     labels = labels.type('torch.FloatTensor')
    #     np.savetxt('extracted_features_train.txt', torch.cat(
    #         [labels.unsqueeze(1), features.cpu()], dim=1).detach().numpy(),
    #             '%.4f')
    #     features, labels = classifier.output_features(test_graphs)
    #     labels = labels.type('torch.FloatTensor')
    #     np.savetxt('extracted_features_test.txt', torch.cat(
    #         [labels.unsqueeze(1), features.cpu()], dim=1).detach().numpy(),
    #             '%.4f')
