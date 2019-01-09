import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from model import TCN
from preprocess import data_generator
from preprocess import stage_dataset, batchify, stage_overlapping_dataset
import numpy as np
import time
from cnn import ConvNet


parser = argparse.ArgumentParser(description='Sequence Modeling - Polyphonic Music')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.25,
                    help='dropout applied to layers (default: 0.25)')
parser.add_argument('--clip', type=float, default=0.2,
                    help='gradient clip, -1 means no clip (default: 0.2)')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit (default: 100)')
parser.add_argument('--ksize', type=int, default=2,
                    help='kernel size (default: 5)')
parser.add_argument('--levels', type=int, default=2,
                    help='# of levels (default: 4)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate (default: 1e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--nhid', type=int, default=512,
                    help='number of hidden units per layer (default: 150)')
parser.add_argument('--data', type=str, default='fold_benchmark',
                    help='the dataset to run (default: MAPS_fold_1)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')


args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

print(args)
input_size = 768
output_size = 88
#X_train, X_valid, X_test = data_generator(args.data)
train_features, train_labels, valid_features, valid_labels, test_features, test_labels = stage_overlapping_dataset(args.data)



n_channels = [args.nhid] * args.levels
kernel_size = args.ksize
dropout = args.dropout
loss_scale = 10000.0

#model = torch.load(open("piano_transcription_fold_benchmark.pt", "rb"))

conv_model = ConvNet()

model = TCN(input_size, output_size, n_channels, kernel_size, dropout=args.dropout)



if args.cuda:
    model.cuda()
    conv_model.cuda()

#criterion = nn.CrossEntropyLoss()
lr = args.lr
optimizer = getattr(optim, args.optim)(list(model.parameters())+list(conv_model.parameters()), lr=lr)


def evaluate(X_data, Y_data):
    model.eval()
    eval_idx_list = np.arange(len(X_data), dtype="int32")
    total_loss = 0.0
    count = 0
    total_p = 0.0
    total_r = 0.0
    total_f1 = 0.0
    total_a = 0.0
    X_train_batch = batchify(X_data, eval_idx_list, 64)
    Y_train_batch = batchify(Y_data, eval_idx_list, 64)
    for idx in eval_idx_list:

        #data_line = X_data[idx]
        #x, y = Variable(data_line[:-1]), Variable(data_line[1:])
        x = Variable(X_train_batch[idx])
        y = Variable(Y_train_batch[idx])

        if args.cuda:
            x, y = x.cuda(), y.cuda()
        conv_output = conv_model(x.unsqueeze(1))
        output = model(conv_output)
        #output = model(x.unsqueeze(0)).squeeze(0)
        #loss = -torch.trace(torch.matmul(y, torch.log(output).float().t()) +
        #                    torch.matmul((1-y), torch.log(1-output).float().t()))

        #loss = log_loss(y, torch.clamp(output, 1e-7, 1.0 - 1e-7)) * loss_scale
        loss = log_loss(y, output) * loss_scale
        tp, fp, tn, fn = eval_framewise(output, y)
        p, r, f1, a = prf_framewise(tp, fp, tn, fn)
        total_p += p
        total_r += r
        total_f1 += f1
        total_a += a
        total_loss += loss.item()
        count += output.size(0)

    steps = len(eval_idx_list)

    mean_p = total_p / steps
    mean_r = total_r / steps
    mean_f1 = total_f1 / steps
    mean_a = total_a / steps
    eval_loss = total_loss / count
    print("Validation/Test loss: {:.5f}".format(eval_loss))
    print("P: {:.5f} / R: {:.5f} / F1: {:.5f} / A: {:.5f}".format(mean_p, mean_r, mean_f1, mean_a))
    return eval_loss


def train(ep):
    model.train()
    total_loss = 0
    count = 0
    # dont use last example, since it might be shorter than the others -> see split func in stage_dataset
    shuffle_idx_list = np.arange(len(train_features), dtype="int32")

    np.random.shuffle(shuffle_idx_list)
    X_train_batch = batchify(train_features, shuffle_idx_list, 64)
    Y_train_batch = batchify(train_labels, shuffle_idx_list, 64)
    print(X_train_batch[1].shape)

    train_idx_list = np.arange(len(X_train_batch), dtype="int32")
    t0 = time.time()
    for idx in train_idx_list:
        #data_line = X_train[idx]
        #print(train_features[idx].size())
        #x, y = Variable(data_line[:-1]), Variable(data_line[1:])
        x = Variable(X_train_batch[idx])
        y = Variable(Y_train_batch[idx])


        if args.cuda:
            x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()
        conv_output = conv_model(x.unsqueeze(1))
        output = model(conv_output)
        #loss = -torch.trace(torch.matmul(y, torch.log(output).float().t()) +
        #                    torch.matmul((1 - y), torch.log(1 - output).float().t()))
        #loss = log_loss(y, torch.clamp(output, 1e-7, 1.0-1e-7)) * loss_scale
        loss = log_loss(y, output) * loss_scale
        total_loss += loss.item()
        count += output.size(0)

        if args.clip > 0:
            torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        loss.backward()
        optimizer.step()
        if idx > 0 and idx % args.log_interval == 0:
            cur_loss = total_loss / count
            print("Epoch {:2d} | lr {:.5f} | loss {:.5f} | elapsed time {:.2f} seconds".format(ep, lr, cur_loss, time.time() - t0))
            t0 = time.time()
            total_loss = 0.0
            count = 0


def log_loss(labels, predictions, epsilon=1e-7, weights=None):
    """Calculate log loss.
        Args:
            labels: The ground truth output tensor, same dimensions as 'predictions'.
            predictions: The predicted outputs.
            epsilon: A small increment to add to avoid taking a log of zero.
            weights: Weights to apply to labels.
        Returns:
            A `Tensor` representing the loss values.
    """

    losses = -torch.mul(labels, torch.log(predictions + epsilon).float()) - torch.mul(
        (1 - labels), torch.log(1 - predictions + epsilon).float())
    if weights is not None:
        losses = torch.mul(losses, weights)

    return torch.mean(losses)


def eval_framewise(predictions, targets, thresh=0.5):
    """

    """
    if predictions.shape != targets.shape:
        raise ValueError('predictions.shape {} != targets.shape {} !'.format(predictions.shape, targets.shape))

    pred = predictions > thresh
    targ = targets > thresh

    tp = pred & targ
    fp = pred ^ tp
    fn = targ ^ tp

    # tp, fp, tn, fn
    return tp.sum(), fp.sum(), 0, fn.sum()


def prf_framewise(tp, fp, tn, fn):
    tp, fp, tn, fn = float(tp), float(fp), float(tn), float(fn)

    if tp + fp == 0.:
        p = 0.
    else:
        p = tp / (tp + fp)

    if tp + fn == 0.:
        r = 0.
    else:
        r = tp / (tp + fn)

    if p + r == 0.:
        f = 0.
    else:
        f = 2 * ((p * r) / (p + r))

    if tp + fp + fn == 0.:
        a = 0.
    else:
        a = tp / (tp + fp + fn)

    return p, r, f, a


if __name__ == "__main__":
    best_vloss = 1e8
    vloss_list = []
    model_name = "piano_transcription_{0}.pt".format(args.data)
    #vloss = evaluate(valid_features, valid_labels)
    for ep in range(1, args.epochs+1):
        train(ep)
        print("Evaluating!")
        vloss = evaluate(valid_features, valid_labels)
        #print("Testing!")
        #tloss = evaluate(test_features, test_labels)
        if vloss < best_vloss:
            with open(model_name, "wb") as f:
                torch.save(model, f)
                print("Saved model!\n")
            best_vloss = vloss
        if ep > 5 and vloss > max(vloss_list[-3:]):
        #if ep % 5 == 0:
            lr /= 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        vloss_list.append(vloss)

    print('-' * 89)
    model = torch.load(open(model_name, "rb"))
    tloss = evaluate(test_features, test_labels)
