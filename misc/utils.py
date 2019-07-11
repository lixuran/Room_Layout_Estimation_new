import torch
import torch.nn as nn
from collections import OrderedDict
import math 
#import PIL
def do(image):
            # Get size before we rotate
            x = image.size[0]
            y = image.size[1]

            # Rotate, while expanding the canvas size
            image = image.rotate(rotation, expand=True, resample=Image.BICUBIC)

            # Get size after rotation, which includes the empty space
            X = image.size[0]
            Y = image.size[1]

            # Get our two angles needed for the calculation of the largest area
            angle_a = abs(rotation)
            angle_b = 90 - angle_a

            # Python deals in radians so get our radians
            angle_a_rad = math.radians(angle_a)
            angle_b_rad = math.radians(angle_b)

            # Calculate the sins
            angle_a_sin = math.sin(angle_a_rad)
            angle_b_sin = math.sin(angle_b_rad)

            # Find the maximum area of the rectangle that could be cropped
            E = (math.sin(angle_a_rad)) / (math.sin(angle_b_rad)) * \
                (Y - X * (math.sin(angle_a_rad) / math.sin(angle_b_rad)))
            E = E / 1 - (math.sin(angle_a_rad) ** 2 / math.sin(angle_b_rad) ** 2)
            B = X - E
            A = (math.sin(angle_a_rad) / math.sin(angle_b_rad)) * B

            # Crop this area from the rotated image
            # image = image.crop((E, A, X - E, Y - A))
            image = image.crop((int(round(E)), int(round(A)), int(round(X - E)), int(round(Y - A))))

            # Return the image, re-sized to the size of the image passed originally
            return image.resize((x, y), resample=Image.BICUBIC)

            

def group_weight(module):
    # Group module parameters into two group
    # One need weight_decay and the other doesn't
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.GroupNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    return [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]


def adjust_learning_rate(optimizer, args):
    if args.cur_iter < args.warmup_iters:
        frac = args.cur_iter / args.warmup_iters
        step = args.lr - args.warmup_lr
        args.running_lr = args.warmup_lr + step * frac
    else:
        frac = (float(args.cur_iter) - args.warmup_iters) / (args.max_iters - args.warmup_iters)
        scale_running_lr = max((1. - frac), 0.) ** args.lr_pow
        args.running_lr = args.lr * scale_running_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = args.running_lr


def save_model(net, path, epoch):
    state_dict = OrderedDict({
        'epoch': epoch,
        'kwargs': {
            'backbone': net.backbone,
            'use_rnn': net.use_rnn,
        },
        'state_dict': net.state_dict(),
    })
    torch.save(state_dict, path)


def load_trained_model(Net, path):
    state_dict = torch.load(path)
    net = Net(**state_dict['kwargs'])
    net.load_state_dict(state_dict['state_dict'])
    return net
