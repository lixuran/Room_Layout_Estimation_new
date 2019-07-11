import os
import sys
import glob
import json
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.ndimage.filters import maximum_filter
from shapely.geometry import Polygon

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import HorizonNet
from dataset import visualize_a_data
from misc import post_proc, panostretch, utils
from misc import pano_lsd_align

def find_N_peaks(signal, r=29, min_v=0.05, N=None):
    max_v = maximum_filter(signal, size=r, mode='wrap')
    pk_loc = np.where(max_v == signal)[0]
    pk_loc = pk_loc[signal[pk_loc] > min_v]
    if N is not None:
        order = np.argsort(-signal[pk_loc])
        pk_loc = pk_loc[order[:N]]
        pk_loc = pk_loc[np.argsort(pk_loc)]
    return pk_loc, signal[pk_loc]


def augment(x_img, flip, rotate,max_x_rotate,max_y_rotate,min_x_rotate,min_y_rotate,x_rotate_prob,y_rotate_prob):
    x_img = x_img.numpy()
    #aug_type = ['']
    #x_imgs_augmented = [x_img]
    aug_type = []
    x_imgs_augmented = []
    if flip:
        aug_type.append('flip')
        x_imgs_augmented.append(np.flip(x_img, axis=-1))
    for shift_p in rotate:
        shift = int(round(shift_p * x_img.shape[-1]))
        aug_type.append('rotate %d' % shift)
        x_imgs_augmented.append(np.roll(x_img, shift, axis=-1))
    if (max_x_rotate >0 and max_x_rotate > min_x_rotate) :
            rnd = np.random.uniform()
            if(rnd> x_rotate_prob):
                rnd_x = np.random.randint(min_x_rotate,max_x_rotate)
                #print(x_img[0].shape)
                xx_img=np.transpose(x_img[0],[1, 2, 0])
               
                x_copy= x_img.copy()
                x_copy[0]=np.transpose(pano_lsd_align.rotatePanorama_degree(xx_img,rnd_x,axis=1),[2,0,1])
                x_imgs_augmented.append(x_copy)
                aug_type.append('xrotate %d' % rnd_x)
    if (max_y_rotate >0 and max_y_rotate > min_y_rotate) :
            rnd = np.random.uniform()
            if(rnd> y_rotate_prob):
                #print(x_img.shape)
                rnd_y = np.random.randint(min_y_rotate,max_y_rotate)
                xx_img=np.transpose(x_img[0],[1, 2, 0])
                x_copy= x_img.copy()
                x_copy[0]=np.transpose(pano_lsd_align.rotatePanorama_degree(xx_img,rnd_y,axis=2),[2,0,1])
                x_imgs_augmented.append(x_copy)
                aug_type.append('yrotate %d' % rnd_y)
    return torch.FloatTensor(np.concatenate(x_imgs_augmented, 0)), aug_type


def augment_undo(x_imgs_augmented, aug_type):
    #print("x imgs augmented shape {}".format(x_imgs_augmented.shape))
    x_imgs_augmented = x_imgs_augmented.cpu().numpy()
    sz = x_imgs_augmented.shape[0] // len(aug_type)
    x_imgs = []
    for i, aug in enumerate(aug_type):
        x_img = x_imgs_augmented[i*sz : (i+1)*sz]
        #print(x_img.shape)
        if aug == 'flip':
            x_imgs.append(np.flip(x_img, axis=-1))
        elif aug.startswith('rotate'):
            shift = int(aug.split()[-1])
            x_imgs.append(np.roll(x_img, -shift, axis=-1))
        elif aug.startswith('xrotate'):
            rnd_x = int(aug.split()[-1])
            if(x_img.shape[1]>1):  # boundary prediction
                x_copy= x_img.copy()
                #print(x_img[0])
                x_copy[0]=pano_lsd_align.rotateBoundaries(x_img[0],degree=-rnd_x,axis=1,coorW=1024,coorH=512)
                #print(x_copy[0])
                x_imgs.append(x_img)
            else:
                x_copy= x_img.copy()
                x_copy[0]=pano_lsd_align.rotatePredictedCorners(x_img[0],-rnd_x,axis=1)[:,1].reshape(1,-1)
                x_imgs.append(x_img)
        elif aug.startswith('yrotate'):
            rnd_y = int(aug.split()[-1])
            #print(x_img.shape)
            if(x_img.shape[1]>1):
                x_copy= x_img.copy()
                x_copy[0]=pano_lsd_align.rotateBoundaries(x_img[0],degree=-rnd_y,axis=2,coorW=1024,coorH=512)
                x_imgs.append(x_img)
            else:
                x_copy= x_img.copy()
                x_copy[0]=pano_lsd_align.rotatePredictedCorners(x_img[0],-rnd_y,axis=2)[:,1].reshape(1,-1)
                x_imgs.append(x_img)
        elif aug == '':
            x_imgs.append(x_img)
        else:
            raise NotImplementedError()
    
    return np.array(x_imgs)


def inference(net, x, device, flip=False, rotate=[], visualize=False,
              force_cuboid=True, min_v=None, r=0.05,max_x_rotate=0,max_y_rotate=0,min_x_rotate=0,min_y_rotate=0,x_rotate_prob=1,y_rotate_prob=1):
    '''
    net   : the trained HorizonNet
    x     : tensor in shape [1, 3, 512, 1024]
    flip  : fliping testing augmentation
    rotate: horizontal rotation testing augmentation
    '''

    H, W = tuple(x.shape[2:])

    # Network feedforward (with testing augmentation)
    x, aug_type = augment(x, args.flip, args.rotate,max_x_rotate,max_y_rotate,min_x_rotate,min_y_rotate,x_rotate_prob,y_rotate_prob)
    #apply some distortion here.
    
    y_bon_, y_cor_ = net(x.to(device))
    #print(y_bon_.shape)
    #print(y_cor_.shape)
    #print()
    
    y_bon_ = augment_undo(y_bon_.cpu(), aug_type).mean(0)
    y_cor_ = augment_undo(torch.sigmoid(y_cor_).cpu(), aug_type).mean(0)
    #print(y_bon_.shape)
    #print() 
    # Visualize raw model output
    if visualize:
        vis_out = visualize_a_data(x[0],
                                   torch.FloatTensor(y_bon_[0]),
                                   torch.FloatTensor(y_cor_[0]))
    else:
        vis_out = None

    y_bon_ = (y_bon_[0] / np.pi + 0.5) * H - 0.5
    y_cor_ = y_cor_[0, 0]

    # Init floor/ceil plane
    z0 = 50
    _, z1 = post_proc.np_refine_by_fix_z(*y_bon_, z0)

    # Detech wall-wall peaks
    if min_v is None:
        min_v = 0 if force_cuboid else 0.05
    r = int(round(W * r / 2))
    N = 4 if force_cuboid else None
    xs_ = find_N_peaks(y_cor_, r=r, min_v=min_v, N=N)[0]

    # Generate wall-walls
    cor, xy_cor = post_proc.gen_ww(xs_, y_bon_[0], z0, tol=abs(0.16 * z1 / 1.6), force_cuboid=force_cuboid)
    if not force_cuboid:
        # Check valid (for fear self-intersection)
        xy2d = np.zeros((len(xy_cor), 2), np.float32)
        for i in range(len(xy_cor)):
            xy2d[i, xy_cor[i]['type']] = xy_cor[i]['val']
            xy2d[i, xy_cor[i-1]['type']] = xy_cor[i-1]['val']
        if not Polygon(xy2d).is_valid:
            print(
                'Fail to generate valid general layout!! '
                'Generate cuboid as fallback.',
                file=sys.stderr)
            xs_ = find_N_peaks(y_cor_, r=r, min_v=0, N=4)[0]
            cor, xy_cor = post_proc.gen_ww(xs_, y_bon_[0], z0, tol=abs(0.16 * z1 / 1.6), force_cuboid=True)

    # Expand with btn coory
    cor = np.hstack([cor, post_proc.infer_coory(cor[:, 1], z1 - z0, z0)[:, None]])

    # Collect corner position in equirectangular
    cor_id = np.zeros((len(cor)*2, 2), np.float32)
    for j in range(len(cor)):
        cor_id[j*2] = cor[j, 0], cor[j, 1]
        cor_id[j*2 + 1] = cor[j, 0], cor[j, 2]

    # Normalized to [0, 1]
    cor_id[:, 0] /= W
    cor_id[:, 1] /= H

    return cor_id, z0, z1, vis_out


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--pth', required=True,
                        help='path to load saved checkpoint.')
    parser.add_argument('--img_glob', required=True,
                        help='NOTE: Remeber to quote your glob path. '
                             'All the given images are assumed to be aligned'
                             'or you should use preporcess.py to do so.')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--visualize', action='store_true')
    # Augmentation related
    parser.add_argument('--flip', action='store_true',
                        help='whether to perfome left-right flip. '
                             '# of input x2.')
    parser.add_argument('--rotate', nargs='*', default=[], type=float,
                        help='whether to perfome horizontal rotate. '
                             'each elements indicate fraction of image width. '
                             '# of input xlen(rotate).')
    # Post-processing realted
    parser.add_argument('--r', default=0.05, type=float)
    parser.add_argument('--min_v', default=None, type=float)
    parser.add_argument('--relax_cuboid', action='store_true')
    parser.add_argument('--max_x_rotate', default=0, type=int,
                        help='maximum degree of y axis rotation')
    parser.add_argument('--min_x_rotate',  default=0, type=int,
                        help='minimum degree of y axis rotation')
    parser.add_argument('--max_y_rotate',  default=0, type=int,
                        help='maximum degree of z axis rotation')
    parser.add_argument('--min_y_rotate',  default=0, type=int,
                        help='minimum degree of z axis rotation')
    parser.add_argument('--y_rotate_prob',  default=1, type=float,
                        help='probability of z axis rotation')
    parser.add_argument('--x_rotate_prob',  default=1, type=float,
                        help='probability of y axis rotation')
    # Misc arguments
    parser.add_argument('--no_cuda', action='store_true',
                        help='disable cuda')
    args = parser.parse_args()

    # Prepare image to processed
    paths = sorted(glob.glob(args.img_glob))
    if len(paths) == 0:
        print('no images found')
    for path in paths:
        assert os.path.isfile(path), '%s not found' % path

    # Check target directory
    if not os.path.isdir(args.output_dir):
        print('Output directory %s not existed. Create one.' % args.output_dir)
        os.makedirs(args.output_dir)
    device = torch.device('cpu' if args.no_cuda else 'cuda')

    # Loaded trained model
    net = utils.load_trained_model(HorizonNet, args.pth).to(device)
    net.eval()

    # Inferencing
    with torch.no_grad():
        for i_path in tqdm(paths, desc='Inferencing'):
            k = os.path.split(i_path)[-1][:-4]

            # Load image
            img_pil = Image.open(i_path)
            if img_pil.size != (1024, 512):
                img_pil = img_pil.resize((1024, 512), Image.BICUBIC)
            img_ori = np.array(img_pil)[..., :3].transpose([2, 0, 1]).copy()
            x = torch.FloatTensor([img_ori / 255])

            # Inferenceing corners
            cor_id, z0, z1, vis_out = inference(net, x, device,
                                                args.flip, args.rotate,
                                                args.visualize,
                                                not args.relax_cuboid,
                                                args.min_v, args.r,
                                                max_x_rotate=args.max_x_rotate,max_y_rotate=args.max_y_rotate,
                                                min_x_rotate=args.min_x_rotate,min_y_rotate=args.min_y_rotate,
                                                x_rotate_prob=args.x_rotate_prob,y_rotate_prob=args.y_rotate_prob)

            # Output result
            with open(os.path.join(args.output_dir, k + '.json'), 'w') as f:
                json.dump({
                    'z0': float(z0),
                    'z1': float(z1),
                    'uv': [[float(u), float(v)] for u, v in cor_id],
                }, f)

            if vis_out is not None:
                vis_path = os.path.join(args.output_dir, k + '.raw.png')
                vh, vw = vis_out.shape[:2]
                Image.fromarray(vis_out)\
                     .resize((vw//2, vh//2), Image.LANCZOS)\
                     .save(vis_path)
