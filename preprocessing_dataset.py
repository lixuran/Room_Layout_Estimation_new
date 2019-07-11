import os
import numpy as np
from PIL import Image
from shapely.geometry import LineString
from scipy.spatial.distance import cdist

import torch
import torch.utils.data as data

from misc import panostretch
from misc import pano_lsd_align

class PanoCorBonDataset(data.Dataset):
    '''
    See README.md for how to prepare the dataset.
    '''

    def __init__(self, root_dir,
                 flip=False, rotate=False, gamma=False, stretch=False,
                 p_base=0.96, max_stretch=2.0,
                 normcor=False, return_cor=False, return_path=False,
                 max_x_rotate=0,max_y_rotate=0,max_y_translation=0,
                 min_x_rotate=0,min_y_rotate=0,x_rotate_prob=1,y_rotate_prob=1):
        self.img_dir = os.path.join(root_dir, 'img')
        self.cor_dir = os.path.join(root_dir, 'label_cor')
        self.img_fnames = sorted([fname for fname in os.listdir(self.img_dir)])
        self.txt_fnames = ['%s.txt' % fname[:-4] for fname in self.img_fnames]
        self.flip = flip
        self.rotate = rotate
        self.gamma = gamma
        self.stretch = stretch
        self.p_base = p_base
        self.max_stretch = max_stretch
        self.normcor = normcor
        self.return_cor = return_cor
        self.return_path = return_path
        self.max_x_rotate=max_x_rotate
        self.max_y_rotate=max_y_rotate
        self.max_y_translation=max_y_translation
        self.min_x_rotate=min_x_rotate
        self.min_y_rotate=min_y_rotate
        self.x_rotate_prob=x_rotate_prob
        self.y_rotate_prob=y_rotate_prob
        self._check_dataset()

    def _check_dataset(self):
        for fname in self.img_fnames:
            assert fname.endswith('.jpg') or fname.endswith('.png'),\
                'All filenames under img_dir should endswith .jpg or .png'

        for fname in self.txt_fnames:
            assert os.path.isfile(os.path.join(self.cor_dir, fname)),\
                '%s not found' % os.path.join(self.cor_dir, fname)

    def __len__(self):
        return len(self.img_fnames)

    def __getitem__(self, idx):
        # Read image
        img_path = os.path.join(self.img_dir,
                                self.img_fnames[idx])
        img = np.array(Image.open(img_path), np.float32)[..., :3] / 255.
        H, W = img.shape[:2]
        # init random variable
        rnd_y,rnd_x=0,0
        
        # Read ground truth corners
        with open(os.path.join(self.cor_dir,
                               self.txt_fnames[idx])) as f:
            cor = np.array([line.strip().split() for line in f], np.float32)

            # Corner with minimum x should at the beginning
            cor = np.roll(cor[:, :2], -2 * np.argmin(cor[::2, 0]), 0)

            # Detect occlusion
            occlusion = find_occlusion(cor[::2].copy()).repeat(2)
            assert (cor[0::2, 0] != cor[1::2, 0]).sum() == 0
            assert (cor[0::2, 1] > cor[1::2, 1]).sum() == 0

        # Stretch augmentation
        if self.stretch:
            xmin, ymin, xmax, ymax = cor2xybound(cor)
            kx = np.random.uniform(1.0, self.max_stretch)
            ky = np.random.uniform(1.0, self.max_stretch)
            if np.random.randint(2) == 0:
                kx = max(1 / kx, min(0.5 / xmin, 1.0))
            else:
                kx = min(kx, max(10.0 / xmax, 1.0))
            if np.random.randint(2) == 0:
                ky = max(1 / ky, min(0.5 / ymin, 1.0))
            else:
                ky = min(ky, max(10.0 / ymax, 1.0))
            img, cor = panostretch.pano_stretch(img, cor, kx, ky)
        
        # Prepare 1d ceiling-wall/floor-wall boundary
        bon = np.zeros((2, W))
        bon[0, :] = 1e9
        bon[1, :] = -1e9
        n_cor = len(cor)
        for i in range(n_cor // 2):
            xys = panostretch.pano_connect_points(cor[i*2],
                                                  cor[(i*2+2) % n_cor],
                                                  z=-50)
            xys = xys.astype(int)
            #bon[0, xys[:, 0]] = np.minimum(bon[0, xys[:, 0]], xys[:, 1])
        for i in range(n_cor // 2):
            xys = panostretch.pano_connect_points(cor[i*2+1],
                                                  cor[(i*2+3) % n_cor],
                                                  z=50)
            xys = xys.astype(int)
            #bon[1, xys[:, 0]] = np.maximum(bon[1, xys[:, 0]], xys[:, 1])
        bon = ((bon + 0.5) / img.shape[0] - 0.5) * np.pi
       
        #print("boundary before {}".format(bon))
        #print("corner before {}".format(cor))
        # bon to relative height v
        # Random flip
        if self.flip and np.random.randint(2) == 0:
            img = np.flip(img, axis=1)
            bon = np.flip(bon, axis=1)
            cor[:, 0] = img.shape[1] - 1 - cor[:, 0]

        # Random horizontal rotate
        if self.rotate:
            dx = np.random.randint(img.shape[1])
            img = np.roll(img, dx, axis=1)
            bon = np.roll(bon, dx, axis=1)
            cor[:, 0] = (cor[:, 0] + dx) % img.shape[1]

        # Random gamma augmentation
        if self.gamma:
            p = np.random.uniform(1, 2)
            if np.random.randint(2) == 0:
                p = 1 / p
            img = img ** p
        #random x rotate around x axis?
        if (self.max_x_rotate >0 and self.max_x_rotate >self.min_x_rotate) :
            rnd = np.random.uniform()
            if(rnd> self.x_rotate_prob):
                rnd_x = np.random.randint(self.min_x_rotate,self.max_x_rotate)
                img=pano_lsd_align.rotatePanorama_degree(img,rnd_x,axis=1)
                bon,cor = pano_lsd_align.rotateCorners(bon,cor,rnd_x,axis=1) #simple :> its not working :< 
        #random z rotate around y axis?
        if (self.max_y_rotate >0 and self.max_y_rotate >self.min_y_rotate) :
            rnd = np.random.uniform()
            if(rnd> self.y_rotate_prob):
                rnd_y = np.random.randint(self.min_y_rotate,self.max_y_rotate)
                img=pano_lsd_align.rotatePanorama_degree(img,rnd_y,axis=2)
                bon,cor = pano_lsd_align.rotateCorners(bon,cor,rnd_y,axis=2) #simple :>
        #print("boundary after {}".format(bon))
        #print("corner after {}".format(cor))        
        # Prepare 1d wall-wall probability
        corx = cor[~occlusion, 0]
        dist_o = cdist(corx.reshape(-1, 1),
                       np.arange(img.shape[1]).reshape(-1, 1),
                       p=1)
        dist_r = cdist(corx.reshape(-1, 1),
                       np.arange(img.shape[1]).reshape(-1, 1) + img.shape[1],
                       p=1)
        dist_l = cdist(corx.reshape(-1, 1),
                       np.arange(img.shape[1]).reshape(-1, 1) - img.shape[1],
                       p=1)
        dist = np.min([dist_o, dist_r, dist_l], 0)
        nearest_dist = dist.min(0)
        y_cor = (self.p_base ** nearest_dist).reshape(1, -1)
        
        # Convert all data to tensor
        x = torch.FloatTensor(img.transpose([2, 0, 1]).copy())
        print(x.shape)
        bon = torch.FloatTensor(bon.copy())
        y_cor = torch.FloatTensor(y_cor.copy())
        #print("boundary final {}".format(bon))
        #print("corner final {}".format(y_cor))
        
        # Check whether additional output are requested
        #rnd_x rnd_y are in degrees 
        out_lst = [x, bon, y_cor,rnd_x,rnd_y]
        if self.return_cor:
            out_lst.append(cor)
        if self.return_path:
            out_lst.append(img_path)
       
            out_lst
        return out_lst


def find_occlusion(coor):
    u = panostretch.coorx2u(coor[:, 0])
    v = panostretch.coory2v(coor[:, 1])
    x, y = panostretch.uv2xy(u, v, z=-50)
    occlusion = []
    for i in range(len(x)):
        raycast = LineString([(0, 0), (x[i], y[i])])
        other_layout = []
        for j in range(i+1, len(x)):
            other_layout.append((x[j], y[j]))
        for j in range(0, i):
            other_layout.append((x[j], y[j]))
        other_layout = LineString(other_layout)
        occlusion.append(raycast.intersects(other_layout))
    return np.array(occlusion)


def cor2xybound(cor):
    ''' Helper function to clip max/min stretch factor '''
    corU = cor[0::2]
    corB = cor[1::2]
    zU = -50
    u = panostretch.coorx2u(corU[:, 0])
    vU = panostretch.coory2v(corU[:, 1])
    vB = panostretch.coory2v(corB[:, 1])

    x, y = panostretch.uv2xy(u, vU, z=zU)
    c = np.sqrt(x**2 + y**2)
    zB = c * np.tan(vB)
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    S = 3 / abs(zB.mean() - zU)
    dx = [abs(xmin * S), abs(xmax * S)]
    dy = [abs(ymin * S), abs(ymax * S)]

    return min(dx), min(dy), max(dx), max(dy)


def visualize_a_data(x, y_bon, y_cor):
    x = (x.numpy().transpose([1, 2, 0]) * 255).astype(np.uint8)
    y_bon = y_bon.numpy()
    y_bon = ((-y_bon / np.pi - 0.5) * x.shape[0]).round().astype(int)
    y_cor = y_cor.numpy()

    gt_cor = np.zeros((30, 1024, 3), np.uint8)
    gt_cor[:] =  y_cor[0][None, :, None] * 255
    img_pad = np.zeros((3, 1024, 3), np.uint8) + 255

    img_bon = (x.copy() * 0.5).astype(np.uint8)
    #print(img_bon.shape)
    #print(x.shape)
    #print(y_bon.shape)
    #print(y_bon)
    #print(y_cor)
    #print( y_cor[0][None, :, None].shape)
    #print(y_cor[0][None, :, None])
    y1 = np.round(y_bon[0]).astype(int)
    y2 = np.round(y_bon[1]).astype(int)
    #print(y1.shape)
    y1 = np.vstack([np.arange(1024), y1]).T.reshape(-1, 1, 2)
    y2 = np.vstack([np.arange(1024), y2]).T.reshape(-1, 1, 2)
    img_bon[y_bon[0], np.arange(len(y_bon[0])), 1] = 255
    img_bon[y_bon[1], np.arange(len(y_bon[1])), 1] = 255

    return np.concatenate([gt_cor, img_pad, img_bon], 0)



if __name__ == '__main__':

    import argparse
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='data/valid/')
    parser.add_argument('--ith', default=0, type=int,
                        help='Pick a data id to visualize.'
                             '-1 for visualize all data')
    parser.add_argument('--flip', action='store_true',
                        help='whether to random flip')
    parser.add_argument('--rotate', action='store_true',
                        help='whether to random horizon rotation')
    parser.add_argument('--gamma', action='store_true',
                        help='whether to random luminance change')
    parser.add_argument('--stretch', action='store_true',
                        help='whether to random pano stretch')
    parser.add_argument('--max_x_rotate', default=0, type=int,
                        help='maximum degree of x axis rotation')
    parser.add_argument('--min_x_rotate',  default=0, type=int,
                        help='minimum degree of x axis rotation')
    parser.add_argument('--max_y_rotate',  default=0, type=int,
                        help='maximum degree of z axis rotation')
    parser.add_argument('--min_y_rotate',  default=0, type=int,
                        help='minimum degree of z axis rotation')
    parser.add_argument('--y_rotate_prob',  default=1, type=float,
                        help='1 - probability of y axis rotation')
    parser.add_argument('--x_rotate_prob',  default=1, type=float,
                        help='1 - probability of x axis rotation')
    parser.add_argument('--max_y_translation', default=0, type=int,
                        help='maximum degree of y(z) axis translation')
    parser.add_argument('--out_dir', default='distorted')
    args = parser.parse_args()
    if (args.max_y_rotate !=0 or args.max_x_rotate!=0):
        out_dir_version= "{}_{}_{}".format(args.out_dir,args.max_y_rotate,args.max_x_rotate)
    else:
        out_dir_version=args.out_dir
    os.makedirs(out_dir_version, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(out_dir_version+"/images", exist_ok=True)
    os.makedirs(out_dir_version+"/labels", exist_ok=True)
    print('args:')
    for key, val in vars(args).items():
        print('    {:16} {}'.format(key, val))

    dataset = PanoCorBonDataset(
        root_dir=args.root_dir,
        flip=args.flip, rotate=args.rotate, gamma=args.gamma, stretch=args.stretch,
        max_x_rotate=args.max_x_rotate,max_y_rotate=args.max_y_rotate,min_y_rotate=args.min_y_rotate,min_x_rotate=args.min_x_rotate,
        x_rotate_prob=args.x_rotate_prob,y_rotate_prob=args.y_rotate_prob,max_y_translation=args.max_y_translation,
        return_path=True)

    # Showing some information about dataset
    print('len(dataset): {}'.format(len(dataset)))
    x, y_bon, y_cor, path,y_rotate,z_rotate = dataset[0]
    print('x', x.size())
    print('y_bon', y_bon.size())
    print('y_cor', y_cor.size())

    if args.ith >= 0:
        to_visualize = [dataset[args.ith]]
    else:
        to_visualize = dataset
    
    for x, y_bon, y_cor,rnd_x,rnd_y, path in tqdm(to_visualize):
        
        fname = os.path.split(path)[-1]
        out = visualize_a_data(x, y_bon, y_cor)
        
        Image.fromarray(out).save(os.path.join(args.out_dir, fname))
      
   # for x, y_bon, y_cor,rnd_x,rnd_y, path in tqdm(dataset):
   #     fname = os.path.split(path)[-1]
        
   #     x = (x.numpy().transpose([1, 2, 0]) * 255).astype(np.uint8)
   #     Image.fromarray(x).save(os.path.join(out_dir_version,"images", fname))
   #     with open(os.path.join(out_dir_version,"labels" ,"{}.txt".format(fname)), "w+") as f: 
   #         f.write("{},{}".format(rnd_x,rnd_y)) 
