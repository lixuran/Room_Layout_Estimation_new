import functools
import numpy as np
from scipy.ndimage import map_coordinates


def uv_meshgrid(w, h):
    uv = np.stack(np.meshgrid(range(w), range(h)), axis=-1)
    uv = uv.astype(np.float64)
    uv[..., 0] = ((uv[..., 0] + 0.5) / w - 0.5) * 2 * np.pi
    uv[..., 1] = ((uv[..., 1] + 0.5) / h - 0.5) * np.pi
    return uv


@functools.lru_cache()
def _uv_tri(w, h):
    uv = uv_meshgrid(w, h)
    sin_u = np.sin(uv[..., 0])
    cos_u = np.cos(uv[..., 0])
    tan_v = np.tan(uv[..., 1])
    return sin_u, cos_u, tan_v


def uv_tri(w, h):
    sin_u, cos_u, tan_v = _uv_tri(w, h)
    return sin_u.copy(), cos_u.copy(), tan_v.copy()


def coorx2u(x, w=1024):
    return ((x + 0.5) / w - 0.5) * 2 * np.pi


def coory2v(y, h=512):
    return ((y + 0.5) / h - 0.5) * np.pi


def u2coorx(u, w=1024):
    return (u / (2 * np.pi) + 0.5) * w - 0.5


def v2coory(v, h=512):
    return (v / np.pi + 0.5) * h - 0.5


def uv2xy(u, v, z=-50):
    c = z / np.tan(v)
    x = c * np.cos(u)
    y = c * np.sin(u)
    return x, y


def pano_connect_points(p1, p2, z=-50, w=1024, h=512):
    if p1[0] == p2[0]:
        return np.array([p1, p2], np.float32)

    u1 = coorx2u(p1[0], w)
    v1 = coory2v(p1[1], h)
    u2 = coorx2u(p2[0], w)
    v2 = coory2v(p2[1], h)

    x1, y1 = uv2xy(u1, v1, z)
    x2, y2 = uv2xy(u2, v2, z)

    if abs(p1[0] - p2[0]) < w / 2:
        pstart = np.ceil(min(p1[0], p2[0]))
        pend = np.floor(max(p1[0], p2[0]))
    else:
        pstart = np.ceil(max(p1[0], p2[0]))
        pend = np.floor(min(p1[0], p2[0]) + w)
    coorxs = (np.arange(pstart, pend + 1) % w).astype(np.float64)
    vx = x2 - x1
    vy = y2 - y1
    us = coorx2u(coorxs, w)
    ps = (np.tan(us) * x1 - y1) / (vy - np.tan(us) * vx)
    cs = np.sqrt((x1 + ps * vx) ** 2 + (y1 + ps * vy) ** 2)
    vs = np.arctan2(z, cs)
    coorys = v2coory(vs)

    return np.stack([coorxs, coorys], axis=-1)


def pano_stretch(img, corners, kx, ky, order=1):
    '''
    img:     [H, W, C]
    corners: [N, 2] in image coordinate (x, y) format
    kx:      Stretching along front-back direction
    ky:      Stretching along left-right direction
    order:   Interpolation order. 0 for nearest-neighbor. 1 for bilinear.
    '''

    # Process image
    sin_u, cos_u, tan_v = uv_tri(img.shape[1], img.shape[0])
    u0 = np.arctan2(sin_u * kx / ky, cos_u)
    v0 = np.arctan(tan_v * np.sin(u0) / sin_u * ky)

    refx = (u0 / (2 * np.pi) + 0.5) * img.shape[1] - 0.5
    refy = (v0 / np.pi + 0.5) * img.shape[0] - 0.5

    # [TODO]: using opencv remap could probably speedup the process a little
    stretched_img = np.stack([
        map_coordinates(img[..., i], [refy, refx], order=order, mode='wrap')
        for i in range(img.shape[-1])
    ], axis=-1)

    # Process corners
    corners_u0 = coorx2u(corners[:, 0], img.shape[1])
    corners_v0 = coory2v(corners[:, 1], img.shape[0])
    corners_u = np.arctan2(np.sin(corners_u0) * ky / kx, np.cos(corners_u0))
    corners_v = np.arctan(np.tan(corners_v0) * np.sin(corners_u) / np.sin(corners_u0) / ky)
    cornersX = u2coorx(corners_u, img.shape[1])
    cornersY = v2coory(corners_v, img.shape[0])
    stretched_corners = np.stack([cornersX, cornersY], axis=-1)

    return stretched_img, stretched_corners


def pano_stretch_no_cor(img, kx, ky, order=1):
    '''
    img:     [H, W, C]
    corners: [N, 2] in image coordinate (x, y) format
    kx:      Stretching along front-back direction
    ky:      Stretching along left-right direction
    order:   Interpolation order. 0 for nearest-neighbor. 1 for bilinear.
    '''

    # Process image
    sin_u, cos_u, tan_v = uv_tri(img.shape[1], img.shape[0])
    u0 = np.arctan2(sin_u * kx / ky, cos_u)
    v0 = np.arctan(tan_v * np.sin(u0) / sin_u * ky)

    refx = (u0 / (2 * np.pi) + 0.5) * img.shape[1] - 0.5
    refy = (v0 / np.pi + 0.5) * img.shape[0] - 0.5

    # [TODO]: using opencv remap could probably speedup the process a little
    stretched_img = np.stack([
        map_coordinates(img[..., i], [refy, refx], order=order, mode='wrap')
        for i in range(img.shape[-1])
    ], axis=-1)

    return stretched_img
    
def visualize_pano_stretch(stretched_img, stretched_cor, title):
    '''
    Helper function for visualizing the effect of pano_stretch
    '''
    thikness = 2
    color = (0, 255, 0)
    for i in range(4):
        xys = pano_connect_points(stretched_cor[i*2], stretched_cor[(i*2+2) % 8], z=-50)
        xys = xys.astype(int)
        blue_split = np.where((xys[1:, 0] - xys[:-1, 0]) < 0)[0]
        if len(blue_split) == 0:
            cv2.polylines(stretched_img, [xys], False, color, 2)
        else:
            t = blue_split[0] + 1
            cv2.polylines(stretched_img, [xys[:t]], False, color, thikness)
            cv2.polylines(stretched_img, [xys[t:]], False, color, thikness)

    for i in range(4):
        xys = pano_connect_points(stretched_cor[i*2+1], stretched_cor[(i*2+3) % 8], z=50)
        xys = xys.astype(int)
        blue_split = np.where((xys[1:, 0] - xys[:-1, 0]) < 0)[0]
        if len(blue_split) == 0:
            cv2.polylines(stretched_img, [xys], False, color, 2)
        else:
            t = blue_split[0] + 1
            cv2.polylines(stretched_img, [xys[:t]], False, color, thikness)
            cv2.polylines(stretched_img, [xys[t:]], False, color, thikness)

    cv2.putText(stretched_img, title, (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 0), 2, cv2.LINE_AA)

    return stretched_img.astype(np.uint8)


if __name__ == '__main__':

    import argparse
    import time
    from PIL import Image
    import cv2

    parser = argparse.ArgumentParser()
    parser.add_argument('--i', default='data/valid/img/pano_abpohapclcyuuz.png')
    parser.add_argument('--i_gt', default='data/valid/label_cor/pano_abpohapclcyuuz.txt')
    parser.add_argument('--o', default='sample_stretched_pano.png')
    parser.add_argument('--kx', default=2, type=float,
                        help='Stretching along front-back direction')
    parser.add_argument('--ky', default=1, type=float,
                        help='Stretching along left-right direction')
    args = parser.parse_args()

    img = np.array(Image.open(args.i), np.float64)
    with open(args.i_gt) as f:
        cor = np.array([line.strip().split() for line in f], np.int32)
    stretched_img, stretched_cor = pano_stretch(img, cor, args.kx, args.ky)

    title = 'kx=%3.2f, ky=%3.2f' % (args.kx, args.ky)
    visual_stretched_img = visualize_pano_stretch(stretched_img, stretched_cor, title)
    Image.fromarray(visual_stretched_img).save(args.o)
