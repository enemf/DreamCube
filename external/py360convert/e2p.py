import numpy as np
from typing import Iterable
from . import utils


def e2p(e_img, fov_deg, u_deg, v_deg, out_hw, in_rot_deg=0, mode='bilinear'):
    '''
    e_img:   ndarray in shape of [H, W, *]
    fov_deg: scalar or (scalar, scalar) field of view in degree
    u_deg:   horizon viewing angle in range [-180, 180]
    v_deg:   vertical viewing angle in range [-90, 90]
    '''
    assert len(e_img.shape) in (2, 3)
    h, w = e_img.shape[:2]

    if isinstance(fov_deg, Iterable):
        h_fov, v_fov = fov_deg[0] * np.pi / 180, fov_deg[1] * np.pi / 180
    else:
        h_fov, v_fov = fov_deg * np.pi / 180, fov_deg * np.pi / 180
    
    in_rot = in_rot_deg * np.pi / 180

    if mode == 'bilinear':
        order = 1
    elif mode == 'nearest':
        order = 0
    else:
        raise NotImplementedError('unknown mode')

    u = -u_deg * np.pi / 180
    v = v_deg * np.pi / 180
    xyz = utils.xyzpers(h_fov, v_fov, u, v, out_hw, in_rot)
    uv = utils.xyz2uv(xyz)
    coor_xy = utils.uv2coor(uv, h, w)

    if len(e_img.shape) == 2:
        pers_img = utils.sample_equirec(e_img, coor_xy, order=order)
    else:
        pers_img = np.stack([
            utils.sample_equirec(e_img[..., i], coor_xy, order=order)
            for i in range(e_img.shape[2])
        ], axis=-1)

    return pers_img
