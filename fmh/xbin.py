import functools as ft
import numpy as np
from bcc import BCC
import homog as hg
import operator




class XformBinner:
    def __init__(self,
                 cart_resl=1,
                 ori_resl=15,
                 cart_bound=512.0,
                 ori_nside=None):
        self.cart_resl = cart_resl
        self.ori_resl = ori_resl
        self.cart_bound = cart_bound
        spec = xbin_make_spec(cart_resl, ori_resl, cart_bound, ori_nside)
        self.bcc6 = BCC(**spec)

    def get_bin_index(self, xform, debug=False):
        face, f6 = xform_to_f6(xform)
        # print('get_bin_index face', face)
        idx = self.bcc6.get_bin_index(f6)
        index = np.bitwise_or(np.left_shift(face, 58), idx)
        if debug: index = index, f6
        return index

    def get_bin_center(self, index, debug=False):
        face = np.right_shift(index, 58)
        if not np.all((0 <= face) * (face < 24)):
            print('get_bin_center face', face[(0 > face) + (face > 24)])
            assert np.all((0 <= face) * (face < 24))
        bcc_key = np.right_shift(np.left_shift(index, 6), 6)
        assert np.all(np.right_shift(bcc_key, 58) == 0)
        f6 = self.bcc6.get_bin_center(bcc_key)
        center = f6_to_xform(face, f6)
        if debug: center = center, f6
        return center


def xbin_make_spec(cart_resl=1, ori_resl=15, cart_bound=512.0, ori_nside=None):
    if cart_resl <= 0:
        raise ValueError('cart_resl must be > 0, not ' + str(cart_resl))
    if ori_resl <= 0:
        raise ValueError('ori_resl must be > 0, not ' + str(ori_resl))
    if cart_bound <= 0:
        raise ValueError('cart_bound must be > 0, not ' + str(cart_bound))
    if not ori_nside:
        ori_nside = int(np.sum(_xform_binner_covrad >= ori_resl) + 1)
    # 1.7 is empirically determined hack...
    sizes = ([int(1.5 * cart_bound / cart_resl)] * 3 + [int(ori_nside)] * 3)
    totsize = ft.reduce(operator.mul, [_ + 2 for _ in sizes], 1)
    if totsize > 2**58:
        raise ValueError('too many bins, increase cart_resl and/or ori_resl,'
                         ' or reduce cart_bound')
    return dict(
        sizes=sizes,
        lower=[-cart_bound] * 3 + [0] * 3,
        upper=[cart_bound] * 3 + [1] * 3)


bt24cell_width = 2 * (np.sqrt(2) - 1)
bt24cell_width_diagonal = 0.696923425058676
_R22 = np.sqrt(2) / 2
half_bt24cell_faces = np.array([
    [1.000, 0.000, 0.000, 0.000],  # 0
    [0.000, 1.000, 0.000, 0.000],  # 1
    [0.000, 0.000, 1.000, 0.000],  # 2
    [0.000, 0.000, 0.000, 1.000],  # 3
    [+0.50, +0.50, +0.50, +0.50],  # 4
    [+0.50, +0.50, +0.50, -0.50],  # 5
    [+0.50, +0.50, -0.50, +0.50],  # 6
    [+0.50, +0.50, -0.50, -0.50],  # 7
    [+0.50, -0.50, +0.50, +0.50],  # 8
    [+0.50, -0.50, +0.50, -0.50],  # 9
    [+0.50, -0.50, -0.50, +0.50],  # 10
    [+0.50, -0.50, -0.50, -0.50],  # 11
    [+_R22, +_R22, 0.000, 0.000],  # 12
    [+_R22, -_R22, 0.000, 0.000],  # 13
    [+_R22, 0.000, +_R22, 0.000],  # 14
    [+_R22, 0.000, -_R22, 0.000],  # 15
    [+_R22, 0.000, 0.000, +_R22],  # 16
    [+_R22, 0.000, 0.000, -_R22],  # 17
    [0.000, +_R22, +_R22, 0.000],  # 18
    [0.000, +_R22, -_R22, 0.000],  # 19
    [0.000, +_R22, 0.000, +_R22],  # 20
    [0.000, +_R22, 0.000, -_R22],  # 21
    [0.000, 0.000, +_R22, +_R22],  # 22
    [0.000, 0.000, +_R22, -_R22],  # 23
])
half_bt24cell_faces.flags.writeable = False


def half_bt24cell_face(quat):
    quat = np.asarray(quat)
    fullaxes = (slice(None), ) + (np.newaxis, ) * (quat.ndim - 1)
    hf = half_bt24cell_faces[fullaxes]
    dots = abs(np.sum(quat[np.newaxis] * hf, axis=-1))
    return np.argmax(dots, axis=0)


def numba_half_bt24cell_face(quat):
    dots = np.abs(half_bt24cell_faces @ quat)
    return np.argmax(dots)


def to_face_0(quat):
    face = half_bt24cell_face(quat)
    to_0 = half_bt24cell_faces[face].copy()
    to_0[..., 0] *= -1.0  # inverse
    q0 = hg.quat.quat_multiply(to_0, quat)
    return face, hg.quat.quat_to_upper_half(q0)


def numba_to_face_0(quat):
    face = numba_half_bt24cell_face(quat)
    to_0 = half_bt24cell_faces[face].copy()
    to_0[0] *= -1.0  # inverse
    q0 = hg.quat.numba_quat_multiply(to_0, quat)
    q0 = hg.quat.numba_quat_to_upper_half(q0)
    return face, q0


_xform_binner_covrad = np.array([
    68.907736, 38.232608, 25.860598, 19.391913, 15.461347, 13.152186,
    11.309651, 9.849257, 8.653823, 7.778357, 7.044655, 6.504872, 5.971447,
    5.631019, 5.240999, 4.830578, 4.609091, 4.373091, 4.113840, 3.855201,
    3.726551, 3.529638, 3.377918, 3.262408, 3.171638, 2.978760, 2.879508,
    2.851197, 2.672736, 2.628641, 2.515808, 2.448288
])
_xform_binner_covrad.flags.writeable = False


def _f6_to_quat(f6):
    quat = np.empty(f6.shape[:-1] + (4, ))
    quat[..., 0] = 1.0
    quat[..., 1:] = bt24cell_width * (np.clip(f6[..., 3:6], 0, 1) - 0.5)
    return quat / np.linalg.norm(quat, axis=-1)[..., np.newaxis]


def is_contained_in_bt24cell_face0(f6):
    return half_bt24cell_face(_f6_to_quat(f6)) == 0


def f6_to_xform(face, f6):
    quat = _f6_to_quat(f6)
    quat = hg.quat.quat_multiply(half_bt24cell_faces[face], quat)
    xform = hg.quat.quat_to_xform(quat)
    xform[..., :3, 3] = f6[..., :3]
    return xform


def xform_to_f6(xform):
    quat = hg.quat.rot_to_quat(xform)
    face, q0 = to_face_0(quat)
    f6 = np.empty(q0.shape[:-1] + (6, ))
    f6[..., :3] = xform[..., :3, 3]
    # TODO (w,x,y,z) why divide by xyz by w? in cell0, w close to 1.0?
    ori_params = 0.5 + q0[..., 1:4] / q0[..., 0, None] / bt24cell_width
    f6[..., 3:6] = np.clip(ori_params, 0, 1)
    return face, f6

def numba_xform_to_f6(xform):
    quat = hg.quat.numba_rot_to_quat(xform)
    face, q0 = numba_to_face_0(quat)
    f6 = np.empty((6, ))
    f6[:3] = xform[:3, 3]
    # TODO (w,x,y,z) why divide by xyz by w? in cell0, w close to 1.0?
    ori_params = 0.5 + q0[1:4] / q0[0] / bt24cell_width
    f6[3:6] = np.minimum(1, np.maximum(0, ori_params))
    return face, f6
