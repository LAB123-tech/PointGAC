# -*- coding: utf-8 -*-
# @Time    : 2024-10-29
# @Author  : lab
# @desc    :
import torch


def sinkhorn_knopp(r, c, M, reg=1e-2, error_min=1e-5, num_iters=100, mask=None):
    """
    Batch sinkhorn iteration.
    @param r: tensor with shape (n, d1), the first distribution .
    @param c: tensor with shape (n, d2), the second distribution.
    @param M: tensor with shape (n, d1, d2) the cost metric.
    @param reg: factor for entropy regularization.
    @param error_min: the error threshold to stop the iteration.
    @param num_iters: number of total iterations.
    @param mask: mask region
    @return:
    """
    n, d1, d2 = M.shape
    assert r.shape[0] == c.shape[0] == n and r.shape[1] == d1 and c.shape[1] == d2, \
        'r.shape=%s, v.shape=%s, M.shape=%s' % (r.shape, c.shape, M.shape)
    if mask is None:
        mask = torch.ones_like(M)

    K = (-M / reg).exp()  # (n, d1, d2)
    u = torch.ones_like(r) / d1  # (n, d1)
    v = torch.ones_like(c) / d2  # (n, d2)

    for _ in range(num_iters):
        r0 = u
        v_full = v[:, None, ].repeat(1, d1, 1)
        u = r / (torch.sum(K * v_full * mask, dim=-1) + 1e-5)
        u_full = u[:, :, None].repeat(1, 1, d2)
        v = c / (torch.sum(K * u_full * mask, dim=1) + 1e-5)
        err = (u - r0).abs().mean()
        if err.item() < error_min:
            break

    T = torch.einsum('ij,ik->ijk', [u, v]) * K * mask
    return T, u, v
