import torch


def sinkhorn_gpu(a, b, M, reg, numItermax=1000, stopThr=1e-9,
                 verbose=False, log=False, device='cuda:0'):
    """Computes the optimal transport plan between empirical measures given by a = [a1, ..., an] and
    b = [b1, ..., bm], given the cost Mij = C(xi, yj).

    Parameters
    ----------
    a : tensor of shape (ns,)
        sample weights in the source domain
    b : tensor of shape (nt,)
        sample weights in the target domain
    M : tensor of shape (ns, nt)
        ground-cost for OT
    reg : float
        regularization penalty
    numItermax : int
        max number of iterations
    stopThr : float
        if error is below stopThr, stop iterations.
    verbose : bool
        prints information during execution
    log : bool
        record error and returns dual solution.

    Returns
    -------
    G : tensor of shape (ns, nt)
        optimal transport plan.
    log : dict
        dictionary containing error info and dual solution. Only returned if log=True.
    """
    if (not torch.cuda.is_available()) and 'cuda' in device:
        raise ValueError("Tried to use cuda device, but cuda is not available.")
    if log: logs = {'err': []}
    if verbose: print("|{:^25}|{:^25}|".format('It', 'Err'))
    if verbose: print("-" * (26 * 2 + 1))
    a = torch.Tensor(a).double() if type(a) != torch.Tensor else a
    b = torch.Tensor(b).double() if type(b) != torch.Tensor else b
    M = torch.Tensor(M).double() if type(M) != torch.Tensor else M
    with torch.no_grad():
        a = a.to(device)
        b = b.to(device)

        # init data
        ns = len(a)
        nt = len(b)

        u = (torch.ones(ns) / ns).to(device).double()
        v = (torch.ones(nt) / nt).to(device).double()

        K = torch.exp(- M / reg).to(device)
        Kp = (1 / a).view(-1, 1) * K
        cpt = 0
        err = 1
        while (err > stopThr and cpt < numItermax):
            uprev = u
            vprev = v
            KtransposeU = K.t().matmul(u)
            v = torch.div(b.view(-1, 1), KtransposeU.view(-1 ,1))
            u = 1. / Kp.matmul(v)

            if (torch.any(KtransposeU == 0) or
                torch.any(torch.isnan(u)) or
                torch.any(torch.isnan(v))):
                u = uprev
                v = vprev
                if verbose: print('Warning: numerical errors at iteration: {}'.format(cpt))
                break

            if cpt % 100 == 0:
                # print(u.shape, K.shape, v.shape)
                # transp = u.view(-1, 1) * (K * v)
                transp = torch.einsum('ik,ij,jk->jk', u, K, v)
                err = (torch.sum(transp) - b).norm(1).pow(2).cpu().numpy()
                if log: logs['err'].append(err)
                if verbose: print("|{:^25}|{:^25}|".format(cpt, err))
            cpt += 1
        # print(u.shape, v.shape, torch.diag(u).shape, K.shape, torch.diag(v).shape)
        # print(u, '\n', K, '\n', v)
        G = torch.matmul(torch.diag(u.view(-1)), torch.matmul(K, torch.diag(v.view(-1))))

        if log:
            logs['u'] = u.detach().cpu().numpy()
            logs['v'] = v.detach().cpu().numpy()
            return G.detach().cpu().numpy(), logs
        a = a.detach().cpu().numpy()
        b = b.detach().cpu().numpy()
        M = M.detach().cpu().numpy()
        return G.detach().cpu().numpy()
