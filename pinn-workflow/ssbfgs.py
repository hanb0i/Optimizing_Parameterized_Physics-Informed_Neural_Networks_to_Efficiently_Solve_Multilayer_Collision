import torch
from torch.optim.optimizer import Optimizer
from functools import reduce


class SSBFGS(Optimizer):
    """
    Implements Self-Scaled BFGS (SSBFGS) optimizer.
    This version includes the Oren-Luenberger (OL) scaling to improve 
    convergence on stiff problems like PINNs.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate (default: 1)
        max_iter (int): maximal number of iterations per optimization step
            (default: 20)
        max_eval (int): maximal number of function evaluations per optimization
            step (default: max_iter * 1.25)
        tolerance_grad (float): termination tolerance on first order optimality
            (default: 1e-8)
        tolerance_change (float): termination tolerance on function value/parameter
            changes (default: 1e-9)
        history_size (int): update history size (default: 20)
        line_search_fn (str): 'strong_wolfe' (backtracking) or None (default: None)
        scaling_mode (str): 'ol' (Oren-Luenberger) or 'none' (default: 'ol')
    """

    def __init__(self, params,
                 lr=1,
                 max_iter=20,
                 max_eval=None,
                 tolerance_grad=1e-8,
                 tolerance_change=1e-9,
                 history_size=20,
                 line_search_fn=None,
                 scaling_mode='ol'):
        if max_eval is None:
            max_eval = max_iter * 5 // 4
        defaults = dict(
            lr=lr,
            max_iter=max_iter,
            max_eval=max_eval,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            history_size=history_size,
            line_search_fn=line_search_fn,
            scaling_mode=scaling_mode
        )
        super(SSBFGS, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("SSBFGS doesn't support per-parameter options "
                             "(parameter groups)")

        self._params = self.param_groups[0]['params']
        self._numel_cache = None

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.new_zeros(p.numel())
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            numel = p.numel()
            p.add_(update[offset:offset + numel].view_as(p), alpha=step_size)
            offset += numel

    def _clone_param(self):
        return [p.clone(memory_format=torch.contiguous_format) for p in self._params]

    def _set_param(self, params_data):
        for p, pdata in zip(self._params, params_data):
            p.copy_(pdata)

    def step(self, closure):
        """Performs a single optimization step.

        Arguments:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
        """
        assert len(self.param_groups) == 1

        group = self.param_groups[0]
        lr = group['lr']
        max_iter = group['max_iter']
        max_eval = group['max_eval']
        tolerance_grad = group['tolerance_grad']
        tolerance_change = group['tolerance_change']
        line_search_fn = group['line_search_fn']
        history_size = group['history_size']
        scaling_mode = group['scaling_mode']

        state = self.state[self._params[0]]
        state.setdefault('func_evals', 0)
        state.setdefault('n_iter', 0)

        # evaluate initial f(x) and f'(x)
        with torch.enable_grad():
            orig_loss = closure()
        loss = orig_loss.item()
        current_evals = 1
        state['func_evals'] += 1

        with torch.no_grad():
            flat_grad = self._gather_flat_grad()
            opt_cond = flat_grad.abs().max() <= tolerance_grad

            if opt_cond:
                return orig_loss

            d = state.get('d')
            t = state.get('t')
            old_dirs = state.get('old_dirs')
            old_stps = state.get('old_stps')
            ro = state.get('ro')
            H_diag = state.get('H_diag')
            prev_flat_grad = state.get('prev_flat_grad')
            prev_loss = state.get('prev_loss')

            n_iter = 0
            while n_iter < max_iter:
                n_iter += 1
                state['n_iter'] += 1

                if state['n_iter'] == 1:
                    d = flat_grad.neg()
                    old_dirs = []
                    old_stps = []
                    ro = []
                    H_diag = 1.0
                else:
                    y = flat_grad.sub(prev_flat_grad)
                    s = d.mul(t)
                    ys = y.dot(s)
                    if ys > 1e-10:
                        if len(old_dirs) == history_size:
                            old_dirs.pop(0)
                            old_stps.pop(0)
                            ro.pop(0)

                        old_dirs.append(y)
                        old_stps.append(s)
                        ro.append(1. / ys)

                        if scaling_mode == 'ol':
                            H_diag = ys / y.dot(y)
                        else:
                            H_diag = ys / y.dot(y)

                    num_old = len(old_dirs)
                    if 'al' not in state:
                        state['al'] = [None] * history_size
                    al = state['al']

                    q = flat_grad.neg()
                    for i in range(num_old - 1, -1, -1):
                        al[i] = old_stps[i].dot(q) * ro[i]
                        q.add_(old_dirs[i], alpha=-al[i])

                    r = q.mul(H_diag)
                    for i in range(num_old):
                        be_i = old_dirs[i].dot(r) * ro[i]
                        r.add_(old_stps[i], alpha=al[i] - be_i)
                    d = r

                if prev_loss is not None:
                    if abs(loss - prev_loss) < tolerance_change:
                        break

                if d.abs().max() <= tolerance_change:
                    break

                prev_flat_grad = flat_grad.clone(memory_format=torch.contiguous_format)
                prev_loss = loss

                if line_search_fn == 'strong_wolfe':
                    x_init = self._clone_param()
                    gtd = flat_grad.dot(d)
                    
                    max_ls = 20
                    rho = 0.5
                    c = 1e-4
                    alpha = lr
                    
                    ls_evals = 0
                    for _ in range(max_ls):
                        self._set_param(x_init)
                        self._add_grad(alpha, d)
                        with torch.enable_grad():
                            loss_t = closure()
                            f_new = loss_t.item()
                        ls_evals += 1
                        if f_new <= loss + c * alpha * gtd:
                            loss = f_new
                            flat_grad = self._gather_flat_grad()
                            t = alpha
                            break
                        alpha *= rho
                    else:
                        t = alpha
                        with torch.enable_grad():
                            loss_t = closure()
                            loss = loss_t.item()
                        flat_grad = self._gather_flat_grad()
                        ls_evals += 1
                    
                    current_evals += ls_evals
                    state['func_evals'] += ls_evals
                else:
                    t = lr
                    self._add_grad(t, d)
                    if n_iter != max_iter:
                        with torch.enable_grad():
                            loss_t = closure()
                            loss = loss_t.item()
                        flat_grad = self._gather_flat_grad()
                        current_evals += 1
                        state['func_evals'] += 1

                if current_evals >= max_eval:
                    break
                if flat_grad.abs().max() <= tolerance_grad:
                    break

            state['d'] = d
            state['t'] = t
            state['old_dirs'] = old_dirs
            state['old_stps'] = old_stps
            state['ro'] = ro
            state['H_diag'] = H_diag
            state['prev_flat_grad'] = prev_flat_grad
            state['prev_loss'] = prev_loss

        return orig_loss
