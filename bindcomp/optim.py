"""Adam with decoupled weight decay over nested-list parameters."""

from __future__ import annotations

from math import sqrt


class Adam:
    def __init__(self, params, lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.params = params
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, betas[0], betas[1], eps, weight_decay
        self.t = 0
        self.m = {k: self._zeros_like(v) for k, v in params.items()}
        self.v = {k: self._zeros_like(v) for k, v in params.items()}

    @staticmethod
    def _zeros_like(v):
        return [[0.0] * len(r) for r in v] if isinstance(v[0], list) else [0.0] * len(v)

    def _update_row(self, p, g, m, v, c1, c2):
        for j in range(len(p)):
            m[j] = self.b1 * m[j] + (1 - self.b1) * g[j]
            v[j] = self.b2 * v[j] + (1 - self.b2) * g[j] * g[j]
            p[j] -= self.lr * ((m[j] / c1) / (sqrt(v[j] / c2) + self.eps) + self.wd * p[j])

    def step(self, grads):
        self.t += 1
        c1 = 1 - self.b1 ** self.t
        c2 = 1 - self.b2 ** self.t
        for k, p in self.params.items():
            g, m, v = grads[k], self.m[k], self.v[k]
            if isinstance(p[0], list):
                for i in range(len(p)):
                    self._update_row(p[i], g[i], m[i], v[i], c1, c2)
            else:
                self._update_row(p, g, m, v, c1, c2)


def grad_norm(grads):
    s = 0.0
    for g in grads.values():
        if isinstance(g[0], list):
            s += sum(x * x for row in g for x in row)
        else:
            s += sum(x * x for x in g)
    return sqrt(s)
