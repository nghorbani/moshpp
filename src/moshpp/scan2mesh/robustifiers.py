#!/usr/bin/env python

__all__ = ['GMOf']

import numpy as np
import scipy
import scipy.sparse as sp
from chumpy import Ch


class Sigmoid(Ch):
    dterms = 'x'
    terms = 'scale'
    term_order = 'x', 'scale'

    def on_changed(self, which):
        if 'x' in which:
            self.x_squared = self.x.r ** 2.
        self.y = self.x_squared + self.scale ** 2.
        self.z = np.sqrt(self.y)

    def compute_r(self):
        return self.x.r * self.scale / self.z

    def compute_dr_wrt(self, wrt):
        if wrt is self.x:
            result = self.z - self.x_squared / self.z
            result *= self.scale
            result /= self.y
            return sp.spdiags(result.ravel(), [0], self.x.r.size, self.x.r.size)


def GMOf(x, sigma):
    """Given x and sigma in some unit (say mm), returns robustified values (in same unit),
    by making use of the Geman-McClure robustifier."""

    result = SignedSqrt(x=GMOfInternal(x=x, sigma=sigma))
    return result
    # result = Ch(lambda xx, xsigma : SignedSqrt(x=GMOfInternal(x=xx, sigma=xsigma)))
    # result.xx = x
    # result.xsigma = sigma
    # return result


class SignedSqrt(Ch):
    dterms = ('x',)
    terms = ()

    def compute_r(self):
        return np.sqrt(np.abs(self.x.r)) * np.sign(self.x.r)

    def compute_dr_wrt(self, wrt):
        if wrt is self.x:
            result = (.5 / np.sqrt(np.abs(self.x.r)))
            result = np.nan_to_num(result)
            result *= (self.x.r != 0).astype(np.uint32)
            return sp.spdiags(result.ravel(), [0], self.x.r.size, self.x.r.size)


class GMOfInternal(Ch):
    dterms = 'x', 'sigma'

    def on_changed(self, which):
        if 'sigma' in which:
            assert (self.sigma.r > 0)

        if 'x' in which:
            self.squared_input = self.x.r ** 2.

    def compute_r(self):
        return (self.sigma.r ** 2 * (self.squared_input / (self.sigma.r ** 2 + self.squared_input))) * np.sign(self.x.r)

    def compute_dr_wrt(self, wrt):
        if wrt is not self.x and wrt is not self.sigma:
            return None

        squared_input = self.squared_input
        result = []
        if wrt is self.x:
            dx = self.sigma.r ** 2 / (self.sigma.r ** 2 + squared_input) - self.sigma.r ** 2 * (
                        squared_input / (self.sigma.r ** 2 + squared_input) ** 2)
            dx = 2 * self.x.r * dx
            result.append(
                scipy.sparse.spdiags((dx * np.sign(self.x.r)).ravel(), [0], self.x.r.size, self.x.r.size, format='csc'))
        if wrt is self.sigma:
            ds = 2 * self.sigma.r * (squared_input / (self.sigma.r ** 2 + squared_input)) - 2 * self.sigma.r ** 3 * (
                        squared_input / (self.sigma.r ** 2 + squared_input) ** 2)
            result.append(
                scipy.sparse.spdiags((ds * np.sign(self.x.r)).ravel(), [0], self.x.r.size, self.x.r.size, format='csc'))

        if len(result) == 1:
            return result[0]
        else:
            return np.sum(result).tocsc()


def GMOf_normalized(x, sigma):
    """Given x and sigma in some unit (say mm), returns robustified values between [0 and 1],
    by making use of the Geman-McClure robustifier.
    The sigma value defines the size of the bassin of attraction
    GMO_normalized(x, sigma) = x**2 / (sigma**2 + x**2) * sign(x) """

    result = SignedSqrt(x=GMOfInternal_normalized(x=x, sigma=sigma))
    return result


class GMOfInternal_normalized(Ch):
    dterms = 'x', 'sigma'

    def on_changed(self, which):
        if 'sigma' in which:
            assert (self.sigma.r > 0)

        if 'x' in which:
            self.squared_input = self.x.r ** 2.

    def compute_r(self):
        return ((self.squared_input / (self.sigma.r ** 2 + self.squared_input))) * np.sign(self.x.r)

    def compute_dr_wrt(self, wrt):
        if wrt is not self.x and wrt is not self.sigma:
            return None

        squared_input = self.squared_input
        result = []
        if wrt is self.x:
            dx = 2 * self.x.r * ((self.sigma.r ** 2 + squared_input) - squared_input)
            dx = dx / ((self.sigma.r ** 2 + squared_input) ** 2)
            result.append(
                scipy.sparse.spdiags((dx * np.sign(self.x.r)).ravel(), [0], self.x.r.size, self.x.r.size, format='csc'))
        if wrt is self.sigma:
            ds = -2 * self.sigma.r * squared_input / ((self.sigma.r ** 2 + squared_input) ** 2)
            result.append(
                scipy.sparse.spdiags((ds * np.sign(self.x.r)).ravel(), [0], self.x.r.size, self.x.r.size, format='csc'))

        if len(result) == 1:
            return result[0]
        else:
            return np.sum(result).tocsc()
