import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jit, vmap

from stopro.GP.gp_stokes_3D import GPStokes3D


class GPStokes2D2C(GPStokes3D):
    """
    Class for the inference using difference of pressure in 2D Stokes system.
    """

    def setup_trainingKs(self):
        self.trainingKs = [
            [
                self.Kuxux,
                self.Kuxuy,
                self.Kuxfx,
                self.Kuxfy,
                self.Kuxfz,
                self.Kuxdiv,
            ],
            [
                self.Kuyuy,
                self.Kuyfx,
                self.Kuyfy,
                self.Kuyfz,
                self.Kuydiv,
            ],
            [self.Kfxfx, self.Kfxfy, self.Kfxfz, self.Kfxdiv],
            [self.Kfyfy, self.Kfyfz, self.Kfydiv],
            [self.Kfzfz, self.Kfzdiv],
            [self.Kdivdiv],
        ]

    def setup_mixedKs(self):
        if self.infer_governing_eqs:
            pass
        elif self.use_difp:
            pass
        else:
            self.mixedKs = [
                [
                    self.Kuxux,
                    self.Kuxuy,
                    self.Kuxfx,
                    self.Kuxfy,
                    self.Kuxfz,
                    self.Kuxdiv,
                ],
                [
                    self.Kuxuy,
                    self.Kuyuy,
                    self.Kuyfx,
                    self.Kuyfy,
                    self.Kuyfz,
                    self.Kuydiv,
                ],
                [
                    self.Kuxuz,
                    self.Kuyuz,
                    self.Kuzfx,
                    self.Kuzfy,
                    self.Kuzfz,
                    self.Kuzdiv,
                ],
            ]

    def setup_testKs(self):
        if self.infer_governing_eqs:
            pass
        else:
            self.testKs = [
                [self.Kuxux, self.Kuxuy, self.Kuxuz],
                [self.Kuyuy, self.Kuyuz],
                [self.Kuzuz],
            ]

class GPStokes2D2CSurface(GPStokes3D):
    """
    Class for the inference using difference of pressure in 2D Stokes system.
    """

    def setup_trainingKs(self):
        self.trainingKs = [
            [
                self.Kuxux,
                self.Kuxuy,
                self.Kuxux,
                self.Kuxuy,
                self.Kuxuz,
                self.Kuxfx,
                self.Kuxfy,
                self.Kuxfz,
                self.Kuxdiv,
            ],
            [
                self.Kuyuy,
                self.Kuyux,
                self.Kuyuy,
                self.Kuyuz,
                self.Kuyfx,
                self.Kuyfy,
                self.Kuyfz,
                self.Kuydiv,
            ],
            [
                self.Kuxux,
                self.Kuxuy,
                self.Kuxuz,
                self.Kuxfx,
                self.Kuxfy,
                self.Kuxfz,
                self.Kuxdiv,
            ],
            [
                self.Kuyuy,
                self.Kuyuz,
                self.Kuyfx,
                self.Kuyfy,
                self.Kuyfz,
                self.Kuydiv,
            ],
            [
                self.Kuzuz,
                self.Kuzfx,
                self.Kuzfy,
                self.Kuzfz,
                self.Kuzdiv,
            ],
            [self.Kfxfx, self.Kfxfy, self.Kfxfz, self.Kfxdiv],
            [self.Kfyfy, self.Kfyfz, self.Kfydiv],
            [self.Kfzfz, self.Kfzdiv],
            [self.Kdivdiv],
        ]

    def setup_mixedKs(self):
        if self.infer_governing_eqs:
            pass
        elif self.use_difp:
            pass
        else:
            self.mixedKs = [
                [
                    self.Kuxux,
                    self.Kuxuy,
                    self.Kuxux,
                    self.Kuxuy,
                    self.Kuxuz,
                    self.Kuxfx,
                    self.Kuxfy,
                    self.Kuxfz,
                    self.Kuxdiv,
                ],
                [
                    self.Kuyux,
                    self.Kuyuy,
                    self.Kuyux,
                    self.Kuyuy,
                    self.Kuyuz,
                    self.Kuyfx,
                    self.Kuyfy,
                    self.Kuyfz,
                    self.Kuydiv,
                ],
                [
                    self.Kuxuz,
                    self.Kuyuz,
                    self.Kuxuz,
                    self.Kuyuz,
                    self.Kuzuz,
                    self.Kuzfx,
                    self.Kuzfy,
                    self.Kuzfz,
                    self.Kuzdiv,
                ],
            ]

    def setup_testKs(self):
        if self.infer_governing_eqs:
            pass
        else:
            self.testKs = [
                [self.Kuxux, self.Kuxuy, self.Kuxuz],
                [self.Kuyuy, self.Kuyuz],
                [self.Kuzuz],
            ]
