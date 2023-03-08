import os
import pickle

import cmocean as cmo
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from stopro.data_generator.stokes_2D_generator import StokesDataGenerator

# from stopro.data_handler.data_handle_module import HdfOperator

class Drag(StokesDataGenerator):
    """data generator for 2D drag flow
    """
    def __init__(self, L=1., H=1., particle_radius=0.0390625, particle_center=0.5, particle_y_velocity = -0.01, slide=0.03, random_arrange=False):
        super().__init__(random_arrange)
        self.L = L
        self.H = H
        self.particle_radius = particle_radius
        self.particle_center = particle_center
        self.particle_y_velocity = particle_y_velocity
        self.slide = slide
        self.num_inner = 15 # 要修正
        self.dr = 0.2 # 要修正

        self.r = []
        self.f = []
        self.r_test = []
        self.f_test = []

        COLORS = ["#DCBCBC", "#C79999", "#B97C7C", "#A25050", "#8F2727",
                  "#7C0000",     "#DCBCBC20", "#8F272720", "#00000060"]
        self.COLOR = {i[0]: i[1] for i in zip(['light', 'light_highlight', 'mid',  'mid_highlight',
                                               'dark', 'dark_highlight', 'light_trans', 'dark_trans', 'superfine'], COLORS)}
    
    def make_r_surface(self, num, use_inner=False):
        θ_start = 0
        θ_end = 2*np.pi
        θ = np.linspace(θ_start, θ_end, num)
        if not use_inner:
            r_mesh, θ_mesh = np.meshgrid(self.particle_radius, θ)
        else:
            radius_range = np.linspace(0.03, self.particle_radius, 3)
            r_mesh, θ_mesh = np.meshgrid(radius_range, θ)
        if not self.random_arrange:
            pass
        else:
            np.random.seed(42)
            r_mesh -= (np.random.random_sample(r_mesh.shape))*0.01
            np.random.seed(42)
            θ_mesh += (np.random.random_sample(θ_mesh.shape)-0.5)*0.2
        xx, yy = r_mesh*np.cos(θ_mesh), r_mesh*np.sin(θ_mesh)
        x = xx.reshape(-1)+self.particle_center
        y = yy.reshape(-1)+self.particle_center
        r = np.stack([x, y], axis=1)
        return r
    
    def get_index_in_domain(self, r, radius_min=None):
        rx = r[:, 0]-self.particle_center
        ry = r[:, 1]-self.particle_center
        if not radius_min:
            index_in_domain = np.where(rx**2+ry**2 > self.particle_radius**2)[0]
        else:
            index_in_domain = np.where(rx**2+ry**2 > radius_min**2)[0]
        return index_in_domain
    
    def delete_out_domain(self, r, radius_min=None):
        index_in_domain = self.get_index_in_domain(r, radius_min)
        return r[index_in_domain]
    
    def make_r_mesh_rectangular(self, numx, numy, pad, radius_min=None):
        x_start = 0.+pad
        x_end = self.L-pad
        y_start = 0.+pad
        y_end = self.H-pad

        r = self.make_r_mesh(x_start, x_end, y_start, y_end, numx, numy)
        r = self.delete_out_domain(r, radius_min)

        return r
    
    def make_r_mesh_circular(self, num_per_side, dr=0.1):
        x_start = self.particle_center-dr
        x_end = self.particle_center+dr
        y_start = x_start
        y_end = x_end

        r = self.make_r_mesh(x_start, x_end, y_start, y_end, num_per_side, num_per_side)

        index_out_of_dr = self.get_index_in_domain(r, radius_min=self.particle_radius+dr)
        # index_in_dr = (r[:, 0] != r[index_out_of_dr][:, 0]) | (r[:, 1] != r[index_out_of_dr][:, 1])
        index_in_dr = np.arange(0, len(r), 1, dtype=int)
        index_in_dr = np.delete(index_in_dr, index_out_of_dr)
        bool_in_dr = np.zeros(len(r), dtype=bool)
        bool_in_dr[index_in_dr] = True
        index_out_of_radius = self.get_index_in_domain(r)
        bool_out_of_radius = np.zeros(len(r), dtype=bool)
        bool_out_of_radius[index_out_of_radius] = index_out_of_radius
        index_in_domain = (bool_in_dr & bool_out_of_radius)
        
        return r[index_in_domain]
    
    def make_r_mesh_mixed(self, num_inner, num_outer, dr, pad):
        radius_min = self.particle_radius+dr
        r_circular = self.make_r_mesh_circular(num_inner, dr)
        r_rectangular = self.make_r_mesh_rectangular(num_outer, num_outer, pad, radius_min)
        r = np.concatenate([r_circular, r_rectangular])
        return r
        
    
    def generate_training_data(self, u_num=None, f_num=None, f_pad=None, div_num=None, div_pad=None, difu_num=None, difp_num=None):
        self.r = []
        self.f = []
        self.generate_u(u_num)
        self.generate_difu(difu_num)
        self.generate_f(f_num, f_pad)
        self.generate_div(div_num, div_pad)
        self.generate_difp(difp_num)
        return self.r, self.f

    def generate_test(self, test_num=None):
        try:
            with open('/home/ogawa_kenta/template_data/0308_drag_test_676.pickle', 'rb') as file:
                save_dict = pickle.load(file)
        except:
            raise FileNotFoundError('File was not found')
        r, ux, uy = save_dict['r'], save_dict['ux'], save_dict['uy']

        self.r_test += [r, r]
        self.f_test += [ux, uy]
        return self.r_test, self.f_test
        

    def generate_u(self, u_num):
        r_ux = self.make_r_surface(u_num)
        ux = np.zeros(len(r_ux))

        r_uy = self.make_r_surface(u_num)
        uy = np.full(len(r_uy), self.particle_y_velocity)

        self.r += [r_ux, r_uy]
        self.f += [ux, uy]

    def generate_f(self, f_num, f_pad, force=0.):
        r_fx = self.make_r_mesh_mixed(self.num_inner, f_num, self.dr, f_pad)
        r_fy = self.make_r_mesh_mixed(self.num_inner, f_num, self.dr, f_pad)
        fx = np.full(len(r_fx), force)
        fy = np.full(len(r_fy), force)

        self.r += [r_fx, r_fy]
        self.f += [fx, fy]

    def generate_div(self, div_num, div_pad, div=0.):
        r_div = self.make_r_mesh_mixed(self.num_inner, div_num, self.dr, div_pad)
        div = np.full(len(r_div), div)

        self.r += [r_div]
        self.f += [div]

    def generate_difu(self, difu_num):
        r_difux_x, r_difux_y, difux_x, difux_y = self.generate_dif(difu_num)
        r_difuy_x, r_difuy_y, difuy_x, difuy_y = self.generate_dif(difu_num)
        self.r += [r_difux_x, r_difux_y, r_difux_x, r_difux_y]
        self.f += [r_difuy_x, r_difuy_y, difuy_x, difuy_y]
        
        
    def generate_dif(self, dif_num):
        r_dif_x = self.make_r_mesh(0, self.L, self.slide, self.H-self.slide, 1, dif_num)
        r_dif_y = self.make_r_mesh(0+self.slide, self.L-self.slide, 0., self.H, dif_num, 1)
        dif_x = np.zeros(len(r_dif_x))
        dif_y = np.zeros(len(r_dif_y))
        return r_dif_x, r_dif_y, dif_x, dif_y


    def generate_difp(self, difp_num):
        difp_num = int(difp_num/2)
        r_difp_x, r_difp_y, difp_x, difp_y = self.generate_dif(difp_num)
        self.r += [r_difp_x, r_difp_y]
        self.f += [difp_x, difp_y]

    def plot_train(self, save=False, path=None, show=False):
        num_surface = 100
        r_surface = self.make_r_surface(num_surface)

        ms = 5
        ms2 = 2
        fig, axs = plt.subplots(
            figsize=(3*3, 3*2), nrows=2, ncols=3, sharex=True, sharey=True)
        clrs = [self.COLOR['dark_highlight'], self.COLOR['superfine'], self.COLOR['superfine'], self.COLOR['dark_highlight'],self.COLOR['superfine'], self.COLOR['superfine'], 'darkblue', 'darkblue', 'darkblue', self.COLOR['superfine'], self.COLOR['superfine']]
        lbls = ['ux', 'uy', 'p', 'fx', 'fy', 'div']
        axes = axs.reshape(-1)
        for i, ax in enumerate(axes):
            if i == 0 or i == 1:
                ax.set_title(lbls[i])
                ax.plot(self.r[i][:, 0], self.r[i][:, 1], ls='None', marker='o', color=clrs[i], ms=3)
                index = i*2 + 2
                ax.plot(self.r[index][:, 0], self.r[index][:, 1], ls='None', marker='o', color=self.COLOR['superfine'], ms=ms)
                index = i*2 + 3 
                ax.plot(self.r[index][:, 0], self.r[index][:, 1], ls='None', marker='o', color=self.COLOR['superfine'], ms=ms)
            elif i == 2:
                ax.set_title(lbls[i])
                index = 9
                ax.plot(self.r[index][:, 0], self.r[index][:, 1], ls='None', marker='o', color=clrs[index], ms=ms)
                index = 10
                ax.plot(self.r[index][:, 0], self.r[index][:, 1], ls='None', marker='o', color=clrs[index], ms=ms)
            else:
                index = i+3
                ax.set_title(lbls[i])
                ax.plot(self.r[index][:, 0], self.r[index][:, 1], ls='None', marker='o', color=clrs[index], ms=ms2)
            ax.set_aspect('equal', adjustable='box')
            ax.plot(r_surface[:, 0], r_surface[:, 1], color='k')
        if save:
            dir_path = f'{path}/fig'
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            fig.savefig(f'{dir_path}/train.png')
        if show:
            plt.show()
        plt.clf()
        plt.close()
    def plot_test(self):
        pass

