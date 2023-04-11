import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from stopro.data_generator.stokes_2D_generator import StokesDataGenerator


# adimentionalize so that H,ρ,η=1.0
class Cylinder(StokesDataGenerator):
    def __init__(self, Re, L, Δp, pin, radius, center, slide):
        self.Re = Re
        self.L = L
        self.Δp = Δp
        self.pin = pin
        self.radius = radius
        self.center = center
        self.slide = slide
        self.f_ux_in = lambda r: -0.5*self.Δp*self.Re*r[:, 1]*(1-r[:, 1])
        self.f_p = lambda r: pin+self.Δp*r[:, 0]
#         self.f_uy=lambda r :0.
#         self.f_p=lambda r:self.Δp/self.L*r[0]+self.pin
        self.r = []
        self.f = []
        self.get_numerical()

    def get_numerical(self):
        with open('../data/cylinder.pickle', 'rb') as f:
            value = pickle.load(f)
        r_test, f_test = value
        r_ux, r_uy, r_p = r_test
        ux, uy, p = f_test

        # uxの出口を取得
        split = 7
        r_uxx = r_ux[:, 0]
        index = np.where(r_uxx == 1)
        r_ux2 = r_ux[index]
        ux2 = ux[index]
        # 出口の速度の端の点は必ずわかるようにする
        self.r_ux_train = np.concatenate(
            [r_ux2[::split], [r_ux2[9]], [r_ux2[-2]]])
        self.ux_train = np.concatenate([ux2[::split], [ux2[9]], [ux2[-2]]])

        # uyの出口を取得
        r_uyx = r_uy[:, 0]
        index = np.where(r_uyx == 1)
        r_uy2 = r_uy[index]
        uy2 = uy[index]
        self.r_uy_train = np.concatenate(
            [r_uy2[::split], [r_uy2[9]], [r_uy2[-2]]])
        self.uy_train = np.concatenate([uy2[::split], [uy2[9]], [uy2[-2]]])

        # pの入り口を取得
        r_px = r_p[:, 0]
        index = np.where(r_px == 0)
        r_p2 = r_p[index]
        p2 = p[index]
        self.r_p_train = r_p2[::2]
        self.p_train = p2[::2]

    def make_r_surface(self, num, inner=False):
        θ_start = 0
        θ_end = 2*np.pi
        θ = np.linspace(θ_start, θ_end, num)
        if not inner:
            r_mesh, θ_mesh = np.meshgrid(self.radius, θ)
        else:
            radius = np.linspace(0.03, self.radius, 3)
            r_mesh, θ_mesh = np.meshgrid(radius, θ)
        r_mesh -= (np.random.random_sample(r_mesh.shape))*0.01
        θ_mesh += (np.random.random_sample(θ_mesh.shape)-0.5)*0.2
        xx, yy = r_mesh*np.cos(θ_mesh), r_mesh*np.sin(θ_mesh)
        x = xx.reshape(-1)+self.center
        y = yy.reshape(-1)+self.center
        r = np.stack([x, y], axis=1)
        return r

    def delete_circle(self, r):
        rx = r[:, 0]-self.center
        ry = r[:, 1]-self.center
        index = np.where(rx**2+ry**2 > self.radius**2)
        return r[index]
    # generation of training data

    def generate_u(self, u_num, u_b=0.):
        """
         premise: ux,uy values are taken at same points  
        """
        inner = False
        inlet = False
        add = True
        # 入り口のみ数値計算結果を用いる場合
        if inlet:
            u_num = int(u_num/5)
            r_ux_wall = self.make_r_mesh_random(
                0.+self.slide, self.L-self.slide, 0., 1., u_num-2, 2, self.slide, 'x')
            r_ux_inlet = self.make_r_mesh_random(
                0., 0., 0.05, 0.95, 1, u_num, self.slide, 'y')
            r_ux_wall = np.concatenate(
                [r_ux_wall, np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])])
            r_ux_surface = self.make_r_surface(u_num*2, inner)
            r_uy_wall = self.make_r_mesh_random(
                0.+self.slide, self.L-self.slide, 0., 1., u_num-2, 2, self.slide, 'x')
            r_uy_inlet = self.make_r_mesh_random(
                0., 0., 0.05, 0.95, 1, u_num, self.slide, 'y')
            r_uy_wall = np.concatenate(
                [r_uy_wall, np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])])
            r_uy_surface = self.make_r_surface(u_num*2, inner)
            r_ux = np.concatenate([r_ux_inlet, r_ux_wall, r_ux_surface])
            r_uy = np.concatenate([r_uy_inlet, r_uy_wall, r_uy_surface])
            ux_inlet = -0.5*self.Δp*self.Re * \
                r_ux_inlet[:, 1]*(1-r_ux_inlet[:, 1])
            ux_bound = np.zeros(len(r_ux_wall)+len(r_ux_surface))
            ux = np.concatenate([ux_inlet, ux_bound])
            uy = np.zeros(len(r_uy))
        # 入り口、出口共に数値計算結果を用いる場合
        elif add:
            u_num = int(u_num/6)
            r_ux_wall = self.make_r_mesh_random(
                0.+self.slide, self.L-self.slide, 0., 1., u_num-2, 2, self.slide, 'x')
            r_ux_inlet = self.make_r_mesh_random(
                0., 0., 0.05, 0.95, 1, u_num, self.slide, 'y')
            r_ux_wall = np.concatenate(
                [r_ux_wall, np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])])
            r_ux_surface = self.make_r_surface(u_num*2, inner)
            r_uy_wall = self.make_r_mesh_random(
                0.+self.slide, self.L-self.slide, 0., 1., u_num-2, 2, self.slide, 'x')
            r_uy_inlet = self.make_r_mesh_random(
                0., 0., 0.05, 0.95, 1, u_num, self.slide, 'y')
            r_uy_wall = np.concatenate(
                [r_uy_wall, np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])])
            r_uy_surface = self.make_r_surface(u_num*2, inner)
            index_ux_train = np.arange(len(self.r_ux_train))
            index_uy_train = np.arange(len(self.r_uy_train))
            np.random.shuffle(index_ux_train)
            np.random.shuffle(index_uy_train)
            r_ux = np.concatenate(
                [r_ux_inlet, r_ux_wall, r_ux_surface, self.r_ux_train[index_ux_train[:u_num]]])
            r_uy = np.concatenate(
                [r_uy_inlet, r_uy_wall, r_uy_surface, self.r_uy_train[index_uy_train[:u_num]]])
            ux_inlet = -0.5*self.Δp*self.Re * \
                r_ux_inlet[:, 1]*(1-r_ux_inlet[:, 1])
            ux_bound = np.zeros(len(r_ux_wall)+len(r_ux_surface))
            ux = np.concatenate(
                [ux_inlet, ux_bound, self.ux_train[index_ux_train[:u_num]]])
            uy_bound = np.zeros(
                len(r_uy_inlet)+len(r_uy_wall)+len(r_uy_surface))
            uy = np.concatenate(
                [uy_bound, self.uy_train[index_uy_train[:u_num]]])
        # 壁と接している部分のみ値を用いる場合
        else:
            u_num = int(u_num/3)
            r_ux_wall = self.make_r_mesh_random(
                0.+self.slide, self.L-self.slide, 0., 1., u_num-2, 2, self.slide, 'x')
            r_ux_wall = np.concatenate(
                [r_ux_wall, np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])])
            r_ux_surface = self.make_r_surface(u_num, inner)
            r_uy_wall = self.make_r_mesh_random(
                0.+self.slide, self.L-self.slide, 0., 1., u_num-2, 2, self.slide, 'x')
            r_uy_wall = np.concatenate(
                [r_uy_wall, np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])])
            r_uy_surface = self.make_r_surface(u_num, inner)
            r_ux = np.concatenate([r_ux_wall, r_ux_surface])
            r_uy = np.concatenate([r_uy_wall, r_uy_surface])
            ux = np.zeros(len(r_ux))
            uy = np.zeros(len(r_uy))
        self.r += [r_ux, r_uy]
        self.f += [ux, uy]

    def generate_test(self, num):
        r = self.make_r_mesh(0., self.L, 0., 1., num, num)
        r_test = [r]*3
        ux_test = self.f_ux_in(r)
        uy_test = np.zeros(len(r))
        p_test = self.f_p(r)
        f_test = [ux_test, uy_test, p_test]
        return r_test, f_test

    def generate_p(self, p_num):
        inlet = True
        uniform = False
        single = False
        only_outlet = False
        p_num = int(p_num/2)
#         pout=self.pin+self.Δp*self.L
#         x_p0=np.full(p_num,0)
#         x_p1=np.full(p_num,self.L)
#         y_p=np.linspace(0.,1.,p_num)
#         r_p=np.stack([np.concatenate([x_p0,x_p1]),np.concatenate([y_p,y_p])],axis=1)
#         p0=np.full(p_num,self.pin)
#         p1=np.full(p_num,pout)
#         p=np.concatenate([p0,p1])
        # 入り口が1、出口が0で均一とした場合（数値計算でこの値を求めることが難しかった）
        if uniform:
            r_p = self.make_r_mesh_random(
                0., 1., 0.05, 0.95, 2, p_num-2, self.slide, 'y')
            r_p = np.concatenate(
                [r_p, np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])])
            p = self.f_p(r_p)
        # 出口の一点でのみ用いる場合（入り口付近の値を予測することが難しかった）
        elif single:
            r_p = np.array([[1., 0.5]])
            p = self.f_p(r_p)
        elif only_outlet:
            r_p = self.make_r_mesh_random(
                1., 1., 0.05, 0.95, 2, p_num-2, self.slide, 'y')
            r_p = np.concatenate([r_p, np.array([[1., 0.], [1., 1.]])])
            p = self.f_p(r_p)
        # 入り口では数値計算結果を、出口では0を用いるとき：数値計算結果を使えていない！
        if inlet:
            r_p = self.make_r_mesh_random(
                1., 1., 0.05, 0.95, 1, p_num-2, self.slide, 'y')
            r_p = np.concatenate([r_p, np.array([[1., 0.], [1., 1.]])])
            p = self.f_p(r_p)
            index_ptrain = np.arange(len(self.r_p_train))
            index_ptrain = index_ptrain[1:-1]
            np.random.seed(2)
            np.random.shuffle(index_ptrain)
            r_p = np.concatenate(
                [r_p, [self.r_p_train[0], self.r_p_train[-1]], self.r_p_train[index_ptrain[:(p_num-2)]]])
            p = np.concatenate(
                [p, [self.p_train[0], self.p_train[1]], self.p_train[index_ptrain[:(p_num-2)]]])
        self.r += [r_p]
        self.f += [p]

    def generate_f(self, f_num, fx_pad, force=0.):
        """
         premise: ux,uy values are taken at same points  
        """

        f_num_gen = int(np.sqrt(f_num) * 1.3)
        x_start = 0.+fx_pad
        x_end = self.L-fx_pad
        y_start = 0.+fx_pad
        y_end = 1.0-fx_pad
        r = r = self.make_r_mesh(
            x_start, x_end, y_start, y_end, f_num_gen, f_num_gen)
#         x_fx=np.linspace(x_start,x_end,total_num)
#         y_fx=np.linspace(y_start,y_end,total_num)
#         xx,yy=np.meshgrid(x_fx,y_fx)
#         r_fx_all=np.stack([xx.reshape(-1),yy.reshape(-1)],axis=1)
#         np.random.seed(10)
#         r_fx=np.random.shuffle(r_fx_all)
#         r_fx=r_fx_all[:f_num,:f_num_w]
#         r_fy=r_fx_all[-f_num:,-f_num_w:]
        r_fx = (np.random.random_sample(r.shape)-0.5)*self.slide+r
        r_fy = (np.random.random_sample(r.shape)-0.5)*self.slide+r
#         fx=np.full(f_num,force)
        r_fx = self.delete_circle(self.delete_out(r_fx))
        r_fy = self.delete_circle(self.delete_out(r_fy))
        total_fx = np.arange(len(r_fx))
        total_fy = np.arange(len(r_fy))
        np.random.shuffle(total_fx)
        np.random.shuffle(total_fy)
        r_fx = r_fx[total_fx[:f_num]]
        r_fy = r_fy[total_fy[:f_num]]

#         r_fx=np.random.choice(r_fx,size=f_num)
#         r_fy=np.random.choice(r_fy,size=f_num)
        fx = np.full(len(r_fx), force)
        fy = np.full(len(r_fy), force)
        self.r += [r_fx, r_fy]
        self.f += [fx, fy]

    def generate_div(self, div_num, div_pad, divu=0.):

        div_num_gen = int(np.sqrt(div_num) * 1.3)
        x_start = 0.+div_pad
        x_end = self.L-div_pad
        y_start = 0.+div_pad
        y_end = 1.0-div_pad
        r = self.make_r_mesh(x_start, x_end, y_start,
                             y_end, div_num_gen, div_num_gen)
        r_div = (np.random.random_sample(r.shape)-0.5)*self.slide+r
        r_div = self.delete_circle(self.delete_out(r_div))
        total_div = np.arange(len(r_div))
        np.random.shuffle(total_div)
        r_div = r_div[total_div[:div_num]]
        div = np.full(len(r_div), divu)
        self.r += [r_div]
        self.f += [div]


#     def generate_test_data(self,u_num_test):
#         x_ux=np.full(u_num_test,self.L/2.)
#         y_ux=np.linspace(-1.,1.,u_num_test)
#         r_ux=np.stack([x_ux,y_ux],axis=1)
#         return [r_ux]*3
