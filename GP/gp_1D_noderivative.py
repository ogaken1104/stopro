import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jit, vmap

from stopro.GP.gp import GPmodel
from stopro.GP.kernels import define_kernel


class GP1DNoDerivative(GPmodel):
    def __init__(self, model_param):
        kernel_type = model_param["kernel_type"]
        kernel_form = model_param["kernel_form"]
        distance_func = model_param["distance_func"]

        def outermap(f):
            return vmap(vmap(f, in_axes=(None, 0, None)), in_axes=(0, None, None))

    def setup_training_and_predicting_functions(self, kernel_type, kernel_form):
        def outermap(f):
            return vmap(vmap(f, in_axes=(None, 0, None)), in_axes=(0, None, None))

        Kernel = define_kernel(kernel_type, kernel_form, input_dim=1)
        K = outermap(Kernel)

        def trainingK_all(θ, train_pts):
            """
            Args :
              θ  : kernel hyperparameters
             args: training points r_ux,r_uy,r_p,r_fx,r_fy,r_div
            """
            θyy = θ

            def Kyy(r, rp):
                return K(r, rp, θyy)

            Ks = [
                [Kyy],
            ]

            return self.calculate_K_symmetric(train_pts, Ks)

        def mixedK_all(θ, test_pts, train_pts):
            θyy = θ

            def Kyy(r, rp):
                return K(r, rp, θyy)

            Ks = [[Kyy]]
            return self.calculate_K_asymmetric(train_pts, test_pts, Ks)

        def testK_all(θ, r_test):
            θyy = θ

            def Kyy(r, rp):
                return K(r, rp, θyy)

            Ks = [[Kyy]]
            return self.calculate_K_symmetric(r_test, Ks)

        def trainingFunction_all(θ, *args):
            """Returns minus log-likelihood given Kernel hyperparamters θ and training data args
            args = velocity position, velocity average, velocity values,
                   force position, force average, force values,
                   jiggle parameter
            """
            # r,μ,f,ϵ=args
            r, μ, f, ϵ = args
            r_num = len(r)
            for i in range(r_num):
                if i == 0:
                    δy = jnp.array(f[i] - μ[i])
                else:
                    δy = jnp.concatenate([δy, f[i] - μ[i]], 0)
            Σ = trainingK_all(θ, r)
            return self.logpGP(δy, Σ, ϵ)

        def predictingFunction_all(θ, *args):
            """Returns conditional posterior average and covariance matrix given Kernel hyperparamters θ  and test and training data
            args = test velocity position, test velocity average,
                   training velocity position, training velocity average, training velocity values
                   training force position, training force average, training force values
                   jiggle parameter

            Returns
            -----------------
            μpost=[μux,μuy,μp]
            Σpost=[Σux,Σuy,Σp]
            """
            r_test, μ_test, r_train, μ, f_train, ϵ = args
            nb = 0
            for r in r_train:
                nb += len(r)
            Σbb = trainingK_all(θ, r_train)
            Σab = mixedK_all(θ, r_test, r_train)
            Σaa = testK_all(θ, r_test)
            for i in range(len(r_train)):
                if i == 0:
                    δfb = jnp.array(f_train[i] - μ[i])
                else:
                    δfb = jnp.concatenate([δfb, f_train[i] - μ[i]])
                    # create single training array, with velocities and forces (second derivatives)
            #         print(f'δy={δy}')
            #         print(f'Σ={Σ}')
            μposts, Σposts = self.postGP(δfb, Σaa, Σab, Σbb, ϵ)
            # seperate μpost,Σpost to 3 section (ux,uy,p)
            sec0 = 0
            sec1 = 0
            μpost = []
            Σpost = []
            for i in range(len(r_test)):
                sec1 += len(r_test[i])
                μpost.append(μposts[sec0:sec1])
                Σpost.append(Σposts[sec0:sec1, sec0:sec1])
                sec0 += len(r_test[i])
                # 一応解決ちょっと疑問残る
                μpost[i] += μ_test[i]
            return μpost, Σpost

        return jit(trainingFunction_all), jit(predictingFunction_all)
