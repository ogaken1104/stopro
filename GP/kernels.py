import jax.numpy as jnp
from jax import jit


#
def distance(r1, r2, lbox=2 * jnp.pi):
    """Compute distance vector between two positions assuming PBC

    Args:
        r1 : center position, origin
        r2 : second position
        lbox : box dimensions

    Returns:
        float : distance vector r2 - r1 wrapped around box"""
    r12 = r2 - r1
    return r12 - jnp.round(r12 / lbox) * lbox


def K_periodic(x1, x2, logl, repitition=2 * jnp.pi):
    return jnp.exp(-2 * (jnp.sin(jnp.pi*jnp.abs(x1-x2)/repitition)*jnp.exp(-logl))**2)


def K_1d_periodic(x1, x2, θ):
    logγ, logl = θ
    return jnp.exp(logγ)*K_periodic(x1, x2, logl)

#


def K_SquareExp(x1, x2, logl):
    return jnp.exp(-0.5*((x1-x2)*jnp.exp(-logl))**2)


def K_SquareExp_distance(x1, x2, logl):
    return jnp.exp(-0.5*(distance(x1, x2)*jnp.exp(-logl))**2)

#


def K_1d_SquareExp(r1, r2, θ):
    logγ, logl = θ
    return jnp.exp(logγ)*K_SquareExp(r1, r2, logl)


def K_1d_SquareExp_distance(r1, r2, θ):
    logγ, logl = θ
    return jnp.exp(logγ)*K_SquareExp_distance(r1, r2, logl)


def K_2d_SquareExp_Add(r1, r2, θ):
    logγ, loglx, logly = θ
    return jnp.exp(logγ)*(K_SquareExp(r1[0], r2[0], loglx)+K_SquareExp(r1[1], r2[1], logly))


def K_2d_SquareExp_Pro(r1, r2, θ):
    logγ, loglx, logly = θ
    return jnp.exp(logγ)*(K_SquareExp(r1[0], r2[0], loglx)*K_SquareExp(r1[1], r2[1], logly))


######### テスト： ハイパーパラメータをそのままの値で与える場合 #################
def K_SquareExp_noexp(x1, x2, l):
    return jnp.exp(-0.5*((x1-x2)/l)**2)


def K_2d_SquareExp_Pro_noexp(r1, r2, θ):
    γ, lx, ly = θ
    return γ*(K_SquareExp_noexp(r1[0], r2[0], lx)*K_SquareExp_noexp(r1[1], r2[1], ly))

############# テスト： ハイパーパラメータをソフトプラス関数で変換したものを与える場合　#############


def K_SquareExp_softplus(x1, x2, l):
    return jnp.exp(-0.5*((x1-x2)/(jnp.log(1+jnp.exp(l))))**2)


def K_2d_SquareExp_Pro_softplus(r1, r2, θ):
    γ, lx, ly = θ
    return jnp.log(1+jnp.exp(γ))*(K_SquareExp_softplus(r1[0], r2[0], lx)*K_SquareExp_softplus(r1[1], r2[1], ly))

###########################################################################


def K_Matern72(x1, x2, logl):
    r = jnp.sqrt(7.0)*jnp.abs(x1-x2)*jnp.exp(-logl)
    return (1.0 + r + (2.0/5.0)*r**2.0 + (1.0/15.0)*r**3.0)*jnp.exp(-r)


def K_2d_Matern72_Add(r1, r2, θ):
    logγ, loglx, logly = θ
    return jnp.exp(logγ)*(K_Matern72(r1[0], r2[0], loglx)+K_Matern72(r1[1], r2[1], logly))


def K_2d_Matern72_Pro(r1, r2, θ):
    logγ, loglx, logly = θ
    return jnp.exp(logγ)*(K_Matern72(r1[0], r2[0], loglx)*K_Matern72(r1[1], r2[1], logly))
###########################################################################


def K_Matern92(x1, x2, logl):
    r = jnp.abs(x1-x2)*jnp.exp(-logl)
    return (1.0 + 3*r + 27/7*r**2 + 18/7*r**3 + 27/35*r**4)*jnp.exp(-3*r)


def K_2d_Matern92_Add(r1, r2, θ):
    logγ, loglx, logly = θ
    return jnp.exp(logγ)*(K_Matern92(r1[0], r2[0], loglx)+K_Matern92(r1[1], r2[1], logly))


def K_2d_Matern92_Pro(r1, r2, θ):
    logγ, loglx, logly = θ
    return jnp.exp(logγ)*(K_Matern92(r1[0], r2[0], loglx)*K_Matern92(r1[1], r2[1], logly))
#############################################################################


def K_Matern32(x1, x2, logl):
    r = jnp.sqrt(3.0)*jnp.abs(x1-x2)*jnp.exp(-logl)
    return (1.0 + r)*jnp.exp(-r)


def K_2d_Matern32_Add(r1, r2, θ):
    logγ, loglx, logly = θ
    return jnp.exp(logγ)*(K_Matern32(r1[0], r2[0], loglx)+K_Matern32(r1[1], r2[1], logly))


def K_2d_Matern32_Pro(r1, r2, θ):
    logγ, loglx, logly = θ
    return jnp.exp(logγ)*(K_Matern32(r1[0], r2[0], loglx)*K_Matern32(r1[1], r2[1], logly))

##############################################################################


def K_Matern52(x1, x2, logl):
    r = jnp.sqrt(5.0)*jnp.abs(x1-x2)*jnp.exp(-logl)
    return (1.0 + r + (1.0/3.0)*r**2.0)*jnp.exp(-r)


def K_2d_Matern52_Add(r1, r2, θ):
    logγ, loglx, logly = θ
    return jnp.exp(logγ)*(K_Matern52(r1[0], r2[0], loglx)+K_Matern52(r1[1], r2[1], logly))


def K_2d_Matern52_Pro(r1, r2, θ):
    logγ, loglx, logly = θ
    return jnp.exp(logγ)*(K_Matern52(r1[0], r2[0], loglx)*K_Matern52(r1[1], r2[1], logly))


def K_NN(x0, x1, θ):
    σ0, σ = θ
    return 2.0/jnp.pi*jnp.arcsin(2.0*(σ0 + σ*x0*x1)/jnp.sqrt((1.0 + 2.0*(σ0 + σ*x0**2.0)) * (1.0 + 2.0*(σ0 + σ*x1**2))))


def K_2d_NN_Add(r1, r2, θ):
    σ0, σx, σy = jnp.exp(θ)
    return K_NN(r1[0], r2[0], [σ0, σx])+K_NN(r1[1], r2[1], [σ0, σy])


def K_2d_NN_Pro(r1, r2, θ):
    σ0, σx, σy = jnp.exp(θ)
    return K_NN(r1[0], r2[0], [σ0, σx])*K_NN(r1[1], r2[1], [σ0, σy])


def K_RQ(x1, x2, length_scale, α):
    return (1 + ((((x1-x2)/length_scale)**2)/(2*α)))**(-α)


def K_2d_RQ_Add(r1, r2, θ):
    γ, lx, ly, α = jnp.exp(2*θ)
    return γ*(K_RQ(r1[0], r2[0], lx, α)+K_RQ(r1[1], r2[1], ly, α))


def K_2d_RQ_Pro(r1, r2, θ):
    γ, lx, ly, α = jnp.exp(2*θ)
    return γ*(K_RQ(r1[0], r2[0], lx, α)*K_RQ(r1[1], r2[1], ly, α))


def K_2d_x_Periodic_y_SE_Pro(r1, r2, θ, lbox):
    logγ, loglx, logly = θ
    return jnp.exp(logγ)*(K_periodic(r1[0], r2[0], loglx, repitition=lbox) * K_SquareExp(r1[1], r2[1], logly))


def K_2d_x_Periodic_y_SE_Add(r1, r2, θ, lbox):
    logγ, loglx, logly = θ
    return jnp.exp(logγ)*(K_periodic(r1[0], r2[0], loglx, repitition=lbox) + K_SquareExp(r1[1], r2[1], logly))


def define_kernel(kernel_type, kernel_form='product', input_dim=2, lbox=None, distance_func=False):

    if input_dim == 2:
        if kernel_form == 'additive':
            if kernel_type == 'se':
                base_kernel = K_2d_SquareExp_Add
            elif kernel_type == 'mt92':
                base_kernel = K_2d_Matern92_Add
            elif kernel_type == 'mt72':
                base_kernel = K_2d_Matern72_Add
            elif kernel_type == 'mt52':
                base_kernel = K_2d_Matern52_Add
            elif kernel_type == 'mt32':
                base_kernel = K_2d_Matern32_Add
            elif kernel_type == 'nn':
                base_kernel = K_2d_NN_Add
            elif kernel_type == 'rq':
                base_kernel = K_2d_RQ_Add
            elif kernel_type == 'x_periodic_y_se':
                def base_kernel(r1, r2, θ): return K_2d_x_Periodic_y_SE_Add(
                    r1, r2, θ, lbox)
        elif kernel_form == 'product':
            if kernel_type == 'se':
                base_kernel = K_2d_SquareExp_Pro
            elif kernel_type == 'mt92':
                base_kernel = K_2d_Matern92_Pro
            elif kernel_type == 'mt72':
                base_kernel = K_2d_Matern72_Pro
            elif kernel_type == 'mt52':
                base_kernel = K_2d_Matern52_Pro
            elif kernel_type == 'mt32':
                base_kernel = K_2d_Matern32_Pro
            elif kernel_type == 'nn':
                base_kernel = K_2d_NN_Pro
            elif kernel_type == 'rq':
                base_kernel = K_2d_RQ_Pro
            elif kernel_type == 'se_noexp':
                #                 print('OK')
                base_kernel = K_2d_SquareExp_Pro_noexp
            elif kernel_type == 'se_softplus':
                base_kernel = K_2d_SquareExp_Pro_softplus
            elif kernel_type == 'x_periodic_y_se':
                def base_kernel(r1, r2, θ): return K_2d_x_Periodic_y_SE_Pro(
                    r1, r2, θ, lbox)
    elif input_dim == 1:
        if kernel_type == 'se':
            if distance_func:
                base_kernel = K_1d_SquareExp_distance
            else:
                base_kernel = K_1d_SquareExp
        elif kernel_type == 'periodic':
            base_kernel = K_1d_periodic

    return base_kernel
