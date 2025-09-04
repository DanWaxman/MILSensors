"""
Custom kernel implementations based on the BayesNewton library.

This module contains specialized kernel implementations that extend the functionality
of the BayesNewton library. These kernels implement additional methods and functionality
that may not be present in the official BayesNewton versions.

The kernels in this file are particularly focused on state-space representations
of Gaussian processes, including quasi-periodic and subband formulations.
"""

import jax.numpy as jnp
import bayesnewton
import objax
from bayesnewton.utils import (
    softplus,
    softplus_inv,
    square_distance,
    scaled_squared_euclid_dist,
    rotation_matrix,
)
from jax.scipy.linalg import block_diag
from warnings import warn
from jax import vmap
from jax.scipy.linalg import cho_factor, cho_solve


class QuasiPeriodicMatern32(bayesnewton.kernels.Kernel):
    """
    Quasi-periodic kernel in SDE form (product of Periodic and Matern-3/2).
    Hyperparameters:
        variance, σ²
        lengthscale of Periodic, l_p
        period, p
        lengthscale of Matern, l_m
    The associated continuous-time state space model matrices are constructed via
    a sum of cosines times a Matern-3/2.
    """

    def __init__(
        self,
        variance=1.0,
        lengthscale_periodic=1.0,
        period=1.0,
        lengthscale_matern=1.0,
        order=6,
    ):
        self.transformed_lengthscale_periodic = objax.TrainVar(
            jnp.array(softplus_inv(lengthscale_periodic))
        )
        self.transformed_variance = objax.TrainVar(jnp.array(softplus_inv(variance)))
        self.transformed_period = objax.TrainVar(jnp.array(softplus_inv(period)))
        self.transformed_lengthscale_matern = objax.TrainVar(
            jnp.array(softplus_inv(lengthscale_matern))
        )
        super().__init__()
        self.name = "Quasi-periodic Matern-3/2"
        self.order = order
        self.igrid = jnp.meshgrid(
            jnp.arange(self.order + 1), jnp.arange(self.order + 1)
        )[1]
        factorial_mesh_K = jnp.array(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                [6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0],
                [24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0],
                [120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0],
                [720.0, 720.0, 720.0, 720.0, 720.0, 720.0, 720.0],
            ]
        )
        b = jnp.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [2.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 6.0, 0.0, 2.0, 0.0, 0.0, 0.0],
                [6.0, 0.0, 8.0, 0.0, 2.0, 0.0, 0.0],
                [0.0, 20.0, 0.0, 10.0, 0.0, 2.0, 0.0],
                [20.0, 0.0, 30.0, 0.0, 12.0, 0.0, 2.0],
            ]
        )
        self.b_fmK_2igrid = b * (1.0 / factorial_mesh_K) * (2.0**-self.igrid)

    @property
    def variance(self):
        return softplus(self.transformed_variance.value)

    @property
    def lengthscale_periodic(self):
        return softplus(self.transformed_lengthscale_periodic.value)

    @property
    def lengthscale_matern(self):
        return softplus(self.transformed_lengthscale_matern.value)

    @property
    def period(self):
        return softplus(self.transformed_period.value)

    def K(self, X, X2):
        r_per = (
            jnp.pi * jnp.sqrt(jnp.maximum(square_distance(X, X2), 1e-36)) / self.period
        )
        k_per = jnp.exp(-0.5 * jnp.square(jnp.sin(r_per) / self.lengthscale_periodic))
        sqrt3 = jnp.sqrt(3.0)
        r_mat = jnp.sqrt(
            jnp.maximum(
                scaled_squared_euclid_dist(X, X2, self.lengthscale_matern), 1e-36
            )
        )
        k_mat32 = (1.0 + sqrt3 * r_mat) * jnp.exp(-sqrt3 * r_mat)
        return self.variance * k_mat32 * k_per

    def kernel_to_state_space(self, R=None):
        var_p = 1.0
        ell_p = self.lengthscale_periodic
        a = (
            self.b_fmK_2igrid
            * ell_p ** (-2.0 * self.igrid)
            * jnp.exp(-1.0 / ell_p**2.0)
            * var_p
        )
        q2 = jnp.sum(a, axis=0)
        # The angular frequency
        omega = 2 * jnp.pi / self.period
        # The model
        F_p = jnp.kron(
            jnp.diag(jnp.arange(self.order + 1)),
            jnp.array([[0.0, -omega], [omega, 0.0]]),
        )
        L_p = jnp.eye(2 * (self.order + 1))
        # Qc_p = jnp.zeros(2 * (self.N + 1))
        Pinf_p = jnp.kron(jnp.diag(q2), jnp.eye(2))
        H_p = jnp.kron(jnp.ones([1, self.order + 1]), jnp.array([1.0, 0.0]))
        lam = 3.0**0.5 / self.lengthscale_matern
        F_m = jnp.array([[0.0, 1.0], [-(lam**2), -2 * lam]])
        L_m = jnp.array([[0], [1]])
        Qc_m = jnp.array(
            [[12.0 * 3.0**0.5 / self.lengthscale_matern**3.0 * self.variance]]
        )
        H_m = jnp.array([[1.0, 0.0]])
        Pinf_m = jnp.array(
            [
                [self.variance, 0.0],
                [0.0, 3.0 * self.variance / self.lengthscale_matern**2.0],
            ]
        )
        # F = jnp.kron(F_p, jnp.eye(2)) + jnp.kron(jnp.eye(14), F_m)
        F = jnp.kron(F_m, jnp.eye(2 * (self.order + 1))) + jnp.kron(jnp.eye(2), F_p)
        L = jnp.kron(L_m, L_p)
        Qc = jnp.kron(Qc_m, Pinf_p)
        H = jnp.kron(H_m, H_p)
        # Pinf = jnp.kron(Pinf_m, Pinf_p)
        Pinf = block_diag(
            jnp.kron(Pinf_m, q2[0] * jnp.eye(2)),
            jnp.kron(Pinf_m, q2[1] * jnp.eye(2)),
            jnp.kron(Pinf_m, q2[2] * jnp.eye(2)),
            jnp.kron(Pinf_m, q2[3] * jnp.eye(2)),
            jnp.kron(Pinf_m, q2[4] * jnp.eye(2)),
            jnp.kron(Pinf_m, q2[5] * jnp.eye(2)),
            jnp.kron(Pinf_m, q2[6] * jnp.eye(2)),
        )
        return F, L, Qc, H, Pinf

    def stationary_covariance(self):
        var_p = 1.0
        ell_p = self.lengthscale_periodic
        a = (
            self.b_fmK_2igrid
            * ell_p ** (-2.0 * self.igrid)
            * jnp.exp(-1.0 / ell_p**2.0)
            * var_p
        )
        q2 = jnp.sum(a, axis=0)
        Pinf_m = jnp.array(
            [
                [self.variance, 0.0],
                [0.0, 3.0 * self.variance / self.lengthscale_matern**2.0],
            ]
        )
        Pinf = block_diag(
            jnp.kron(Pinf_m, q2[0] * jnp.eye(2)),
            jnp.kron(Pinf_m, q2[1] * jnp.eye(2)),
            jnp.kron(Pinf_m, q2[2] * jnp.eye(2)),
            jnp.kron(Pinf_m, q2[3] * jnp.eye(2)),
            jnp.kron(Pinf_m, q2[4] * jnp.eye(2)),
            jnp.kron(Pinf_m, q2[5] * jnp.eye(2)),
            jnp.kron(Pinf_m, q2[6] * jnp.eye(2)),
        )
        return Pinf

    def measurement_model(self):
        H_p = jnp.kron(jnp.ones([1, self.order + 1]), jnp.array([1.0, 0.0]))
        H_m = jnp.array([[1.0, 0.0]])
        H = jnp.kron(H_m, H_p)
        return H

    def state_transition(self, dt):
        """
        Calculation of the closed form discrete-time state
        transition matrix A = expm(FΔt) for the Quasi-Periodic Matern-3/2 prior
        :param dt: step size(s), Δt = tₙ - tₙ₋₁ [M+1, 1]
        :return: state transition matrix A [M+1, D, D]
        """
        lam = jnp.sqrt(3.0) / self.lengthscale_matern
        # The angular frequency
        omega = 2 * jnp.pi / self.period
        harmonics = jnp.arange(self.order + 1) * omega
        R0 = self.subband_mat32(dt, lam, harmonics[0])
        R1 = self.subband_mat32(dt, lam, harmonics[1])
        R2 = self.subband_mat32(dt, lam, harmonics[2])
        R3 = self.subband_mat32(dt, lam, harmonics[3])
        R4 = self.subband_mat32(dt, lam, harmonics[4])
        R5 = self.subband_mat32(dt, lam, harmonics[5])
        R6 = self.subband_mat32(dt, lam, harmonics[6])
        A = jnp.exp(-dt * lam) * block_diag(R0, R1, R2, R3, R4, R5, R6)
        return A

    @staticmethod
    def subband_mat32(dt, lam, omega):
        R = rotation_matrix(dt, omega)
        Ri = jnp.block(
            [[(1.0 + dt * lam) * R, dt * R], [-dt * lam**2 * R, (1.0 - dt * lam) * R]]
        )
        return Ri

    def feedback_matrix(self):
        # The angular frequency
        omega = 2 * jnp.pi / self.period
        # The model
        F_p = jnp.kron(
            jnp.diag(jnp.arange(self.order + 1)),
            jnp.array([[0.0, -omega], [omega, 0.0]]),
        )
        lam = 3.0**0.5 / self.lengthscale_matern
        F_m = jnp.array([[0.0, 1.0], [-(lam**2), -2 * lam]])
        F = jnp.kron(F_m, jnp.eye(2 * (self.order + 1))) + jnp.kron(jnp.eye(2), F_p)
        return F


class SubbandMatern32(bayesnewton.kernels.StationaryKernel):
    """
    Subband Matern-3/2 kernel in SDE form (product of Cosine and Matern-3/2).
    Hyperparameters:
        variance, σ²
        lengthscale, l
        radial frequency, ω
    The associated continuous-time state space model matrices are constructed via
    kronecker sums and products of the Matern3/2 and cosine components:
    letting λ = √3 / l
    F      = F_mat3/2 ⊕ F_cos  =  ( 0     -ω     1     0
                                    ω      0     0     1
                                   -λ²     0    -2λ   -ω
                                    0     -λ²    ω    -2λ )
    L      = L_mat3/2 ⊗ I      =  ( 0      0
                                    0      0
                                    1      0
                                    0      1 )
    Qc     = I ⊗ Qc_mat3/2     =  ( 4λ³σ²  0
                                    0      4λ³σ² )
    H      = H_mat3/2 ⊗ H_cos  =  ( 1      0     0      0 )
    Pinf   = Pinf_mat3/2 ⊗ I   =  ( σ²     0     0      0
                                    0      σ²    0      0
                                    0      0     3σ²/l² 0
                                    0      0     0      3σ²/l²)
    and the discrete-time transition matrix is (for step size Δt),
    R = ( cos(ωΔt)   -sin(ωΔt)
          sin(ωΔt)    cos(ωΔt) )
    A = exp(-Δt/l) ( (1+Δtλ)R   ΔtR
                     -Δtλ²R    (1-Δtλ)R )
    """

    def __init__(
        self, variance=1.0, lengthscale=1.0, radial_frequency=1.0, fix_variance=False
    ):
        self.transformed_radial_frequency = objax.TrainVar(
            jnp.array(softplus_inv(radial_frequency))
        )
        super().__init__(
            variance=variance, lengthscale=lengthscale, fix_variance=fix_variance
        )
        self.name = "Subband Matern-3/2"
        self.state_dim = 4

    @property
    def variance(self):
        return softplus(self.transformed_variance.value)

    @property
    def lengthscale(self):
        return softplus(self.transformed_lengthscale.value)

    @property
    def radial_frequency(self):
        return softplus(self.transformed_radial_frequency.value)

    def K_r(self, r):
        k_cos = jnp.cos(self.radial_frequency * r * self.lengthscale)

        sqrt3 = jnp.sqrt(3.0)
        k_mat = (1.0 + sqrt3 * r) * jnp.exp(-sqrt3 * r)

        return self.variance * k_mat * k_cos

    def kernel_to_state_space(self, R=None):
        lam = 3.0**0.5 / self.lengthscale
        F_mat = jnp.array([[0.0, 1.0], [-(lam**2), -2 * lam]])
        L_mat = jnp.array([[0], [1]])
        Qc_mat = jnp.array([[12.0 * 3.0**0.5 / self.lengthscale**3.0 * self.variance]])
        H_mat = jnp.array([[1.0, 0.0]])
        Pinf_mat = jnp.array(
            [[self.variance, 0.0], [0.0, 3.0 * self.variance / self.lengthscale**2.0]]
        )
        F_cos = jnp.array([[0.0, -self.radial_frequency], [self.radial_frequency, 0.0]])
        H_cos = jnp.array([[1.0, 0.0]])
        # F = (0   -ω   1   0
        #      ω    0   0   1
        #      -λ²  0  -2λ -ω
        #      0   -λ²  ω  -2λ)
        F = jnp.kron(F_mat, jnp.eye(2)) + jnp.kron(jnp.eye(2), F_cos)
        L = jnp.kron(L_mat, jnp.eye(2))
        Qc = jnp.kron(jnp.eye(2), Qc_mat)
        H = jnp.kron(H_mat, H_cos)
        Pinf = jnp.kron(Pinf_mat, jnp.eye(2))
        return F, L, Qc, H, Pinf

    def stationary_covariance(self):
        Pinf_mat = jnp.array(
            [[self.variance, 0.0], [0.0, 3.0 * self.variance / self.lengthscale**2.0]]
        )
        Pinf = jnp.kron(Pinf_mat, jnp.eye(2))
        return Pinf

    def measurement_model(self):
        H_mat = jnp.array([[1.0, 0.0]])
        H_cos = jnp.array([[1.0, 0.0]])
        H = jnp.kron(H_mat, H_cos)
        return H

    def state_transition(self, dt):
        """
        Calculation of the closed form discrete-time state
        transition matrix A = expm(FΔt) for the Subband Matern-3/2 prior
        :param dt: step size(s), Δt = tₙ - tₙ₋₁ [1]
        :return: state transition matrix A [4, 4]
        """
        lam = jnp.sqrt(3.0) / self.lengthscale
        R = rotation_matrix(dt, self.radial_frequency)
        A = jnp.exp(-dt * lam) * jnp.block(
            [[(1.0 + dt * lam) * R, dt * R], [-dt * lam**2 * R, (1.0 - dt * lam) * R]]
        )
        return A

    def feedback_matrix(self):
        lam = 3.0**0.5 / self.lengthscale
        F_mat = jnp.array([[0.0, 1.0], [-(lam**2), -2 * lam]])
        F_cos = jnp.array([[0.0, -self.radial_frequency], [self.radial_frequency, 0.0]])
        # F = (0   -ω   1   0
        #      ω    0   0   1
        #      -λ²  0  -2λ -ω
        #      0   -λ²  ω  -2λ)
        F = jnp.kron(F_mat, jnp.eye(2)) + jnp.kron(jnp.eye(2), F_cos)
        return F


class SpatioTemporalPartialOptKernel(bayesnewton.kernels.SpatioTemporalKernel):
    """
    The Spatio-Temporal GP class with partially optimized inducing points

    :param temporal_kernel: the temporal prior, must be a member of the Prior class
    :param spatial_kernel: the kernel used for the spatial dimensions
    :param z: the initial spatial locations
    :param conditional: specifies which method to use for computing the covariance of the spatial conditional;
                        must be one of ['DTC', 'FIC', 'Full']
    :param sparse: boolean specifying whether the model is sparse in space
    :param opt_z: boolean specifying whether to optimise the spatial input locations z
    """

    def __init__(
        self,
        temporal_kernel,
        spatial_kernel,
        z_train=None,
        z_fixed=None,
        conditional=None,
        sparse=True,
        spatial_dims=None,
    ):
        self.temporal_kernel = temporal_kernel
        self.spatial_kernel = spatial_kernel
        if conditional is None:
            if sparse:
                conditional = "Full"
            else:
                conditional = "DTC"
        if not sparse:
            # z should not be optimised if the model is not sparse
            warn("spatial inducing inputs z will not be optimised because sparse=False")
        self.sparse = sparse

        if z_train.ndim < 2:
            z_train = z_train[:, jnp.newaxis]
        if z_fixed.ndim < 2:
            z_fixed = z_fixed[:, jnp.newaxis]

        assert z_train.ndim == z_fixed.ndim

        if spatial_dims is None:
            spatial_dims = z_train.ndim - 1

        assert spatial_dims == z_train.ndim - 1

        self.M = z_train.shape[0] + z_fixed.shape[0]

        self.z_train = objax.TrainVar(jnp.array(z_train))
        self.z_fixed = objax.StateVar(jnp.array(z_fixed))

        if conditional in ["DTC", "dtc"]:
            self.conditional_covariance = self.deterministic_training_conditional
        elif conditional in ["FIC", "FITC", "fic", "fitc"]:
            self.conditional_covariance = self.fully_independent_conditional
        elif conditional in ["Full", "full"]:
            self.conditional_covariance = self.full_conditional
        else:
            raise NotImplementedError("conditional method not recognised")
        if (not sparse) and (conditional != "DTC"):
            warn(
                "You chose a non-deterministic conditional, but 'DTC' will be used because the model is not sparse"
            )

    @property
    def z(self):
        return jnp.concatenate((self.z_train.value, self.z_fixed.value), axis=0)

    def spatial_conditional(self, X=None, R=None, predict=False):
        """
        Compute the spatial conditional, i.e. the measurement model projecting the latent function u(t) to f(X,R)
            f(X,R) | u(t) ~ N(f(X,R) | B u(t), C)
        """
        (
            Qzz,
            Lzz,
        ) = self.inducing_precision()  # pre-calculate inducing precision and its Cholesky factor
        if self.sparse or predict:
            # TODO: save compute if R is constant:
            # gridded_data = np.all(np.abs(np.diff(R, axis=0)) < 1e-10)
            # if gridded_data:
            #     R = R[:1]
            R = R.reshape((R.shape[0],) + (-1,) + self.z.shape[1:])
            Krz = vmap(self.spatial_kernel, [0, None])(R, self.z)
            K = Krz @ Qzz  # Krz / Kzz
            B = K @ Lzz
            C = vmap(self.conditional_covariance)(
                X, R, Krz, K
            )  # conditional covariance
        else:
            B = Lzz
            # conditional covariance (deterministic mapping is exact in non-sparse case)
            C = jnp.zeros([B.shape[0], B.shape[0]])
        return B, C

    def inducing_precision(self):
        """
        Compute the covariance and precision of the inducing spatial points to be used during filtering
        """
        Kzz = self.spatial_kernel(self.z, self.z)
        Lzz, low = cho_factor(Kzz, lower=True)  # K_zz^(1/2)
        Qzz = cho_solve((Lzz, low), jnp.eye(self.M))  # K_zz^(-1)
        return Qzz, Lzz

    def kernel_to_state_space(self, R=None):
        F_t, L_t, Qc_t, H_t, Pinf_t = self.temporal_kernel.kernel_to_state_space()
        Kzz = self.spatial_kernel(self.z, self.z)
        F = jnp.kron(jnp.eye(self.M), F_t)
        Qc = None
        L = None
        H = self.measurement_model()
        Pinf = jnp.kron(Kzz, Pinf_t)
        return F, L, Qc, H, Pinf


class SubbandMatern12(bayesnewton.kernels.StationaryKernel):
    """
    Subband Matern-1/2 (i.e. Exponential) kernel in SDE form (product of Cosine and Matern-1/2).
    Hyperparameters:
        variance, σ²
        lengthscale, l
        radial frequency, ω
    The associated continuous-time state space model matrices are constructed via
    kronecker sums and products of the exponential and cosine components:
    F      = F_exp ⊕ F_cos  =  ( -1/l  -ω
                                 ω     -1/l )
    L      = L_exp ⊗ I      =  ( 1      0
                                 0      1 )
    Qc     = I ⊗ Qc_exp     =  ( 2σ²/l  0
                                 0      2σ²/l )
    H      = H_exp ⊗ H_cos  =  ( 1      0 )
    Pinf   = Pinf_exp ⊗ I   =  ( σ²     0
                                 0      σ² )
    and the discrete-time transition matrix is (for step size Δt),
    A      = exp(-Δt/l) ( cos(ωΔt)   -sin(ωΔt)
                          sin(ωΔt)    cos(ωΔt) )
    """

    def __init__(
        self,
        variance=1.0,
        lengthscale=1.0,
        radial_frequency=1.0,
        fix_variance=False,
        fix_lengthscale=False,
        fix_radial_frequency=False,
    ):
        self.transformed_radial_frequency = objax.TrainVar(
            jnp.array(softplus_inv(radial_frequency))
        )
        super().__init__(
            variance=variance, lengthscale=lengthscale, fix_variance=fix_variance
        )
        self.name = "Subband Matern-1/2"
        self.state_dim = 2

    @property
    def variance(self):
        return softplus(self.transformed_variance.value)

    @property
    def lengthscale(self):
        return softplus(self.transformed_lengthscale.value)

    @property
    def radial_frequency(self):
        return softplus(self.transformed_radial_frequency.value)

    def K_r(self, r):
        k_cos = jnp.cos(self.radial_frequency * r * self.lengthscale)

        k_mat = jnp.exp(-r)

        return self.variance * k_mat * k_cos

    def kernel_to_state_space(self, R=None):
        F_mat = jnp.array([[-1.0 / self.lengthscale]])
        L_mat = jnp.array([[1.0]])
        Qc_mat = jnp.array([[2.0 * self.variance / self.lengthscale]])
        H_mat = jnp.array([[1.0]])
        Pinf_mat = jnp.array([[self.variance]])
        F_cos = jnp.array([[0.0, -self.radial_frequency], [self.radial_frequency, 0.0]])
        H_cos = jnp.array([[1.0, 0.0]])
        # F = (-1/l -ω
        #      ω    -1/l)
        F = jnp.kron(F_mat, jnp.eye(2)) + F_cos
        L = jnp.kron(L_mat, jnp.eye(2))
        Qc = jnp.kron(jnp.eye(2), Qc_mat)
        H = jnp.kron(H_mat, H_cos)
        Pinf = jnp.kron(Pinf_mat, jnp.eye(2))
        return F, L, Qc, H, Pinf

    def stationary_covariance(self):
        Pinf_mat = jnp.array([[self.variance]])
        Pinf = jnp.kron(Pinf_mat, jnp.eye(2))
        return Pinf

    def measurement_model(self):
        H_mat = jnp.array([[1.0]])
        H_cos = jnp.array([[1.0, 0.0]])
        H = jnp.kron(H_mat, H_cos)
        return H

    def state_transition(self, dt):
        """
        Calculation of the closed form discrete-time state
        transition matrix A = expm(FΔt) for the Subband Matern-1/2 prior:
        A = exp(-Δt/l) ( cos(ωΔt)   -sin(ωΔt)
                         sin(ωΔt)    cos(ωΔt) )
        :param dt: step size(s), Δt = tₙ - tₙ₋₁ [1]
        :return: state transition matrix A [2, 2]
        """
        R = rotation_matrix(dt, self.radial_frequency)
        A = jnp.exp(-dt / self.lengthscale) * R  # [2, 2]
        return A

    def feedback_matrix(self):
        F_mat = jnp.array([[-1.0 / self.lengthscale]])
        F_cos = jnp.array([[0.0, -self.radial_frequency], [self.radial_frequency, 0.0]])
        # F = (-1/l -ω
        #      ω    -1/l)
        F = jnp.kron(F_mat, jnp.eye(2)) + F_cos
        return F
