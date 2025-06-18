import jax.numpy as jnp
from models import MarkovVariationalGP

def get_posterior_predictive_tr(model : MarkovVariationalGP, t : jnp.ndarray, R : jnp.ndarray) -> jnp.ndarray:
    """
    Get the posterior predictive trace of the model.

    Args:
        t: The time points to predict the posterior predictive trace at.
        R: The number of samples to predict the posterior predictive trace at.

    Returns:
        The posterior predictive trace at the given time points.
    """
    _, var_y = model.predict(t, R) # (N_t, N_s)

    return jnp.sum(var_y, axis=1)

def get_posterior_predictive_logdet(model : MarkovVariationalGP, t : jnp.ndarray, R : jnp.ndarray) -> jnp.ndarray:
    """
    Get the posterior predictive log determinant of the model for each time in `t`.

    Args:
        t: The time points to predict the posterior predictive log determinant at.
        R: The number of samples to predict the posterior predictive log determinant at.

    Returns:
        The posterior predictive log determinant at the given time points.
    """
    _, cov_y = model.predict(t, R, return_diag_cov_only=False) # (N_t, N_s, N_s)

    return jnp.linalg.slogdet(cov_y)[1]
