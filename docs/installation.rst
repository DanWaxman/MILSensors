Installation
============

MILSensors is currently distributed from source (not PyPI). The recommended workflow mirrors the root README and uses `uv` to manage a virtual environment and install dependencies.

Prerequisites
-------------

* Python 3.8+
* Optional: CUDA-capable GPU for accelerated JAX

Dependencies and Version Pins
-----------------------------

The project relies on BayesNewton, which in turn supports a specific JAX stack. As pinned in ``uv.lock``:

* jax == 0.4.6
* jaxlib == 0.4.6
* numpy < 2.0
* scipy == 1.11.3
* objax, pandas, tqdm, uncertainty-toolbox

These version constraints exist due to BayesNewton compatibility. Installing other JAX/JAXLIB versions may lead to runtime import errors.

Recommended (uv) setup
----------------------

.. code-block:: bash

   # Create and activate a virtual environment
   uv venv
   source .venv/bin/activate

   # Install project dependencies from the root requirements file
   uv pip install -r requirements.in

Pip-only setup
--------------

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.in

CUDA-enabled JAX (optional)
---------------------------

If you have a CUDA-capable GPU and want accelerated computation, install CUDA-enabled JAX/JAXLIB versions that match the pins above (jax==0.4.6, jaxlib==0.4.6). Refer to the official JAX installation guide for the exact wheel URLs and CUDA/cuDNN requirements for that version:

.. code-block:: text

   https://github.com/google/jax#installation

Note: Ensure the CUDA toolkit and cuDNN versions you install match those required by the JAX/JAXLIB wheels for 0.4.6.

Verification
------------

After installing dependencies, you can import the package modules (from the source tree) to verify:

.. code-block:: python

   import milsensors.data_helper as dh
   import milsensors.models as models
   print("MILSensors modules import OK")

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

1. **JAX/JAXLIB version mismatch**: Use the pinned versions (jax==0.4.6, jaxlib==0.4.6) to ensure BayesNewton compatibility.

2. **Missing CUDA runtime**: For GPU use, install the CUDA/cuDNN versions required by the JAX 0.4.6 wheels (see JAX install guide).

3. **Build toolchain errors**: If NumPy/SciPy wheels fail to install, upgrade ``pip``/``setuptools``/``wheel`` and ensure a compatible compiler toolchain is present.

Getting Help
------------

If you encounter issues during installation, please:

1. Check the troubleshooting section above
2. Search existing issues on the project repository
3. Create a new issue with detailed error information
