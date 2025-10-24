MILSensors Documentation
========================

MILSensors is a Python toolkit for spatiotemporal Gaussian Process (GP) modeling and optimal experimental design. It is domain-agnostic and can be applied to any gridded or irregular spatiotemporal data (e.g., environmental monitoring, mobility, energy, climate, and beyond).

What it provides
----------------

* **Spatiotemporal GP models** with Matern, quasi-periodic, and subband kernel constructions
* **Efficient variational inference** via Markov state-space GP filtering and smoothing
* **Optimal experimental design (MIL)** utilities for principled sensor placement and sensor network design
* **Data utilities** to prepare inputs for spatiotemporal modeling
* **Modular components** that can be mixed-and-matched or extended

Quick Start
-----------

.. code-block:: python

   import numpy as np
   from milsensors.models import make_model_matern
   from milsensors.train_and_eval import training

   # Synthetic example data (replace with your own)
   # X: (N_obs, 3) columns = [time, lat, lon]
   # Y: (N_obs, 1)
   # R: (N_times, N_sites, 2) spatial grid; t: (N_times, 1)
   N_obs = 200
   X = np.random.rand(N_obs, 3)
   Y = np.random.randn(N_obs, 1)
   t = X[:, :1]
   R = X[:, 1:]

   # Build a simple Matern spatiotemporal model and train
   model = make_model_matern(N_obs_pts=N_obs, X=X, Y=Y, R=R, t=t, var_y=0.1)
   model = training(model, iters=50, verbose=True)

Installation
------------

See the :doc:`installation` guide for recommended environments, pinned JAX versions, and optional CUDA setup.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Documentation:

   installation
   api_reference
   examples

API Reference
-------------

.. toctree::
   :maxdepth: 2
   :caption: API Documentation:

   modules/data_helper
   modules/sensor_design
   modules/bayes_oed
   modules/kernels
   modules/models
   modules/train_and_eval

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
