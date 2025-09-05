Examples
========

This section provides hands-on examples demonstrating the MILSensors library for spatiotemporal Gaussian Process modeling and optimal sensor placement.

Jupyter Notebooks
------------------

.. toctree::
   :maxdepth: 1

   sensor_design_example

MIL-Based Sensor Design
-----------------------

The :doc:`sensor_design_example` notebook provides a complete walkthrough of:

- Generating synthetic spatiotemporal temperature data on an irregular domain
- Training spatiotemporal variational Gaussian Process models
- Extracting optimal sensor locations using the MIL criterion
- Visualizing results and uncertainty quantification

This example demonstrates the core workflow for environmental monitoring applications where sensor placement decisions have significant cost and logistical implications.

Additional Examples
-------------------

For more advanced examples and real-world applications, see the ``experiments/`` directory in the repository, which includes:

- ``run_sensor_design_synth.py``: Extended version of the notebook example with additional visualizations
- ``get_N_optimal.py``: Comprehensive study comparing different numbers of inducing points
- Scripts for working with WRF weather data from the Phoenix metropolitan area