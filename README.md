[![arXiv](https://img.shields.io/badge/arXiv-2512.05940-b31b1b.svg)](https://arxiv.org/abs/2512.05940)

# Designing an Optimal Sensor Network via Minimizing Information Loss

This repository contains the code for the paper "Designing an Optimal Sensor Network via Minimizing Information Loss", available in [Bayesian Analysis](https://projecteuclid.org/journals/bayesian-analysis/advance-publication/Designing-an-Optimal-Sensor-Network-via-Minimizing-Information-Loss/10.1214/25-BA1574.full), which proposes a model-based sensor placement criterion based on variational spatiotemporal Gaussian processes. The paper is also available [on the arXiv](https://arxiv.org/abs/2512.05940). 

If you find the paper or code helpful in your research, please consider citing the paper:
```
@article{waxman2025milsensors,
author = {Daniel Waxman and Fernando Llorente and Katia Lamer and Petar M. Djurić},
title = {{Designing an Optimal Sensor Network via Minimizing Information Loss}},
journal = {Bayesian Analysis},
publisher = {International Society for Bayesian Analysis},
pages = {1 -- 29},
year = {2025},
doi = {10.1214/25-BA1574},
URL = {https://doi.org/10.1214/25-BA1574}
}
```

## Paper Abstract
Optimal experimental design is a classic topic in statistics, with many well-studied problems, applications, and solutions. The design problem we study is the placement of sensors to monitor spatiotemporal processes, explicitly accounting for the temporal dimension in our modeling and optimization. We observe that recent advancements in computational sciences often yield large datasets based on physics-based simulations, which are rarely leveraged in experimental design. We introduce a novel model-based sensor placement criterion, along with a highly-efficient optimization algorithm, which integrates physics-based simulations and Bayesian experimental design principles to identify sensor networks that "minimize information loss" from simulated data. Our technique relies on sparse variational inference and (separable) Gauss-Markov priors, and thus may adapt many techniques from Bayesian experimental design. We validate our method through a case study monitoring air temperature in Phoenix, Arizona, using state-of-the-art physics-based simulations. Our results show our framework to be superior to random or quasi-random sampling, particularly with a limited number of sensors. We conclude by discussing practical considerations and implications of our framework, including more complex modeling tools and real-world deployments. 

## Project Overview
This project implements the methodology described in the paper "Designing an Optimal Sensor Network via Minimizing Information Loss". The codebase is structured to facilitate the design and evaluation of sensor networks for monitoring spatiotemporal processes, with a focus on minimizing information loss. 

## Installation
To install dependencies, you can use `pip`:
```bash
pip install -r requirements.in
```
Alternatively, our recommended workflow is to use `uv` with a virtual environment:
```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.in
``` 

## Usage
With the correct dependencies installed, you can run our code for the case study in Phoenix, Arizona by running:
```bash
python src/get_N_optimal.py --N_runs <N_runs> --N_sites_to_try <N_sites_to_try> --device_num <device_num> --dataset "phoenix" --data_file "data/WRF_data_2013_phoenix.npz" --N_t <N_t>  
```
where `<N_runs>` is the number of runs of the optimization, `<N_sites_to_try>` is the number of sites to try, `<device_num>` is the device number, `<N_t>` is the number of time steps.

## Citation
If you use this code, please cite the paper as follows:
```bibtex
@article{waxman2025designing,
  title={Designing an Optimal Sensor Network via Minimizing Information Loss},
  author={Waxman, Daniel and Llorente, Fernando and Lamer, Katia and Djuric, Petar M.},
  year={2026},
  journal={Bayesian Analysis},
  note={To Appear.}
}
```
