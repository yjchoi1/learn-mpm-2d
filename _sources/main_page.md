# Learn MPM 2D with Python
> Yongjin Choi

This repository provides a general background of material point method (MPM)and hands-on practice for simple 2D material point method (MPM) code using Python jupyter notebook. At the end of the course, you will be able to implement following two simulations with MPM:

**Elastic ball collision:**
![practice-1](figs/sim_config.png)

**Granular column collapse:**
![practice-2](figs/granular_column_collapse.png)

## Prerequisite
* This practice uses numpy library in python. 
* To review basic information of MPM, please refer to [this site](https://www.geoelements.org/LearnMPM/mpm.html). It introduces governing equations, discretisation, and time integration scheme in MPM, with hands-on python practice for 1D MPM.

## Install
```shell
# Initiate a python virtual environment.
python -m virtualenv venv
# Activate the virtual environment.
source venv/bin/activate
# Install dependencies.
python -m pip install --upgrade pip
pip install -r ./book/requirements.txt
```

## Inspiration
* https://github.com/geoelements/LearnMPM
* Nguyen, V. P., de Vaucorbeil, A., & Bordas, S. (2023). The material point method. Cham: Springer International Publishing.
* https://github.com/vinhphunguyen/mpmat?tab=readme-ov-file

## Acknowledgement