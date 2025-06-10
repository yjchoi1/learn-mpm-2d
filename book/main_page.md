# Learn MPM 2D with Python
> Yongjin Choi

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Book](https://img.shields.io/badge/Book-Learn%202D%20MPM-blue.svg)](https://yjchoi1.github.io/learn-mpm-2d/)

This repository is part of the instructional materials for **CEE-8813-H: Numerical Modeling and Data Science in Geotechnics**, offered by the School of Civil and Environmental Engineering at Georgia Tech. 

It provides a general background of material point method (MPM) and hands-on practice for 2D material point method (MPM) code using Python jupyter notebook. At the end of the course, you will be able to implement following two simulations with MPM:

**Elastic ball collision:**
![practice-1](figs/sim_config.png) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yjchoi1/learn-mpm-2d/blob/main/book/mpm2d-elastic-ball.ipynb)

**Granular column collapse:**
![practice-2](figs/granular_column_collapse.png) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yjchoi1/learn-mpm-2d/blob/main/book/mpm2d-column-collapse.ipynb)

## Prerequisite
* This practice uses numpy library in python. 
* If you want to start with simpler 1D MPM examples, 
[this site](https://www.geoelements.org/LearnMPM/mpm.html) provides solid background and hands-on examples.

## Install
The examples are ready to run on Google Colab without any installation. 
However, if you prefer to run them locally on your machine, follow these steps:
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

##  Acknowledgments
Thanks for Dr. Macedo, Vivek Bokkisa and Sean Flournoy for providing valuable feedbacks and suggestions.