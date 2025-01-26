# Description of deformation

In continuum mechanics, a body $\mathcal{B}$ is modeled as consisting of an infinite collection of material points, each possessing specific mechanical properties. The initial position of a material point in the undeformed configuration is represented by the vector $\mathbf{X}$, defined relative to a chosen coordinate basis. This vector, $\mathbf{X}$, is referred to as the material or Lagrangian coordinate. In contrast, the position of the same point in the deformed configuration is denoted by $\mathbf{x}$, known as the spatial or Eulerian coordinate.

![algorithm](figs/deformation.png)
*Motion of a deformable body in continuum mechanics*

The motion or deformation of a solid is characterized by a mapping function $\boldsymbol{\varphi}(\mathbf{X}, t)$, which relates the spatial coordinates to the material coordinates as follows:

$$
\mathbf{x} = \boldsymbol{\varphi}(\mathbf{X}, t)
$$

Key kinematic fields, such as displacement, velocity, and acceleration, describe the motion of a body. The displacement of a material point $\mathbf{X}$, denoted by $\mathbf{u}(\mathbf{X}, t)$, is the difference between the current position $\boldsymbol{\varphi}(\mathbf{X}, t)$ and the initial position $\boldsymbol{\varphi}(\mathbf{X}, 0)$, expressed as:

$$
\mathbf{u}(\mathbf{X}, t) := \boldsymbol{\varphi}(\mathbf{X}, t) - \boldsymbol{\varphi}(\mathbf{X}, 0) = \mathbf{x} - \mathbf{X}
$$

The velocity of a material point $\mathbf{X}$, denoted by $\mathbf{v}(\mathbf{X}, t)$, is defined as the time derivative of its position:

$$
\mathbf{v}(\mathbf{X}, t) := \frac{\partial \boldsymbol{\varphi}(\mathbf{X}, t)}{\partial t}
$$

This velocity field corresponds to the Lagrangian description. While a Eulerian form of the velocity field exists, it is not covered here as the Material Point Method (MPM) adopts a Lagrangian framework.

The acceleration of a material point $\mathbf{X}$, denoted by $\mathbf{a}(\mathbf{X}, t)$, is the time derivative of the velocity or, equivalently, the second time derivative of the position function:

$$
\mathbf{a}(\mathbf{X}, t) := \frac{\partial \mathbf{v}(\mathbf{X}, t)}{\partial t} = \frac{\partial^2 \boldsymbol{\varphi}(\mathbf{X}, t)}{\partial t^2}
$$


## Reference
* Nguyen, V. P., de Vaucorbeil, A., & Bordas, S. (2023). The Material Point Method: Theory, Implementations, and Applications. Springer. https://doi.org/10.1007/978-3-031-24070-6
* https://www.geoelements.org/LearnMPM/mpm.html#motion-of-deformable-body