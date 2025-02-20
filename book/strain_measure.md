
# Stress and strain

## Strain measures

### Deformation gradient tensor
The deformation gradient tensor, $\mathbf{F}$, is a crucial concept in finite deformation (large deformation) continuum mechanics, as it forms the basis for deriving all measures of deformation. It provides a linear mapping between an initial configuration, $d\mathbf{X}$, to an deformed configuration, $d\mathbf{x}$, without assuming that displacements or rotations are small. It is expressed as:

$$
\mathbf{F} := \frac{\partial \boldsymbol{\phi}}{\partial \mathbf{X}} = \frac{\partial \mathbf{x}}{\partial \mathbf{X}}, \quad \text{or in component form,} \quad F_{ij} = \frac{\partial x_i}{\partial X_j}.
$$

### Velocity gradient tensor

The velocity gradient tensor, $\mathbf{L}$, quantifies the spatial rate of change of the velocity field, $\mathbf{v}$, within the material. It captures the local rate of stretching, shearing, and rotation. $\mathbf{L}$ is defined as the gradient of the velocity vector, and its components are given by:

$$
\dot{\mathbf{F}} = \frac{\partial}{\partial t}\left(\frac{\partial \phi(\mathbf{X},t)}{\partial \mathbf{X}}\right) = \frac{\partial \mathbf{v}}{\partial \mathbf{X}} = \frac{\partial \mathbf{v}}{\partial \mathbf{x}} \cdot \frac{\partial \mathbf{x}}{\partial \mathbf{X}} = \mathbf{L} \cdot \mathbf{F}, \quad \Rightarrow \mathbf{L} = \dot{\mathbf{F}} \cdot \mathbf{F}^{-1}
$$

$$
\bf{L}=\nabla v =\left[
\begin{matrix}
\frac{\partial v_x}{\partial x} & \frac{\partial v_x}{\partial y} & \frac{\partial v_x}{\partial z} \\\\
\frac{\partial v_y}{\partial x} & \frac{\partial v_y}{\partial y} & \frac{\partial v_y}{\partial z} \\\\
\frac{\partial v_z}{\partial x} & \frac{\partial v_z}{\partial y} & \frac{\partial v_z}{\partial z}
\end{matrix}
\right]
$$

### Numerical integration of deformation gradient

To numerically compute $ \mathbf{F} $ over a discrete time step $ \Delta t $, we use the forward Euler approximation for $ \dot{\mathbf{F}} $:

$$
\mathbf{F}_p^{t+\Delta t} = \mathbf{F}_p^t + \dot{\mathbf{F}} \cdot \Delta t
$$

Substituting $ \dot{\mathbf{F}} = \mathbf{L}_p \cdot \mathbf{F}_p^t $:

$$
\mathbf{F}_p^{t+\Delta t} = \mathbf{F}_p^t + \left( \mathbf{L}_p \cdot \mathbf{F}_p^t \right) \cdot \Delta t
$$

Factoring $ \mathbf{F}_p^t $:

$$
\mathbf{F}_p^{t+\Delta t} = \mathbf{F}_p^t \cdot \left( \mathbf{I} + \mathbf{L}_p \cdot \Delta t \right)
$$



### Rate of deformation tensor (strain rate tensor)
The rate of deformation tensor, $\mathbf{D}$, also known as the strain rate tensor, represents the symmetric part of the velocity gradient tensor, $\mathbf{L}$. It describes the rate of change of strain in the material (rate of stretching and shearing). Mathematically, $\mathbf{D}$ is given by:

$$
\mathbf{D} = \frac{1}{2}(\mathbf{L} + \mathbf{L}^T) = 
\left[
\begin{matrix}
\frac{\partial v_x}{\partial x}  & \frac{1}{2}(\frac{\partial v_x}{\partial y} + \frac{\partial v_y}{\partial x}) & \frac{1}{2}(\frac{\partial v_x}{\partial z} + \frac{\partial v_z}{\partial x}) \\
\frac{1}{2}(\frac{\partial v_x}{\partial y} + \frac{\partial v_y}{\partial x}) & \frac{\partial v_y}{\partial y}  & \frac{1}{2}(\frac{\partial v_y}{\partial z} + \frac{\partial v_z}{\partial y}) \\
\frac{1}{2}(\frac{\partial v_x}{\partial z} + \frac{\partial v_z}{\partial x}) & \frac{1}{2}(\frac{\partial v_y}{\partial z} + \frac{\partial v_z}{\partial y}) & \frac{\partial v_z}{\partial z}
\end{matrix}
\right]
$$

As a 6x1 the rate of deformation tensor $\mathbf{D} = \left[\frac{dv_x}{dx}, \frac{dv_y}{dy}, \frac{dv_z}{dz}, \frac{1}{2}(\frac{dv_x}{dy} + \frac{dv_y}{dx}), \frac{1}{2}(\frac{dv_y}{dz} + \frac{dv_z}{dy}), \frac{1}{2}(\frac{dv_x}{dz} + \frac{dv_z}{dx})\right]^T$



## Stress and Strain

### Strain increment
Strain increments $ \Delta \boldsymbol{\varepsilon}_p $ are obtained from gradients of the nodal velocities evaluated at the material point $\boldsymbol{p}$.

$$
\Delta \boldsymbol{\varepsilon}_p = \Delta t \cdot \frac{1}{2} \left( \mathbf{L}_p + \mathbf{L}_p^T \right) = \Delta t \cdot \boldsymbol{D}
$$
where $ \mathbf{L}_p $ is the velocity gradient.

### Stress increment
Stress increments are obtained from the strain increments. It is computed using the material's constitutive model:

$$
\Delta \boldsymbol{\sigma}_p = f (\boldsymbol{\sigma}_p, \Delta \boldsymbol{\varepsilon}_p, \boldsymbol{\theta})
$$
