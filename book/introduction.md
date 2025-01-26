# Overview of Material Point Method

In continuum mechanics, there are two main descriptions of material motion: Lagrangian and Eulerian. 
The Lagrangian (material) description uses the initial configuration $\boldsymbol{X}$ (i.e. undeformed configuration at time $t = 0$) to describe the physical quantities and the deformation state of a continuum body $\boldsymbol{x}$. So, the emphasis is given to individual particles in Lagrangian kinematic description. 
On the other hand, the Eulerian (or spatial) approach describes the motion of a continuum body with respect to the current coordinates and time. 
The spatial description therefore focuses on a specific position in space at current time.

The Material Point Method (MPM) is a hybrid Eulerian-Lagrangian approach, which uses moving material points on a fixed computational background grid. 
The MPM is a particle based method that represents the material as a collection of material points, and their deformations are determined by Newton’s laws of motion. This approach is very effective particularly in the context of large deformations.

The MPM algorithm involves the following steps, and also illustrated in the figure below.

1. **Particle to node**: A representation of material points overlaid on a computational grid. Arrows represent material point state vectors (mass, volume, velocity, etc.) being projected to the nodes of the computational grid. 
2. **Nodal solution**: The equations of motion are solved onto the nodes, resulting in updated nodal velocities and positions. 
3. **Note to particle**: The updated nodal kinematics are interpolated back to the material points. 
4. **Update particle**: The state of the material points is updated, and the
computational grid is reset.

![algorithm](figs/mpm-algorithm.png)
*Illustration of the MPM algorithm*

## References
* https://www.geoelements.org/LearnMPM/mpm.html
