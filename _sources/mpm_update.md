# MPM Updates

## Update stress last (USL) scheme

Below is the USL (Update Stress Last) Material Point Method (MPM) scheme with explanations for each step. 


### 1. Initialization

1. **Set up the background mesh (i.e., grid), set time $ t = 0 $.**

     A fixed grid is established over the computational domain. Although the grid itself does not deform, it serves as a temporary computational structure to evaluate forces and accelerations. The simulation time is initialized.

2. **Set up particle data: $ x^0_p, v^0_p, \sigma^0_p, F^0_p, V^0_p, m_p, \rho_p^0 $.**

     Material points (or particles) carry the physical information of the continuum body. Each particle is assigned:
     - **Position $ x^0_p $:** Where the particle is located.
     - **Velocity $ v^0_p $:** Its initial speed and direction.
     - **Stress $ \sigma^0_p $:** The initial state of internal forces.
     - **Deformation gradient $ F^0_p $:** Describes how the particle's local configuration deforms.
     - **Volume $ V^0_p $:** Its initial volume.
     - **Mass $ m_p $:** Its mass, which is constant.
     - **Density $ \rho_p^0 $:** The initial mass per unit volume.



### 2. Time-Stepping Loop

The simulation advances in time steps $\Delta t$ until a final time $ t_f $ is reached.

#### Step 2.1: Reset Grid Quantities

- **What is done:**  
  At the start of each time step, for every grid node $ I $, the following quantities are reset to zero:
  - Mass: $ m^t_I = 0 $
  - Momentum: $ (mv)^t_I = 0 $
  - External force: $ f^{ext,t}_I = 0 $
  - Internal force: $ f^{int,t}_I = 0 $
  - Total force: $ f^t_I = 0 $

- **Explanation:**  
  Each time step involves re-computing the nodal quantities from scratch based on updated particle data.



#### Step 2.2: Mapping from Particles to Nodes (P2G)

The Particle-to-Grid (P2G) step transfers particle information to the grid nodes using shape functions, denoted by $ \phi_I(x^t_p) $. The shape functions determine how much influence a particle has on a given node based on its location.

1. **Compute nodal mass**

   $$
   m^t_I = \sum_p \phi_I(x^t_p)m_p.
   $$

   - **Explanation:**  
     Each particle's mass $ m_p $ is distributed to surrounding grid nodes, weighted by the shape function. This creates a nodal mass field that is used in subsequent momentum and acceleration calculations.

2. **Compute nodal momentum**

   $$
   (mv)^t_I = \sum_p \phi_I(x^t_p)(mv)^t_p.
   $$

   - **Explanation:**  
     Similarly, the momentum (mass times velocity) of each particle is mapped to the grid nodes. The resulting nodal momentum provides the basis for updating nodal velocities.

3. **Compute external force**

   $$
   f^{ext,t}_I = \sum_p \phi_I(x^t_p)m_p b(x_p).
   $$

   - **Explanation:**  
     External forces (for example, gravity, represented by $ b(x_p) $) are applied to the particles. These forces are then transferred to the grid nodes, ensuring that body forces influence the motion of the system.

4. **Compute internal force**

   $$
   f^{int,t}_I = - \sum_p V^0_p \sigma^t_p \nabla \phi_I(x^t_p).
   $$

   - **Explanation:**  
     Internal forces arise from the stress within the material. The particle stress $ \sigma^t_p $ is weighted by the gradient of the shape function, and the negative sign follows from the conventional formulation in continuum mechanics (where tension/compression results in opposite force directions). The initial volume $ V^0_p $ is used to scale the stress contribution appropriately.

5. **Compute nodal force**

   $$
   f^t_I = f^{ext,t}_I + f^{int,t}_I.
   $$

   - **Explanation:**  
     The total force at each grid node is simply the sum of the external and internal forces. This net force is what will drive the change in momentum on the grid.



#### Step 2.3: Update the Momenta

- **Operation:**

  $$
  (mv)^{t+\Delta t}_I = (mv)^t_I + f^t_I \Delta t.
  $$

- **Explanation:**  
  This is a direct application of Newton's second law (force equals the rate of change of momentum). By multiplying the net force by the time step $\Delta t$, the nodal momentum is updated explicitly. This step is crucial for advancing the system in time.



#### Step 2.4: Fix Dirichlet Nodes

- **Operation:**

  For nodes with prescribed (Dirichlet) boundary conditions, set

  $$
  (mv)^t_I = 0 \quad \text{and} \quad (mv)^{t+\Delta t}_I = 0.
  $$

- **Explanation:**  
  Certain boundaries in the physical problem might be fixed (e.g., a wall or a clamped edge). To enforce these conditions, the momentum at the corresponding grid nodes is manually set to zero, ensuring that the nodes do not move contrary to the specified constraints.



#### Step 2.5: Update Particles (G2P)

After computing grid nodal quantities, the updated information is transferred back to the particles in the Grid-to-Particle (G2P) step.

1. **Get nodal velocities**

   $$
   v^t_I = \frac{(mv)^t_I}{m^t_I} \quad \text{and} \quad v^{t+\Delta t}_I = \frac{(mv)^{t+\Delta t}_I}{m^t_I}.
   $$

   - **Explanation:**  
     The velocity at each grid node is computed by dividing the nodal momentum by the nodal mass. These nodal velocities are then used to update particle properties.

2. **Update particle velocities**

   $$
   v^{t+\Delta t}_p = \alpha \left(v^t_p + \sum_I \phi_I(x^t_p)\left[v^{t+\Delta t}_I - v^t_I \right] \right) + (1 - \alpha) \sum_I \phi_I(x^t_p)v^{t+\Delta t}_I.
   $$

   - **Explanation:**  
     This step updates the particle velocity by blending:
     - The change in velocity from the grid nodes ($v^{t+\Delta t}_I - v^t_I$), which is added to the previous particle velocity $v^t_p$, and
     - The newly computed grid velocity $v^{t+\Delta t}_I$.
     
     The weighting factor $\alpha$ (between 0 and 1) is 0, the particle velocity update follows the *Particle in Cell (PIC)* way where the particle velocity is obtained using the total grid velocity. When $\alpha$ is 1, the particle velocity update follows the *Fluid Implicit Particle (FLIP)* way where the particle velocity is obtained using the grid velocity increment at the particle position. PIC method is more numerically stable, but less energy conservative than FLIP. When $\alpha$ is between 0 and 1, the particle velocity update is a blend of PIC and FLIP.

3. **Update particle positions**

   $$
   x^{t+\Delta t}_p = x^t_p + \Delta t \sum_I \phi_I(x^t_p)v^{t+\Delta t}_I.
   $$

   - **Explanation:**  
     Each particle's new position is determined by moving it in the direction of the interpolated grid velocity. The displacement is the product of the time step $\Delta t$ and the weighted nodal velocities.

4. **Compute velocity gradient**

   $$
   L^{t+\Delta t}_p = \sum_I \nabla \phi_I(x^t_p)v^{t+\Delta t}_I.
   $$

   - **Explanation:**  
     The velocity gradient $L$ quantifies how the velocity field changes in space around each particle. This information is necessary for updating the deformation of the material.

5. **Update gradient deformation tensor**

   $$
   F^{t+\Delta t}_p = (I + L^{t+\Delta t}_p \Delta t) F^t_p.
   $$

   - **Explanation:**  
     The deformation gradient $F$ captures the cumulative deformation history of a particle. The term $I + L^{t+\Delta t}_p \Delta t$ represents the incremental deformation during the time step. Multiplying by the previous deformation gradient $F^t_p$ updates the particle's state.

6. **Update volume**

   $$
   V^{t+\Delta t}_p = \det(F^{t+\Delta t}_p) V^0_p.
   $$

   - **Explanation:**  
     As the material deforms, the volume of each particle may change. The determinant of the deformation gradient $ \det(F) $ gives the local volume change ratio. Multiplying this by the original volume $ V^0_p $ provides the updated volume.

7. **Compute the rate of deformation matrix**

   $$
   D^{t+\Delta t}_p = 0.5\left(L^{t+\Delta t}_p + \left(L^{t+\Delta t}_p\right)^T\right).
   $$

   - **Explanation:**  
     The rate of deformation matrix $D$ is the symmetric part of the velocity gradient. It represents the pure deformation (stretching and shearing) without any rigid-body rotation. This is key for constitutive models that relate strain rate to stress rate.

8. **Compute the strain increment**

   $$
   \Delta \varepsilon_p = \Delta t\, D^{t+\Delta t}_p.
   $$

   - **Explanation:**  
     The strain increment is the change in strain over the time step, calculated by multiplying the rate of deformation $D$ by the time step $\Delta t$. This increment is then used to update the stress state.

9. **Update stresses**

   $$
   \sigma^{t+\Delta t}_p = \sigma^t_p + \Delta \sigma\, \Delta \varepsilon_p \quad \text{or} \quad \sigma^{t+\Delta t}_p = \sigma^t\left(F^{t+\Delta t}_p\right).
   $$

   - **Explanation:**  
     Finally, the stress in each particle is updated. There are typically two approaches:
     - **Incremental update:** The stress is updated using the strain increment multiplied by a stiffness-like term $ \Delta \sigma $ (which depends on the material model).
     - **Direct evaluation:** The stress is computed as a function of the updated deformation gradient $ F^{t+\Delta t}_p $.
     
     The choice depends on the constitutive model employed for the material.



#### Step 2.6: Advance Time

- **Operation:**

  $$
  t = t + \Delta t.
  $$

- **Explanation:**  
  After completing the updates for the current time step, the simulation time is advanced by the time increment $\Delta t$, preparing the algorithm for the next cycle.

### End of computation cycle


### Summary

- **Mappings:**  
  The algorithm uses two mappings:
  - **P2G (Particle-to-Grid):** Transfers mass, momentum, and forces from the particles to the grid, enabling the evaluation of spatial derivatives and forces.
  - **G2P (Grid-to-Particle):** Transfers updated nodal velocities and computed quantities back to the particles, allowing them to update their state variables (velocity, position, deformation, and stress).

- **Explicit Time Integration:**  
  The scheme is explicit, meaning that each update is computed directly from known quantities at the current time step. This requires careful selection of the time step $\Delta t$ to maintain numerical stability.

- **Update Stress Last:**  
  In the USL formulation, stress is updated after the particle's kinematic quantities (like velocity and deformation) have been updated. 