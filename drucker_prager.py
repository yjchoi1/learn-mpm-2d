import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Temporary hard-coded variable
nu = 0.3

def principal_stresses(stress_tensor):
    """
    Calculate the principal stresses sigma_1 and sigma_2 for a 2D stress tensor using eigen decomposition.

    Parameters:
    stress_tensor (list): A list of the form [sigma_xx, sigma_yy, sigma_xy].

    Returns:
    tuple: Principal stresses (sigma_1, sigma_2) with sigma_1 >= sigma_2.
    """
    # Unpack the stress components
    sigma_xx, sigma_yy, sigma_xy = stress_tensor

    # Form the 2D stress tensor matrix
    stress_matrix = np.array([[sigma_xx, sigma_xy],
                              [sigma_xy, sigma_yy]])

    # Perform eigen decomposition to get the eigenvalues (principal stresses)
    eigenvalues, _ = np.linalg.eig(stress_matrix)

    # Sort eigenvalues in descending order: sigma_1 >= sigma_2
    sigma_1, sigma_2 = np.sort(eigenvalues)[::-1]

    return sigma_1, sigma_2


# Function to compute alpha and k from friction angle and cohesion
def drucker_prager_parameters(friction_angle_deg, cohesion):
    # Convert friction angle to radians
    phi = np.radians(friction_angle_deg)
    
    # Compute alpha and k
    alpha = (2 * np.sin(phi)) / (np.sqrt(3) * (3 - np.sin(phi)))
    # alpha = 0
    k = (6 * cohesion * np.cos(phi)) / (np.sqrt(3) * (3 - np.sin(phi)))
    
    return alpha, k

# Elasticity matrix (2D plane strain assumption)
def elasticity_matrix(E, nu):
    D = (E / ((1 + nu) * (1 - 2 * nu))) * np.array([
        [1 - nu,     nu,          0],
        [nu,         1 - nu,      0],
        [0,          0,           (1 - 2 * nu) / 2]
    ])
    return D

# Invariants of the stress tensor
def compute_invariants(stress, stress_condition="plane_strain"):
    sigma_xx, sigma_yy, tau_xy = stress
    
    if stress_condition != "plane_strain":
        # First invariant
        I1 = sigma_xx + sigma_yy  
        
        # J2 invariant
        # Deviatoric stresses
        s_xx = sigma_xx - I1 / 2
        s_yy = sigma_yy - I1 / 2
        s_xy = tau_xy
        
        # J2 = 0.5 * s:s
        J2 = 0.5 * (s_xx**2 + s_yy**2 + s_xy**2)
        # The above is equivalent to:
        # J2 = (1/4) * (sigma_x - sigma_y)**2 + tau_xy**2
        
    else:  # Plain strain constraint
        sigma_zz = nu * (sigma_xx + sigma_yy)
        I1 = sigma_xx + sigma_yy + sigma_zz
        
        # Deviatoric stresses
        s_xx = sigma_xx - I1 / 3
        s_yy = sigma_yy - I1 / 3
        s_zz = sigma_zz - I1 / 3
        s_xy = tau_xy
        
        # J2 = 0.5 s:s
        J2 = 0.5 * (s_xx**2 + s_yy**2 + s_zz**2 + s_xy**2)
    
    return I1, J2

# Drucker-Prager yield function
def drucker_prager_yield(stress, alpha, k):
    I1, J2 = compute_invariants(stress)
    return alpha * I1 + np.sqrt(J2) - k

# Gradient of yield function df/dsigma
def df_dsigma(stress, alpha):
    sigma_x, sigma_y, tau_xy = stress
    I1, J2 = compute_invariants(stress)
    
    dfdx = alpha + (sigma_x - sigma_y) / np.sqrt(J2)
    dfdy = alpha - (sigma_x - sigma_y) / np.sqrt(J2)
    dftau = 2 * tau_xy / np.sqrt(J2)
    return np.array([dfdx, dfdy, dftau])

# Update stress and strain process
def update_stress_strain(stress_n, strain_n, strain_increment, D, alpha, k):
    # Step 1: Compute trial stress
    stress_trial = stress_n + np.dot(D, strain_increment)
    
    # Step 2: Check yield condition
    f_trial = drucker_prager_yield(stress_trial, alpha, k)
    # print(f_trial)
    
    if f_trial <= 0:
        # Elastic step: no plastic deformation
        return stress_trial, strain_n + strain_increment, 0.0  # No plastic multiplier
    
    # Step 3: Plastic step - compute the plastic multiplier Δλ
    df_sigma = df_dsigma(stress_trial, alpha)
    
    H = np.dot(df_sigma, np.dot(D, df_sigma))
    delta_lambda = f_trial / H
    
    # Step 4: Update stress using the return mapping scheme
    stress_updated = stress_trial - delta_lambda * np.dot(D, df_sigma)
    # stress_updated[1] = 100  # yc: keep sigma_y constant for triaxial test
    
    # Step 5: Update plastic strain
    plastic_strain_increment = delta_lambda * df_sigma
    strain_updated = strain_n + plastic_strain_increment
    
    return stress_updated, strain_updated, delta_lambda

# Example usage
if __name__ == "__main__":
    # Material properties
    E = 2e6   # Young's modulus in Pascals
    nu = 0.3  # Poisson's ratio
    friction_angle = 30  # Degrees
    cohesion = 2e3       # Cohesion in Pascals
    
    # Compute Drucker-Prager parameters alpha and k
    alpha, k = drucker_prager_parameters(friction_angle, cohesion)
    
    # Elasticity matrix
    D = elasticity_matrix(E, nu)
    
    # Arrays to store results
    axial_strain_history = []
    deviatoric_stress_history = []
    I1_history = []
    J2_history = []
    p_history = []
    q_history = []
    strain_history = []
    
    # Initial stress and strain
    stress_n = np.array([100, 100, 0])  # [sigma_x, sigma_y, tau_xy]
    strain_n = np.array([0, 0, 0])  # [epsilon_x, epsilon_y, gamma_xy]
    
    # Incremental loading
    num_steps = 500
    strain_increment = np.array([1e-5, 0, 0])  # Axial loading in x-direction
    
    for _ in range(num_steps):
        # Update stress and strain
        stress_updated, strain_updated, _ = update_stress_strain(
            stress_n, strain_n, strain_increment, D, alpha, k)
        strain_history.append(strain_updated)
        
        # Calculate p and q
        I1, J2 = compute_invariants(stress_updated)
        sigma_1, sigma_2 = principal_stresses(stress_updated)
        p = I1 / 3
        q = np.sqrt(3 * J2)
        
        # Store results
        I1_history.append(I1)
        J2_history.append(J2)
        p_history.append(p)
        q_history.append(q)
        axial_strain_history.append(strain_updated[0])
        deviatoric_stress_history.append(q)
        
        # Update for next step
        stress_n = stress_updated
        strain_n = strain_updated
    
    
    # Convert history lists to numpy arrays for easier indexing
    axial_strain_history = np.array(axial_strain_history)
    deviatoric_stress_history = np.array(deviatoric_stress_history)
    I1_history = np.array(I1_history)
    J2_history = np.array(J2_history)
    
    # Create a colormap based on the number of steps
    num_points = len(axial_strain_history)
    colors = cm.viridis(np.linspace(0, 1, num_points))
    
    # Create a colormap based on the number of steps
    num_points = len(axial_strain_history)
    colors = cm.viridis(np.linspace(0, 1, num_points))
    


    # Plot axial strain vs deviatoric stress with color gradient
    fig, ax = plt.subplots(figsize=(10, 6))
    sc1 = ax.scatter(axial_strain_history, deviatoric_stress_history, c=colors, s=10)
    ax.set_xlabel('Axial Strain')
    ax.set_ylabel('Deviatoric Stress (Pa)')
    ax.set_title('Axial Strain vs Deviatoric Stress')
    ax.grid(True)
    fig.colorbar(cm.ScalarMappable(cmap='viridis'), ax=ax, label='Load Step')
    ax.legend()
    plt.show()
    
    # Strain history
    strain_names = ['e_xx', 'e_yy', 'e_xy']
    strain_history = np.array(strain_history)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    for i in range(3):
        axes[i].plot(strain_history[:, i], label=strain_names[i])
        axes[i].set_xlabel('Load Step')
        axes[i].set_ylabel('Strain')
        axes[i].legend()
        axes[i].grid(True)
    plt.tight_layout()
    plt.show()
    
    # yield function line in terms of I1 and J2
    I1 = np.linspace(-10, 10000, 100)
    J2 = (alpha * I1 - k)**2

    # Plot I1 vs J2 stress path with yield function line and color gradient
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(I1, J2, 'r--', label='Yield Function Line')
    ax.scatter(I1_history, J2_history, c=colors, s=10, label='Stress Path')
    ax.set_xlabel('$I_1$ (First Invariant)')
    ax.set_ylabel('$J_2$ (Second Deviatoric Invariant)')
    ax.set_title('Stress Path with Yield Function in $I_1$-$J_2$ Space')
    ax.legend()
    ax.grid(True)
    plt.show()


    # # I1 vs J2
    # fig, ax = plt.subplots(figsize=(10, 6))
    # ax.plot(I1, J2, 'r--', label='Yield Function Line')
    # ax.scatter(I1_history, J2_history, c=colors, s=10, label='Stress Path')
    # ax.set_xlabel('$I_1$ (First Invariant)')
    # ax.set_ylabel('$J_2$ (Second Deviatoric Invariant)')
    # ax.set_title('Stress Path with Yield Function in $I_1$-$J_2$ Space')
    # ax.legend()
    # ax.grid(True)
    # plt.show()