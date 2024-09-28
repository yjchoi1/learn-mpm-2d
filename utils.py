import numpy as np

# Function to generate particles within a circle
def generate_circle_particles(center, radius, spacing):
    # Determine the grid origin, so the grid is consistent regardless of the center
    grid_origin = np.array(center) // spacing * spacing
    
    # Calculate grid limits
    xmin = grid_origin[0] - radius
    xmax = grid_origin[0] + radius
    ymin = grid_origin[1] - radius
    ymax = grid_origin[1] + radius
    
    # Generate grid points
    x_coords = np.arange(xmin, xmax + spacing, spacing)
    y_coords = np.arange(ymin, ymax + spacing, spacing)
    Xp, Yp = np.meshgrid(x_coords, y_coords)
    Xp = Xp.flatten()
    Yp = Yp.flatten()
    
    # Keep only points inside the circle
    inside = (Xp - center[0])**2 + (Yp - center[1])**2 <= radius**2
    Xp = Xp[inside]
    Yp = Yp[inside]
    
    return np.vstack([Xp, Yp]).T


def particle_element_mapping(
    xp, delta_x, delta_y, n_ele_x, n_elements):
    """
    Find the elements to which each particle belongs.
    This is only for quad element.

    Parameters:
    - xp (ndarray): Array of particle positions.
    - delta_x (float): Cell size in the x-direction.
    - delta_y (float): Cell size in the y-direction.
    - n_ele_x (int): Number of elements in the x-direction.

    Returns:
    - ele_ids_of_particles (ndarray): element indices to which each particle belongs.
    - p_ids_in_eles (list): List of particle indices for each element.
    """
    # Vectorized computation of element ids for each particle
    x_indices = np.floor(xp[:, 0] / delta_x).astype(int)
    y_indices = np.floor(xp[:, 1] / delta_y).astype(int)
    
    # Compute the element index using the formula (no need for a loop here)
    ele_ids_of_particles = x_indices + n_ele_x * y_indices
    
    # Create list of particle indices for each element
    p_ids_in_eles = [[] for _ in range(n_elements)]
    for p, e in enumerate(ele_ids_of_particles):
        p_ids_in_eles[e].append(p)
    
    return ele_ids_of_particles, p_ids_in_eles


def define_elements(
    node_pattern, num_cols, num_rows, col_increment, row_increment):
    """
    Creates a connectivity list for grid elements (i.e., cell).

    Parameters:
    node_pattern (array-like): The pattern of node connections for a single element.
    For example, the node patter is as follows if `n_node_x` is 21.
        42---43--- ...    ---62
        | 20 | 21 |     |    |
        21---22--- ...    ---41 
        | 0  | 1  |     | 19 |
        0----1---  ...    ---20
    num_cols (int): Number of columns in the grid.
    num_rows (int): Number of rows in the grid.
    col_increment (int): Increment for moving to the next column.
    row_increment (int): Increment for moving to the next row.

    Returns:
    numpy.ndarray: A 2D array representing the connectivity list.
    """
    
    node_pattern = np.array(node_pattern)
    increment = np.zeros(node_pattern.shape, dtype=int)
    element_index = 0
    connectivity_list = np.zeros((num_cols * num_rows, node_pattern.shape[0]), dtype=int)

    for row in range(num_rows):
        for col in range(num_cols):
            connectivity_list[element_index, :] = node_pattern + increment
            increment += col_increment
            element_index += 1
        increment = (row + 1) * row_increment

    return connectivity_list


def create_square_grid(start, length, n_node_x, n_node_y):
    # Create 1D arrays for x and y
    x = np.linspace(start, start + length, n_node_x, endpoint=True)
    y = np.linspace(start, start + length, n_node_y, endpoint=True)
    
    # Create 2D meshgrid
    X, Y = np.meshgrid(x, y)
    
    # Reshape and stack to get the desired output format
    grid_points = np.column_stack((X.ravel(), Y.ravel()))
    
    return grid_points


def square_node_array(corner1, corner2, corner3, corner4, num_nodes_u, num_nodes_v, u_ratio=1, v_ratio=1):
    def lagrange_basis_Q4(local_coords):
        """
        Calculate the Lagrange basis functions for a 4-node quadrilateral element.
        
        local_coords: List or array containing the local coordinates [xi, eta].
        
        Returns a numpy array of shape (4,) containing the basis functions 
        evaluated at the given local coordinates.
        """
        xi, eta = local_coords
        N1 = 0.25 * (1 - xi) * (1 - eta)  # Basis function associated with corner1
        N2 = 0.25 * (1 + xi) * (1 - eta)  # Basis function associated with corner2
        N3 = 0.25 * (1 + xi) * (1 + eta)  # Basis function associated with corner3
        N4 = 0.25 * (1 - xi) * (1 + eta)  # Basis function associated with corner4
        return np.array([N1, N2, N3, N4])

    # Determine the spacing of nodes along the u direction (from corner1 to corner2)
    if u_ratio == 1:
        # Uniform spacing if u_ratio is 1
        u_coords = np.linspace(-1, 1, num_nodes_u)
    elif u_ratio > 0:
        # Non-uniform spacing if u_ratio > 0
        u_spacing_ratio = u_ratio ** (1 / (num_nodes_u - 2))  # Calculate the spacing ratio
        u_coords = np.zeros(num_nodes_u)
        spacing = 1
        for i in range(1, num_nodes_u):
            u_coords[i] = u_coords[i - 1] + spacing  # Increase by the current spacing
            spacing /= u_spacing_ratio  # Decrease spacing by the ratio
        u_coords = 2 * u_coords / u_coords[-1] - 1  # Normalize to range [-1, 1]
    else:
        # Error handling for invalid u_ratio
        raise ValueError("u_ratio must be greater than 0")
        u_coords = np.linspace(-1, 1, num_nodes_u)

    # Determine the spacing of nodes along the v direction (from corner2 to corner3)
    if v_ratio == 1:
        # Uniform spacing if v_ratio is 1
        v_coords = np.linspace(-1, 1, num_nodes_v)
    elif v_ratio > 0:
        # Non-uniform spacing if v_ratio > 0
        v_spacing_ratio = v_ratio ** (1 / (num_nodes_v - 2))  # Calculate the spacing ratio
        v_coords = np.zeros(num_nodes_v)
        spacing = 1
        for i in range(1, num_nodes_v):
            v_coords[i] = v_coords[i - 1] + spacing  # Increase by the current spacing
            spacing /= v_spacing_ratio  # Decrease spacing by the ratio
        v_coords = 2 * v_coords / v_coords[-1] - 1  # Normalize to range [-1, 1]
    else:
        # Error handling for invalid v_ratio
        raise ValueError("v_ratio must be greater than 0")
        v_coords = np.linspace(-1, 1, num_nodes_v)

    # Coordinates of the four corner points
    x_corners = np.array([corner1[0], corner2[0], corner3[0], corner4[0]])
    y_corners = np.array([corner1[1], corner2[1], corner3[1], corner4[1]])

    # Initialize the output array to hold the coordinates of the generated nodes
    node_coords = np.zeros((num_nodes_u * num_nodes_v, 2))

    # Generate the node positions by interpolating between the corner points
    for row in range(num_nodes_v):
        eta = v_coords[row]  # Get the local coordinate eta for the current row
        for col in range(num_nodes_u):
            xi = u_coords[col]  # Get the local coordinate xi for the current column
            # Compute the shape functions (interpolation basis) at (xi, eta)
            shape_functions = lagrange_basis_Q4([xi, eta])
            # Compute the global coordinates of the node using the shape functions
            node_coords[row * num_nodes_u + col, :] = [
                np.dot(x_corners, shape_functions),
                np.dot(y_corners, shape_functions)
            ]

    return node_coords


def elasticity_matrix(E0, nu0, stress_state):
    """
    Elasticity matrix for isotropic elastic materials.

    Parameters:
    E0 (float): Young's modulus
    nu0 (float): Poisson's ratio
    stress_state (str): Stress state, either 'PLANE_STRESS', 'PLANE_STRAIN', or '3D'

    Returns:
    np.ndarray: Elasticity matrix
    """
    
    if stress_state == 'PLANE_STRESS':  # Plane Stress case
        C = E0 / (1 - nu0**2) * np.array([
            [1, nu0, 0],
            [nu0, 1, 0],
            [0, 0, (1 - nu0) / 2]
        ])
        
    elif stress_state == 'PLANE_STRAIN':  # Plane Strain case
        C = E0 / ((1 + nu0) * (1 - 2 * nu0)) * np.array([
            [1 - nu0, nu0, 0],
            [nu0, 1 - nu0, 0],
            [0, 0, 1/2 - nu0]
        ])
        
    else:  # 3D case
        C = np.zeros((6, 6))
        C[:3, :3] = E0 / ((1 + nu0) * (1 - 2 * nu0)) * np.array([
            [1 - nu0, nu0, nu0],
            [nu0, 1 - nu0, nu0],
            [nu0, nu0, 1 - nu0]
        ])
        C[3:, 3:] = E0 / (2 * (1 + nu0)) * np.eye(3)
    
    return C


def lagrange_basis(type, coord, dim=1):
    """
    Returns the Lagrange interpolant basis and its gradients with respect to the parent coordinate system.

    Parameters:
        type (str): Topological class of the finite element. Examples: 'L2', 'L3', 'T3', 'Q4', etc.
        coord (list or numpy array): Parent coordinates at which the basis and its gradients are to be evaluated.
        dim (int, optional): Dimension of the vector representation of the N matrix. Default is 1.

    Returns:
        tuple: (N, dNdxi)
            N: Lagrange interpolant basis evaluated at coord.
            dNdxi: Gradients of the basis with respect to the parent coordinates.
    """

    if type == 'L2':
        # L2 TWO NODE LINE ELEMENT
        if len(coord) < 1:
            raise ValueError('Error coordinate needed for the L2 element')
        else:
            xi = coord[0]
            N = np.array([1 - xi, 1 + xi]) / 2
            dNdxi = np.array([-1, 1]) / 2
    
    elif type == 'L3':
        # L3 THREE NODE LINE ELEMENT
        if len(coord) < 1:
            raise ValueError('Error coordinate needed for the L3 element')
        else:
            xi = coord[0]
            N = np.array([(1 - xi) * xi / -2, (1 + xi) * xi / 2, 1 - xi ** 2])
            dNdxi = np.array([xi - 0.5, xi + 0.5, -2 * xi])
    
    elif type == 'T3':
        # T3 THREE NODE TRIANGULAR ELEMENT
        #   
        #               3
        #             /  \
        #            /    \
        #           /      \
        #          /        \
        #         /          \
        #        /            \
        #       /              \
        #      /                \
        #     /                  \
        #    1--------------------2
    
        if len(coord) < 2:
            raise ValueError('Error two coordinates needed for the T3 element')
        else:
            xi, eta = coord
            N = np.array([1 - xi - eta, xi, eta])
            dNdxi = np.array([[-1, -1], [1, 0], [0, 1]])
    
    elif type == 'T3fs':
        # T3fs THREE NODE TRIANGULAR ELEMENT with free surface
        if len(coord) < 2:
            raise ValueError('Error two coordinates needed for the T3fs element')
        else:
            xi, eta = coord
            N = np.array([1 - xi - eta, xi, eta])
            dNdxi = np.array([[-1, -1], [1, 0], [0, 1]])
    
    elif type == 'T4':
        # T4 FOUR NODE TRIANGULAR CUBIC BUBBLE ELEMENT
        #   
        #               3
        #             /  \
        #            /    \
        #           /      \
        #          /        \
        #         /          \
        #        /      4     \
        #       /              \
        #      /                \
        #     /                  \
        #    1--------------------2
        
        if len(coord) < 2:
            raise ValueError('Error two coordinates needed for the T4 element')
        else:
            xi, eta = coord
            N = np.array([1 - xi - eta - 3 * xi * eta,
                          xi * (1 - 3 * eta),
                          eta * (1 - 3 * xi),
                          9 * xi * eta])
            dNdxi = np.array([[-1 - 3 * eta, -1 - 3 * xi],
                              [1 - 3 * eta, -3 * xi],
                              [-3 * eta, 1 - 3 * xi],
                              [9 * eta, 9 * xi]])
    
    elif type == 'T6':
        # T6 SIX NODE TRIANGULAR ELEMENT
        #   
        #               3
        #             /  \
        #            /    \
        #           /      \
        #          /        \
        #         6          5
        #        /            \
        #       /              \
        #      /                \
        #     /                  \
        #    1---------4----------2
        
        if len(coord) < 2:
            raise ValueError('Error two coordinates needed for the T6 element')
        else:
            xi, eta = coord
            N = np.array([1 - 3 * (xi + eta) + 4 * xi * eta + 2 * (xi ** 2 + eta ** 2),
                          xi * (2 * xi - 1),
                          eta * (2 * eta - 1),
                          4 * xi * (1 - xi - eta),
                          4 * xi * eta,
                          4 * eta * (1 - xi - eta)])
            dNdxi = np.array([[4 * (xi + eta) - 3, 4 * (xi + eta) - 3],
                              [4 * xi - 1, 0],
                              [0, 4 * eta - 1],
                              [4 * (1 - eta - 2 * xi), -4 * xi],
                              [4 * eta, 4 * xi],
                              [-4 * eta, 4 * (1 - xi - 2 * eta)]])
    
    elif type == 'Q4':
        # Q4 FOUR NODE QUADRILATERAL ELEMENT
        #
        #    4--------------------3
        #    |                    |
        #    |                    |
        #    |                    |
        #    |                    |
        #    |                    |
        #    |                    |
        #    |                    |
        #    |                    |
        #    |                    |
        #    1--------------------2
        
        xi, eta = coord
        N = np.array([(1 - xi) * (1 - eta),
                      (1 + xi) * (1 - eta),
                      (1 + xi) * (1 + eta),
                      (1 - xi) * (1 + eta)]) / 4
        dNdxi = np.array([[-(1 - eta), -(1 - xi)],
                          [(1 - eta), -(1 + xi)],
                          [(1 + eta), (1 + xi)],
                          [-(1 + eta), (1 - xi)]]) / 4
    
    elif type == 'Q9':
        # Q9 NINE NODE QUADRILATERAL ELEMENT
        #
        #    4---------7----------3
        #    |                    |
        #    |                    |
        #    |                    |
        #    |                    |
        #    8          9         6
        #    |                    |
        #    |                    |
        #    |                    |
        #    |                    |
        #    1----------5---------2
        
        if len(coord) < 2:
            raise ValueError('Error two coordinates needed for the Q9 element')
        else:
            xi, eta = coord
            N = np.array([xi * eta * (xi - 1) * (eta - 1),
                          xi * eta * (xi + 1) * (eta - 1),
                          xi * eta * (xi + 1) * (eta + 1),
                          xi * eta * (xi - 1) * (eta + 1),
                          -2 * eta * (xi + 1) * (xi - 1) * (eta - 1),
                          -2 * xi * (xi + 1) * (eta + 1) * (eta - 1),
                          -2 * eta * (xi + 1) * (xi - 1) * (eta + 1),
                          -2 * xi * (xi - 1) * (eta + 1) * (eta - 1),
                          4 * (xi + 1) * (xi - 1) * (eta + 1) * (eta - 1)]) / 4
            dNdxi = np.array([[eta * (2 * xi - 1) * (eta - 1), xi * (xi - 1) * (2 * eta - 1)],
                              [eta * (2 * xi + 1) * (eta - 1), xi * (xi + 1) * (2 * eta - 1)],
                              [eta * (2 * xi + 1) * (eta + 1), xi * (xi + 1) * (2 * eta + 1)],
                              [eta * (2 * xi - 1) * (eta + 1), xi * (xi - 1) * (2 * eta + 1)],
                              [-4 * xi * eta * (eta - 1), -2 * (xi + 1) * (xi - 1) * (2 * eta - 1)],
                              [-2 * (2 * xi + 1) * (eta + 1) * (eta - 1), -4 * xi * eta * (xi + 1)],
                              [-4 * xi * eta * (eta + 1), -2 * (xi + 1) * (xi - 1) * (2 * eta + 1)],
                              [-2 * (2 * xi - 1) * (eta + 1) * (eta - 1), -4 * xi * eta * (xi - 1)],
                              [8 * xi * (eta ** 2 - 1), 8 * eta * (xi ** 2 - 1)]]) / 4
    
    elif type == 'H4':
        # H4 FOUR NODE TETRAHEDRAL ELEMENT
        #
        #             4
        #           / | \
        #          /  |  \
        #         /   |   \ 
        #        /    |    \ 
        #       /     |     \
        #      1 -----|------3
        #         -   2  -
    
        if len(coord) < 3:
            raise ValueError('Error three coordinates needed for the H4 element')
        else:
            xi, eta, zeta = coord
            N = np.array([1 - xi - eta - zeta, xi, eta, zeta])
            dNdxi = np.array([[-1, -1, -1],
                              [1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]])
    
    elif type == 'H10':
        # H10 TEN NODE TETRAHEDRAL ELEMENT (Not yet supported)
        raise NotImplementedError('Element H10 not yet supported')
    
    elif type == 'B8':
        # B8 EIGHT NODE BRICK ELEMENT
        # 
        #                   8 
        #                /    \    
        #             /          \
        #          /                \
        #       5                     \
        #       |\                     7
        #       |   \                / |
        #       |     \     4    /     |
        #       |        \    /        |
        #       |           6          |
        #       1           |          |
        #        \          |          3
        #           \       |        /
        #             \     |     /
        #                \  |  /
        #                   2
        
        if len(coord) < 3:
            raise ValueError('Error three coordinates needed for the B8 element')
        else:
            xi, eta, zeta = coord
            I1 = 1 / 2 - np.array(coord) / 2
            I2 = 1 / 2 + np.array(coord) / 2
            N = np.array([I1[0] * I1[1] * I1[2],
                          I2[0] * I1[1] * I1[2],
                          I2[0] * I2[1] * I1[2],
                          I1[0] * I2[1] * I1[2],
                          I1[0] * I1[1] * I2[2],
                          I2[0] * I1[1] * I2[2],
                          I2[0] * I2[1] * I2[2],
                          I1[0] * I2[1] * I2[2]])
            dNdxi = np.array(
                [[-1 + eta + zeta - eta * zeta, -1 + xi + zeta - xi * zeta, -1 + xi + eta - xi * eta],
                 [1 - eta - zeta + eta * zeta, -1 - xi + zeta + xi * zeta, -1 - xi + eta + xi * eta],
                 [1 + eta - zeta - eta * zeta, 1 + xi - zeta - xi * zeta, -1 - xi - eta - xi * eta],
                 [-1 - eta + zeta + eta * zeta, 1 - xi - zeta + xi * zeta, -1 + xi - eta + xi * eta],
                 [-1 + eta - zeta + eta * zeta, -1 + xi - zeta + xi * zeta, 1 - xi - eta + xi * eta],
                 [1 - eta + zeta - eta * zeta, -1 - xi - zeta - xi * zeta, 1 + xi - eta - xi * eta],
                 [1 + eta + zeta + eta * zeta, 1 + xi + zeta + xi * zeta, 1 + xi + eta + xi * eta],
                 [-1 - eta - zeta - eta * zeta, 1 - xi + zeta - xi * zeta, 1 - xi + eta - xi * eta]]) / 8
    
    else:
        raise ValueError(f"Element {type} not yet supported")
    
    return N, dNdxi
