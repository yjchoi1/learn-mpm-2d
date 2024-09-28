from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import LineCollection
from matplotlib.collections import PolyCollection
import matplotlib.animation as animation
import numpy as np
import itertools
from pyevtk.hl import pointsToVTK
import os


def save_vtk(istep, result, save_dir):
    """_summary_

    Args:
        istep (int): _description_
        result (dict): _description_
        save_dir (str): _description_
    """
    stress_xx = np.ascontiguousarray(result["stress"][:, 0])
    stress_yy = np.ascontiguousarray(result["stress"][:, 1])
    stress_xy = np.ascontiguousarray(result["stress"][:, 2])
    mean_stress = 0.5 * (stress_xx + stress_yy)
    dev_stress_xx = stress_xx - mean_stress
    dev_stress_yy = stress_yy - mean_stress
    dev_stress_xy = stress_xy  
    # Stress magnitude based on second invariant of the deviatoric stress tensor
    von_mises_stress = np.sqrt(
        0.5 * (dev_stress_xx**2 + dev_stress_yy**2 + 6 * dev_stress_xy**2)
    )
        
    if not os.path.exists(f"{save_dir}"):
        os.makedirs(f"{save_dir}")
    
    pointsToVTK(
        f"{save_dir}/result-{istep}",
        x=np.ascontiguousarray(result["positions"][:, 0]),
        y=np.ascontiguousarray(result["positions"][:, 1]),
        z=np.ascontiguousarray(np.zeros(result["positions"][:, 1].shape)),
        data={
            "stress_xx": np.ascontiguousarray(stress_xx),
            "stress_yy": np.ascontiguousarray(stress_yy),
            "stress_xy": np.ascontiguousarray(stress_xy),
            "stress_mag": np.ascontiguousarray(von_mises_stress)
        }
    )


def animate_particle_positions(pos, interval=50, save_path=None):
    """
    Function to animate particle positions over time.
    
    Parameters:
    pos (list): List of numpy arrays containing particle positions at each time step.
    interval (int): Time interval between frames in milliseconds (default is 50ms).
    save_path (str): Filename to save the animation as a video (e.g., 'animation.mp4').
                   If None, the animation will just be displayed without saving.
    """
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)  # Adjust based on your domain size
    ax.set_ylim(0, 1)  # Adjust based on your domain size
    
    scatter = ax.scatter([], [], c='blue', s=10)  # Initialize scatter plot for particles

    def init():
        scatter.set_offsets([])
        return scatter,

    def update(frame):
        scatter.set_offsets(pos[frame])
        return scatter,

    ani = animation.FuncAnimation(
        fig, update, frames=len(pos), init_func=init, blit=True, interval=interval
    )

    if save_path:
        ani.save(save_path, writer='ffmpeg')
    else:
        plt.show()


def plot_2d_meshes(meshes, particles):
    """
    Plots multiple 2D meshes using node coordinates and face indices.

    Parameters:
    - meshes: A list of tuples, where each tuple contains:
        - node_coords: A NumPy array of shape (n_nodes, 2) containing the (x, y) coordinates of the mesh nodes.
        - faces: A list or array of faces, where each face is represented by a list of node indices (referring to the node_coords).
    - particles: A numpy array for particle coordinates (n_particles, 2) containing the (x, y) positions of the particles.

    Example:
    - meshes = [
        (np.array([[0, 0], [1, 0], [1, 1], [0, 1]]), [[0, 1, 2, 3]]),
        (np.array([[2, 2], [3, 2], [3, 3], [2, 3]]), [[0, 1, 2, 3]])
      ]
    """
    
    # Define a color cycle for different meshes
    colors = itertools.cycle(plt.cm.tab10.colors)  # Using tab10 colormap for up to 10 distinct colors
        
    # Set up the plot
    fig, ax = plt.subplots()
    
    # Iterate over the meshes and plot each one
    for node_coords, faces in meshes:
        # Create a list of polygons using the face indices and node coordinates
        polygons = [node_coords[face] for face in faces]
        
        # Create a PolyCollection from the polygons
        poly_collection = PolyCollection(
            polygons, edgecolors='black', facecolors='none', alpha=0.5)
        
        # Add the PolyCollection to the plot
        ax.add_collection(poly_collection)
        
    ax.scatter(particles[:, 0], particles[:, 1], color='blue', s=2.0)
    
    # Set the limits of the plot to the bounding box of all node coordinates
    all_coords = np.vstack([node_coords for node_coords, _ in meshes])
    ax.set_xlim(np.min(all_coords[:, 0]), np.max(all_coords[:, 0]))
    ax.set_ylim(np.min(all_coords[:, 1]), np.max(all_coords[:, 1]))
    ax.set_aspect('equal')
    
    # Display the plot
    plt.show()


def plot_mesh(X, connect, elem_type, line_color='w', linewidth=1):
    """
    Plot a nodal mesh and associated connectivity efficiently using LineCollection for 2D or Line3DCollection for 3D.

    Parameters:
    X (numpy.ndarray): Nodal coordinates
    connect (numpy.ndarray): Connectivity matrix
    elem_type (str): Element type ('L2', 'L3', 'T3', 'T6', 'Q4', 'Q8', 'Q9', 'H4', or 'B8')
    line_color (str or tuple): Color for plotting (default: 'w')
    linewidth (float): Line width for plotting (default: 1)
    alpha (float): Transparency of lines (default: 1)
    """

    is_3d = X.shape[1] == 3

    # Ensure X has at least 2 columns
    if X.shape[1] < 2:
        raise ValueError("X must have at least 2 columns")

    fig = plt.figure()
    if is_3d:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)

    element_orders = {
        'Q9': [0, 4, 1, 5, 2, 6, 3, 7, 0],
        'Q8': [0, 4, 1, 5, 2, 6, 3, 7, 0],
        'T3': [0, 1, 2, 0],
        'T6': [0, 3, 1, 4, 2, 5, 0],
        'Q4': [0, 1, 2, 3, 0],
        'L2': [0, 1],
        'L3': [0, 2, 1],
        'H4': [0, 1, 3, 0, 2, 3, 1, 2],
        'B8': [0, 4, 5, 1, 2, 6, 7, 3, 0, 1, 2, 3, 7, 4, 5, 6]
    }

    if elem_type not in element_orders:
        raise ValueError(f"Unsupported element type: {elem_type}")

    order = element_orders[elem_type]

    # Create a list of line segments
    segments = []
    for e in range(connect.shape[0]):
        points = X[connect[e, order], :]
        segments.extend(zip(points[:-1], points[1:]))

    if is_3d:
        # Create the Line3DCollection for 3D
        lc = Line3DCollection(segments, colors=line_color, linewidths=linewidth)
        ax.add_collection(lc)
        ax.set_zlim(X[:, 2].min(), X[:, 2].max())
        ax.set_box_aspect((np.ptp(X[:, 0]), np.ptp(X[:, 1]), np.ptp(X[:, 2])))
    else:
        # Create the LineCollection for 2D
        lc = LineCollection(segments, colors=line_color, linewidths=linewidth)
        ax.add_collection(lc)
        ax.set_aspect('equal')

    # Set plot limits
    ax.set_xlim(X[:, 0].min(), X[:, 0].max())
    ax.set_ylim(X[:, 1].min(), X[:, 1].max())

    plt.show()
