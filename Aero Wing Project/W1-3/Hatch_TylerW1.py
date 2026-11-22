import numpy as np
import json
import matplotlib.pyplot as plt

# functions

# read planform data from json
def read_planform(filename="W1_input.json"):
    with open(filename, "r") as f:
        data = json.load(f)

    planform = data["wing"]["planform"]
    planform_type = planform["type"] # "elliptic" or "tapered"
    aspect_ratio  = planform["aspect_ratio"]
    taper_ratio   = planform.get("taper_ratio", None) # elliptic doesn't need it

    return planform_type, aspect_ratio, taper_ratio

# read wing properties from json
def read_wing_properties(filename="W1_input.json"):
    with open(filename, "r") as f:
        data = json.load(f)

    wing = data["wing"]
    airfoil_lift_slope = wing["airfoil_lift_slope"] # per radian
    nodes_per_semispan = int(wing["nodes_per_semispan"])

    return airfoil_lift_slope, nodes_per_semispan

# read conditions from json
def read_condition(filename="W1_input.json"):
    """Read the operating condition from the input JSON file."""
    with open(filename, "r") as f:
        data = json.load(f)

    condition = data["condition"]
    alpha_root_deg = condition["alpha_root[deg]"]   # (alpha - alpha_L0)_root, degrees

    return alpha_root_deg

# read view state from json
def read_view(filename="W1_input.json"):
    """Read the plot settings from the input JSON file."""
    with open(filename, "r") as f:
        data = json.load(f)

    view = data["view"]
    show_planform = view.get("planform", False)   # True = draw the planform plot

    return show_planform

# compute theta at each node
def compute_theta(nodes_per_semispan):
    N = 2 * nodes_per_semispan - 1 # total nodes, must be odd
    theta = np.zeros(N)
    for i in range(N):
        theta[i] = i * np.pi / (N - 1)
    return N, theta

# compute chord at each node
def compute_chord(z_b, planform_type, aspect_ratio, taper_ratio):
    if planform_type == "elliptic":
        c_b = (4.0 / (np.pi * aspect_ratio)) * np.sqrt(1.0 - (2.0 * z_b)**2)
    elif planform_type == "tapered":
        c_b = (2.0 / (aspect_ratio * (1.0 + taper_ratio))) * (1.0 - (1.0 - taper_ratio) * np.abs(2.0 * z_b))
    else:
        raise ValueError("Planform type must be 'elliptic' or 'tapered'.")
    return c_b

# build the C matrix
def compute_C_matrix(N, theta, c_b, airfoil_lift_slope):
    C = np.zeros((N, N))
    for i in range(N):            
        for j in range(N):       
            n = j + 1            
            if i == 0:            
                C[i, j] = n**2
            elif i == N - 1:      
                C[i, j] = (-1)**(n + 1) * n**2
            else:                 
                C[i, j] = (4.0 / (airfoil_lift_slope * c_b[i]) + n / np.sin(theta[i])) * np.sin(n * theta[i])
    return C

# compute wing coefficients from the Fourier coefficients
def compute_coefficients(a, aspect_ratio, airfoil_lift_slope, alpha_root_deg):
    N = len(a)
    n = np.arange(1, N + 1)                       # Fourier term numbers 1 to N
    alpha_root = np.radians(alpha_root_deg)       # convert to radians

    # induced drag factor and span efficiency
    kappa_D = np.sum(n[1:] * (a[1:] / a[0])**2)   # sum from n = 2 to N
    e_s = 1.0 / (1.0 + kappa_D)

    # lift slope and lift slope factor
    CL_alpha = np.pi * aspect_ratio * a[0]
    term = 1.0 + np.pi * aspect_ratio / airfoil_lift_slope
    kappa_L = (1.0 - term * a[0]) / (term * a[0])

    # operating lift and induced drag
    CL = CL_alpha * alpha_root
    CDi = CL**2 / (np.pi * aspect_ratio * e_s)

    return kappa_L, kappa_D, e_s, CL_alpha, CL, CDi

# plotting
def plot_planform(z_b, c_b):
    x_le = 0.25 * c_b      
    x_te = -0.75 * c_b     

    plt.figure()
    plt.plot(z_b, x_le, 'k', label="Leading edge")
    plt.plot(z_b, x_te, 'k', label="Trailing edge")
    plt.plot(z_b, np.zeros_like(z_b), 'r', label="Lifting line")
    for i in range(len(z_b)):
        plt.plot([z_b[i], z_b[i]], [x_te[i], x_le[i]], 'b')
    plt.xlabel("z/b")
    plt.ylabel("x/b")
    plt.title("Wing Planform")
    plt.axis("equal")
    plt.legend()
    plt.show()


# planform stuff
planform_type, aspect_ratio, taper_ratio = read_planform()
print("Planform type:", planform_type)
print("Aspect ratio:", aspect_ratio)
print("Taper ratio:", taper_ratio)

# wing properties
airfoil_lift_slope, nodes_per_semispan = read_wing_properties()
print("\nAirfoil lift slope:", airfoil_lift_slope)
print("Nodes per semispan:", nodes_per_semispan)

# conditions
alpha_root_deg = read_condition()
print("\nAlpha root:", alpha_root_deg)

# view condition
show_planform = read_view()
print("\nView condition:", show_planform)

# calcualte theta
N, theta = compute_theta(nodes_per_semispan)
z_b = -0.5 * np.cos(theta) 
print("\nN:", N)
print("theta [deg]:\n", np.degrees(theta))
print("\nz/b:\n", z_b)

# chord at each node
c_b = compute_chord(z_b, planform_type, aspect_ratio, taper_ratio)
print("\nc/b:\n", c_b)

# C matrix, inverse, and Fourier coefficients
C = compute_C_matrix(N, theta, c_b, airfoil_lift_slope)
C_inv = np.linalg.inv(C)
a = np.matmul(C_inv, np.ones(N))

# write C matrix, inverse, and Fourier coefficients to text file
with open("Solution.txt", "w") as file:
    file.write("C_matrix:\n")
    np.savetxt(file, C, fmt='%f', delimiter='\t')

    file.write("\nC_matrix_inverse:\n")
    np.savetxt(file, C_inv, fmt='%f', delimiter='\t')

    file.write("\nFourier Coefficients (a vals):\n")
    np.savetxt(file, a, fmt='%f', delimiter='\t')

# wing coefficients
kappa_L, kappa_D, e_s, CL_alpha, CL, CDi = compute_coefficients(a, aspect_ratio, airfoil_lift_slope, alpha_root_deg)
print("\nkappa_L:", kappa_L)
print("kappa_D:", kappa_D)
print("e_s:", e_s)
print("CL_alpha [1/rad]:", CL_alpha)
print("CL:", CL)
print("CDi:", CDi)
print("\n")

# planform plot
if show_planform:
    plot_planform(z_b, c_b)