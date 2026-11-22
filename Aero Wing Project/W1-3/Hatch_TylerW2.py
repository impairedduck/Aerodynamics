import numpy as np
import json
import matplotlib.pyplot as plt

# functions

# read planform data from json
def read_planform(filename="W2_input.json"):
    with open(filename, "r") as f:
        data = json.load(f)

    planform = data["wing"]["planform"]
    planform_type = planform["type"] # "elliptic" or "tapered"
    aspect_ratio  = planform["aspect_ratio"]
    taper_ratio   = planform.get("taper_ratio", None) # elliptic doesn't need it

    return planform_type, aspect_ratio, taper_ratio

# read wing properties from json
def read_wing_properties(filename="W2_input.json"):
    with open(filename, "r") as f:
        data = json.load(f)

    wing = data["wing"]
    airfoil_lift_slope = wing["airfoil_lift_slope"] # per radian
    nodes_per_semispan = int(wing["nodes_per_semispan"])

    return airfoil_lift_slope, nodes_per_semispan

# read conditions from json
def read_condition(filename="W2_input.json"):
    with open(filename, "r") as f:
        data = json.load(f)

    condition = data["condition"]
    alpha_root_deg = condition["alpha_root[deg]"]

    return alpha_root_deg

# read view state from json and read more washout stuff
def read_view(filename="W2_input.json"):
    with open(filename, "r") as f:
        data = json.load(f)

    view = data["view"]
    show_planform = view.get("planform", False)
    show_washout = view.get("washout_distribution", False)

    return show_planform, show_washout

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

# compute wing coefficients from the Fourier coefficients with washout
def compute_coefficients(a, b, aspect_ratio, airfoil_lift_slope, alpha_root_deg, distribution, amount, CL_design):
    N = len(a)
    n = np.arange(1, N + 1)                        
    alpha_root = np.radians(alpha_root_deg)        

    kappa_D = np.sum(n[1:] * (a[1:] / a[0])**2)
    e_s = 1.0 / (1.0 + kappa_D)
    CL_alpha = np.pi * aspect_ratio * a[0]
    term = 1.0 + np.pi * aspect_ratio / airfoil_lift_slope
    kappa_L = (1.0 - term * a[0]) / (term * a[0])

    # washout factors
    eps_omega = b[0] / a[0]
    x = b[1:] / a[0] - eps_omega * a[1:] / a[0]
    kappa_DL = 2.0 * np.sum(n[1:] * (a[1:] / a[0]) * x)
    kappa_DOmega = np.sum(n[1:] * x**2)

    if distribution == "none":
        Omega = 0.0
    elif amount == "optimum":
        Omega = kappa_DL * CL_design / (2.0 * kappa_DOmega * CL_alpha)
    else:
        Omega = np.radians(amount)

    CL = CL_alpha * (alpha_root - eps_omega * Omega)
    CDi = (CL**2 * (1.0 + kappa_D) - kappa_DL * CL * CL_alpha * Omega
           + kappa_DOmega * (CL_alpha * Omega)**2) / (np.pi * aspect_ratio)

    return kappa_L, kappa_D, e_s, CL_alpha, eps_omega, kappa_DL, kappa_DOmega, Omega, CL, CDi

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

# read washout stuff from json
def read_washout(filename="W2_input.json"):
    with open(filename, "r") as f:
        data = json.load(f)

    washout = data["wing"].get("washout", {})     
    distribution = washout.get("distribution", "none")
    amount = washout.get("amount[deg]", 0.0)
    CL_design = washout.get("CL_design", None)

    return distribution, amount, CL_design

# compute washout distribution at each node
def compute_washout(z_b, c_b, distribution, planform_type):
    if distribution == "none":
        omega = np.zeros(len(z_b))
    elif distribution == "linear":
        omega = np.abs(2.0 * z_b)
    elif distribution == "optimum":
        if planform_type == "elliptic":
            omega = np.zeros(len(z_b))
        else:
            c_root = c_b[len(c_b) // 2]
            omega = 1.0 - np.sqrt(1.0 - (2.0 * z_b)**2) / (c_b / c_root)
    else:
        raise ValueError("Washout distribution must be 'none', 'linear', or 'optimum'.")
    return omega

# plot the washout distribution
def plot_washout(z_b, omega):
    plt.figure()
    plt.plot(z_b, omega, 'b-o')
    plt.xlabel("z/b")
    plt.ylabel("omega")
    plt.title("Washout Distribution")
    plt.grid()
    plt.show()

# plot CDi vs CL with and without 5 deg linear washout
def plot_drag_polar(z_b, a, C_inv, aspect_ratio, airfoil_lift_slope, planform_type):
    CL = np.linspace(0.0, 1.0, 100)
    plt.figure()
    for distribution, amount, label in [("none", 0.0, "No washout"), ("linear", 5.0, "5 deg linear washout")]:
        omega = compute_washout(z_b, None, distribution, planform_type)
        b = np.matmul(C_inv, omega)
        kappa_L, kappa_D, e_s, CL_alpha, eps_omega, kappa_DL, kappa_DOmega, Omega, _, _ = compute_coefficients(
            a, b, aspect_ratio, airfoil_lift_slope, 0.0, distribution, amount, None)
        CDi = (CL**2 * (1.0 + kappa_D) - kappa_DL * CL * CL_alpha * Omega
               + kappa_DOmega * (CL_alpha * Omega)**2) / (np.pi * aspect_ratio)
        plt.plot(CL, CDi, label=label)
    plt.xlabel("CL")
    plt.ylabel("CDi")
    plt.title("Induced Drag vs Lift, " + planform_type + " wing")
    plt.legend()
    plt.grid()
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
show_planform, show_washout = read_view()
print("\nView planform:", show_planform)
print("View washout:", show_washout)

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

# washout
distribution, amount, CL_design = read_washout()
print("\nWashout distribution:", distribution)
print("Washout amount [deg]:", amount)
print("CL design:", CL_design)

omega = compute_washout(z_b, c_b, distribution, planform_type)
print("\nomega:\n", omega)

# washout Fourier coefficients
b = np.matmul(C_inv, omega)
print("\nb:\n", b)

# write C matrix, inverse, and Fourier coefficients to text file
with open("Solution.txt", "w") as file:
    # C matrix
    file.write("C_matrix:\n")
    np.savetxt(file, C, fmt='%f', delimiter='\t')

    # inverse of C
    file.write("\nC_matrix_inverse:\n")
    np.savetxt(file, C_inv, fmt='%f', delimiter='\t')

    # fourier coefficients
    file.write("\nFourier Coefficients (a vals):\n")
    np.savetxt(file, a, fmt='%f', delimiter='\t')

    # b matrix
    file.write("\nFourier Coefficients (b vals):\n")
    np.savetxt(file, b, fmt='%f', delimiter='\t')

# wing coefficients
kappa_L, kappa_D, e_s, CL_alpha, eps_omega, kappa_DL, kappa_DOmega, Omega, CL, CDi = compute_coefficients(a, b, aspect_ratio, airfoil_lift_slope, alpha_root_deg, distribution, amount, CL_design)
print("\nkappa_L:", kappa_L)
print("kappa_D:", kappa_D)
print("e_s:", e_s)
print("CL_alpha [1/rad]:", CL_alpha)
print("eps_Omega:", eps_omega)
print("kappa_DL:", kappa_DL)
print("kappa_DOmega:", kappa_DOmega)
print("Omega [deg]:", np.degrees(Omega))
print("CL:", CL)
print("CDi:", CDi)

# plots
if show_planform:
    plot_planform(z_b, c_b)
if show_washout:
    plot_washout(z_b, omega)
plot_drag_polar(z_b, a, C_inv, aspect_ratio, airfoil_lift_slope, planform_type)