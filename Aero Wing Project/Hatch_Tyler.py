import numpy as np
import json
import matplotlib.pyplot as plt

# functions
# read planform data from json file
def read_planform(filename="input.json"):
    with open(filename, "r") as f:
        data = json.load(f)

    planform = data["wing"]["planform"]
    planform_type = planform["type"]
    aspect_ratio  = planform.get("aspect_ratio", None)
    taper_ratio   = planform.get("taper_ratio", None)
    planform_file = planform.get("filename", None)     

    return planform_type, aspect_ratio, taper_ratio, planform_file

# read a chord distribution file and compute Ra
def read_chord_file(planform_file):
    data = np.loadtxt(planform_file, skiprows=1)
    z_half = data[:, 0]
    c_half = data[:, 1]

    z_file = np.concatenate((-z_half[:0:-1], z_half))
    c_file = np.concatenate((c_half[:0:-1], c_half))

    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    aspect_ratio = 1.0 / trapezoid(c_file, z_file)

    c_file = np.maximum(c_file, 0.001)

    return z_file, c_file, aspect_ratio

# read wing properties from json
def read_wing_properties(filename="input.json"):
    with open(filename, "r") as f:
        data = json.load(f)

    wing = data["wing"]
    airfoil_lift_slope = wing["airfoil_lift_slope"] # per rad
    nodes_per_semispan = int(wing["nodes_per_semispan"])

    return airfoil_lift_slope, nodes_per_semispan

# read conditions from json
def read_condition(filename="input.json"):
    with open(filename, "r") as f:
        data = json.load(f)

    condition = data["condition"]
    alpha_root_deg = condition["alpha_root[deg]"]
    delta_a_deg = condition.get("aileron_deflection[deg]", 0.0) 
    pbar = condition.get("pbar", 0.0)                                                       
    CL_operating = condition.get("CL", None)

    return alpha_root_deg, delta_a_deg, pbar, CL_operating

# read view state from json and read more washout and aileron stuff
def read_view(filename="input.json"):
    with open(filename, "r") as f:
        data = json.load(f)

    view = data["view"]
    show_planform = view.get("planform", False)
    show_washout = view.get("washout_distribution", False)
    show_aileron = view.get("aileron_distribution", False)

    return show_planform, show_washout, show_aileron

# compute theta at each node
def compute_theta(nodes_per_semispan):
    N = 2 * nodes_per_semispan - 1 # needs to be an odd number
    theta = np.zeros(N)
    for i in range(N):
        theta[i] = i * np.pi / (N - 1)
    return N, theta

# compute chord at each node
def compute_chord(z_b, planform_type, aspect_ratio, taper_ratio, chord_data=None):
    if planform_type == "elliptic":
        c_b = (4.0 / (np.pi * aspect_ratio)) * np.sqrt(1.0 - (2.0 * z_b)**2)
    elif planform_type == "tapered":
        c_b = (2.0 / (aspect_ratio * (1.0 + taper_ratio))) * (1.0 - (1.0 - taper_ratio) * np.abs(2.0 * z_b))
    elif planform_type == "file":
        z_file, c_file = chord_data
        c_b = np.interp(z_b, z_file, c_file)
    else:
        raise ValueError("Planform type must be 'elliptic', 'tapered', or 'file'.")
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

# compute wing coefficients from the fourier coefficients with washout
def compute_coefficients(a, b, aspect_ratio, airfoil_lift_slope, alpha_root_deg, distribution, amount, CL_design, CL_operating):
    N = len(a)
    n = np.arange(1, N + 1)                              

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

    if alpha_root_deg == "CL":
        alpha_root = CL_operating / CL_alpha + eps_omega * Omega
    else:
        alpha_root = np.radians(alpha_root_deg)

    CL = CL_alpha * (alpha_root - eps_omega * Omega)
    CDi = (CL**2 * (1.0 + kappa_D) - kappa_DL * CL * CL_alpha * Omega
           + kappa_DOmega * (CL_alpha * Omega)**2) / (np.pi * aspect_ratio)

    return kappa_L, kappa_D, e_s, CL_alpha, eps_omega, kappa_DL, kappa_DOmega, Omega, np.degrees(alpha_root), CL, CDi

# plotting
def plot_planform(z_b, c_b, z_begin, z_end, cf_begin, cf_end, planform_type, aspect_ratio, taper_ratio, chord_data=None):
    x_le = 0.25 * c_b
    x_te = -0.75 * c_b

    plt.figure()
    plt.plot(z_b, x_le, 'k', label="Leading edge")
    plt.plot(z_b, x_te, 'k', label="Trailing edge")
    plt.plot(z_b, np.zeros_like(z_b), 'r', label="Lifting line")
    for i in range(len(z_b)):
        plt.plot([z_b[i], z_b[i]], [x_te[i], x_le[i]], 'b')

    c_ends = compute_chord(np.array([z_begin, z_end]), planform_type, aspect_ratio, taper_ratio, chord_data)
    x_te_ends = -0.75 * c_ends
    x_hinge_ends = x_te_ends + np.array([cf_begin, cf_end]) * c_ends
    for side in [-1.0, 1.0]:
        z_ends = side * np.array([z_begin, z_end])
        plt.plot(z_ends, x_hinge_ends, 'g', label="Aileron" if side < 0 else None)
        for k in range(2):
            plt.plot([z_ends[k], z_ends[k]], [x_te_ends[k], x_hinge_ends[k]], 'g')

    plt.xlabel("z/b")
    plt.ylabel("x/b")
    plt.title("Wing Planform")
    plt.axis("equal")
    plt.legend()
    plt.show()

# read washout stuff from json
def read_washout(filename="input.json"):
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

# plot the aileron distribution
def plot_aileron(z_b, chi):
    plt.figure()
    plt.plot(z_b, chi, 'b-o')
    plt.xlabel("z/b")
    plt.ylabel("chi")
    plt.title("Aileron Distribution")
    plt.grid()
    plt.show()

# compute flap chord fraction and ideal flap effectiveness at each node
def compute_flap_effectiveness(z_b, c_b, z_begin, z_end, cf_begin, cf_end, planform_type, aspect_ratio, taper_ratio, chord_data=None):
    c_ends = compute_chord(np.array([z_begin, z_end]), planform_type, aspect_ratio, taper_ratio, chord_data)
    x_hinge_begin = -0.75 * c_ends[0] + cf_begin * c_ends[0]
    x_hinge_end   = -0.75 * c_ends[1] + cf_end * c_ends[1]

    cf_c = np.zeros(len(z_b))
    eps_fi = np.zeros(len(z_b))
    for i in range(len(z_b)):
        z = abs(z_b[i])                                    
        if z_begin <= z <= z_end:
            x_hinge = x_hinge_begin + (x_hinge_end - x_hinge_begin) * (z - z_begin) / (z_end - z_begin)
            x_te = -0.75 * c_b[i]
            cf_c[i] = (x_hinge - x_te) / c_b[i]
            theta_f = np.arccos(2.0 * cf_c[i] - 1.0)
            eps_fi[i] = 1.0 - (theta_f - np.sin(theta_f)) / np.pi
    return cf_c, eps_fi

# read aileron from json
def read_aileron(filename="input.json"):
    with open(filename, "r") as f:
        data = json.load(f)

    aileron = data["wing"]["aileron"]
    z_begin = aileron["begin[z/b]"]
    z_end = aileron["end[z/b]"]
    cf_begin = aileron["begin[cf/c]"]
    cf_end = aileron["end[cf/c]"]
    hinge_efficiency = aileron["hinge_efficiency"]

    return z_begin, z_end, cf_begin, cf_end, hinge_efficiency

# compute aileron distribution at each node
def compute_aileron_distribution(z_b, eps_fi, hinge_efficiency):
    deflection_efficiency = 1.0        
    sign = np.where(z_b < 0.0, 1.0, -1.0)
    chi = sign * hinge_efficiency * deflection_efficiency * eps_fi
    return chi

# compute rolling results with aileron and roll
def compute_roll_coefficients(a, b, c, d, aspect_ratio, alpha_root_deg, Omega, delta_a_deg, pbar):
    N = len(a)
    n = np.arange(1, N + 1)
    alpha_root = np.radians(alpha_root_deg)
    delta_a = np.radians(delta_a_deg)

    Cl_da = -np.pi * aspect_ratio / 4.0 * c[1]
    Cl_p = -np.pi * aspect_ratio / 4.0 * d[1]

    if pbar == "steady":
        pbar = -Cl_da * delta_a / Cl_p

    A = a * alpha_root - b * Omega + c * delta_a + d * pbar

    CL = np.pi * aspect_ratio * A[0]
    Cl = -np.pi * aspect_ratio / 4.0 * A[1]
    Cn = (np.pi * aspect_ratio / 4.0 * np.sum((2 * n[1:] - 1) * A[:-1] * A[1:])
          - np.pi * aspect_ratio * pbar / 8.0 * (A[0] + A[2]))
    CDi = np.pi * aspect_ratio * np.sum(n * A**2) - np.pi * aspect_ratio * pbar / 2.0 * A[1]

    return Cl_da, Cl_p, pbar, CL, Cl, Cn, CDi

# read inputs
planform_type, aspect_ratio, taper_ratio, planform_file = read_planform()
airfoil_lift_slope, nodes_per_semispan = read_wing_properties()
alpha_root_deg, delta_a_deg, pbar, CL_operating = read_condition()
show_planform, show_washout, show_aileron = read_view()
distribution, amount, CL_design = read_washout()
z_begin, z_end, cf_begin, cf_end, hinge_efficiency = read_aileron()

# chord distribution from file
chord_data = None
if planform_type == "file":
    z_file, c_file, aspect_ratio = read_chord_file(planform_file)
    chord_data = (z_file, c_file)

print("Planform type:", planform_type)
print("Aspect ratio:", aspect_ratio)
print("Taper ratio:", taper_ratio)
print("Nodes per semispan:", nodes_per_semispan)
print("Washout distribution:", distribution)
print("Washout amount [deg]:", amount)
print("Alpha root [deg]:", alpha_root_deg)
print("Aileron deflection [deg]:", delta_a_deg)
print("pbar:", pbar)

# nodes and chord
N, theta = compute_theta(nodes_per_semispan)
z_b = -0.5 * np.cos(theta)
c_b = compute_chord(z_b, planform_type, aspect_ratio, taper_ratio, chord_data)

C = compute_C_matrix(N, theta, c_b, airfoil_lift_slope)
C_inv = np.linalg.inv(C)
a = np.matmul(C_inv, np.ones(N))

omega = compute_washout(z_b, c_b, distribution, planform_type)
b = np.matmul(C_inv, omega)

cf_c, eps_fi = compute_flap_effectiveness(z_b, c_b, z_begin, z_end, cf_begin, cf_end, planform_type, aspect_ratio, taper_ratio, chord_data)
chi = compute_aileron_distribution(z_b, eps_fi, hinge_efficiency)
c = np.matmul(C_inv, chi)

d = np.matmul(C_inv, np.cos(theta))

# write C matrix, inverse, and Fourier coefficients to text file
with open("Solution.txt", "w") as file:
    file.write("C_matrix:\n")
    np.savetxt(file, C, fmt='%f', delimiter='\t')

    file.write("\nC_matrix_inverse:\n")
    np.savetxt(file, C_inv, fmt='%f', delimiter='\t')

    file.write("\nFourier Coefficients (a vals):\n")
    np.savetxt(file, a, fmt='%f', delimiter='\t')

    file.write("\nFourier Coefficients (b vals):\n")
    np.savetxt(file, b, fmt='%f', delimiter='\t')

    file.write("\nFourier Coefficients (c vals):\n")
    np.savetxt(file, c, fmt='%f', delimiter='\t')

    file.write("\nFourier Coefficients (d vals):\n")
    np.savetxt(file, d, fmt='%f', delimiter='\t')

kappa_L, kappa_D, e_s, CL_alpha, eps_omega, kappa_DL, kappa_DOmega, Omega, alpha_root_deg, CL, CDi = compute_coefficients(
    a, b, aspect_ratio, airfoil_lift_slope, alpha_root_deg, distribution, amount, CL_design, CL_operating)

Cl_da, Cl_p, pbar, CL, Cl, Cn, CDi = compute_roll_coefficients(
    a, b, c, d, aspect_ratio, alpha_root_deg, Omega, delta_a_deg, pbar)

print("\n     Kappa Results     ")
print("kappa_L:", kappa_L)
print("CL_alpha [1/rad]:", CL_alpha)
print("eps_Omega:", eps_omega)
print("kappa_D:", kappa_D)
print("kappa_DL:", kappa_DL)
print("kappa_DOmega:", kappa_DOmega)
print("Cl_da [1/rad]:", Cl_da)
print("Cl_pbar:", Cl_p)

print("\n     Operating Results     ")
print("Omega [deg]:", np.degrees(Omega))
print("Alpha root [deg]:", alpha_root_deg)  
print("CL:", CL)
print("CDi:", CDi)
print("Cl:", Cl)
print("Cn:", Cn)
print("pbar:", pbar)

# plots
if show_planform:
    plot_planform(z_b, c_b, z_begin, z_end, cf_begin, cf_end, planform_type, aspect_ratio, taper_ratio, chord_data)
if show_washout:
    plot_washout(z_b, omega)
if show_aileron:
    plot_aileron(z_b, chi)

















