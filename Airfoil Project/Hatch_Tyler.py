import numpy as np # does math stuff    
import json # interfaces with .json input file 
import matplotlib.pyplot as plt # does plotting stuff

def load_json(json_filename):
    # extract json dictionary info
    json_string = open(json_filename).read() # read it as txt
    input_vals = json.loads(json_string) # load it as a json dictionary using the json package
    # now the info is easy to extract! 
    geometry = input_vals['geometry']  
    alpha_rad = np.deg2rad(input_vals["alpha[deg]"])
    vel_inf = input_vals["freestream_velocity"]
    return geometry, alpha_rad, vel_inf

def get_nodes_from_geom(geometry): 
    # this get's the nodes from the geometry txt file 
    with open(geometry, "r") as f:
        info = []
        for line in f:
            info.append([float(coordinate) for coordinate in line.split()])
    # now make the info list a numpy array 
    nodes = np.array(info)
    return nodes

def calc_panel_lengths(N_panels, nodes):
    lengths = np.zeros((N_panels,1)) # array of panel lengths
    for i in range(N_panels):
        lengths[i] = np.sqrt((nodes[i+1,0]-nodes[i,0]) ** 2 + (nodes[i+1,1]-nodes[i,1]) ** 2)
    return lengths

def calc_control_point_locations(N_panels, nodes):
    control_point_locations = np.zeros((N_panels,2)) # array of control points 

    for i in range(N_panels):
        # control point x values
        control_point_locations[i,0] = 0.5 * (nodes[i,0] + nodes[i+1,0])
        # control point y values
        control_point_locations[i,1] = 0.5 * (nodes[i,1] + nodes[i+1,1])

    return control_point_locations

def calc_P_matrix(nodes, lengths, j, x, y):
    x_j = nodes[j,0]
    y_j = nodes [j,1]
    x_jp1 = nodes[j+1, 0]
    y_jp1 = nodes[j+1, 1]
    l_j = lengths[j,0]

    dx = x_jp1 - x_j
    dy = y_jp1 - y_j

    xi = 1/l_j * (dx * (x - x_j) + dy * ( y- y_j))
    eta = 1/l_j * (-dy * (x - x_j) + dx * (y - y_j))

    phi = np.arctan2(eta * l_j, eta ** 2 + xi ** 2 - xi * l_j)
    psi = 0.5 * np.log((xi ** 2 + eta ** 2) / ((xi-l_j) ** 2 + eta ** 2))

    # left side of P matrix
    left = np.array([[dx , -dy],
                     [dy, dx]])
    # right side of P matrix eqn
    right = np.array([[(l_j - xi) * phi + eta * psi,          xi * phi - eta * psi],
                      [eta * phi - (l_j - xi) * psi - l_j,   -eta * phi - xi * psi + l_j]])

    P = (1 / (2 * np.pi * l_j ** 2)) * (left @ right)

    return P

def calc_A_matrix(N_nodes, N_panels, nodes, lengths, control_points):
    A = np.zeros((N_nodes, N_nodes))

    for i in range(N_panels):
        dx_i = nodes[i+1,0] - nodes[i,0]
        dy_i = nodes[i+1,1] - nodes[i,1]
        l_i = lengths[i,0]

        for j in range(N_panels):
            P = calc_P_matrix(nodes, lengths, j, control_points[i,0], control_points[i,1])
            A[i,j] += dx_i/l_i * P[1,0]  -  dy_i/l_i * P[0,0]
            A[i, j+1] += dx_i/l_i * P[1,1]  -  dy_i/l_i * P[0,1]

    A[N_nodes - 1, 0] = 1
    A[N_nodes - 1, N_nodes - 1] = 1

    return A

def calc_B_vector(N_nodes, N_panels, nodes, lengths, alpha_rad, vel_inf):
    B = np.zeros((N_nodes))

    for i in range(N_panels):
        dx_i = nodes[i+1,0] - nodes[i,0]
        dy_i = nodes[i+1,1] - nodes[i,1]
        l_i = lengths[i,0]

        B[i] = vel_inf * (dy_i * np.cos(alpha_rad) - dx_i * np.sin(alpha_rad)) / l_i

    return B

def calc_coefficients(N_panels, nodes, lengths, gamma, alpha_rad, vel_inf):
    CL = 0
    CmLE = 0

    for i in range(N_panels):
        x_i = nodes[i,0]
        y_i = nodes[i,1]
        x_ip1 = nodes[i+1,0]
        y_ip1 = nodes[i+1,1]
        l_i = lengths[i,0]
        c = 1

        CL += l_i / c * (gamma[i] + gamma[i+1]) / vel_inf

        # math for CmLE
        first_term = 2*x_i*gamma[i] + x_i*gamma[i+1] + x_ip1*gamma[i] + 2*x_ip1*gamma[i+1]
        second_term = 2*y_i*gamma[i] + y_i*gamma[i+1] + y_ip1*gamma[i] + 2*y_ip1*gamma[i+1]
        CmLE += l_i/vel_inf * (first_term*np.cos(alpha_rad) + second_term*np.sin(alpha_rad))

    CmLE = -1/3 * CmLE

    Cmc4 =  CmLE + 0.25 * CL * np.cos(alpha_rad)
        
    return CL, CmLE, Cmc4

if __name__ == "__main__": # Instructions after this line are what the code is actually following. 
    np.set_printoptions(precision=6, suppress=True, linewidth=150)
    json_filename = "input.json" 
    geometry, alpha_rad, vel_inf = load_json('input.json')
    print("\nPreliminary Information:")
    print("\ngeometry:\n", geometry)
    print("\nFreestream velocity:\n", vel_inf)
    print("\nalpha [deg]:\n", np.rad2deg(alpha_rad))
    nodes = get_nodes_from_geom(geometry)
    print("\nnodes:\n", nodes)
    N_nodes = len(nodes)
    print("\nN (nodes):\n",N_nodes)
    N_panels = N_nodes - 1 
    print("\nN (panels):\n",N_panels)

    # control points
    control_points = calc_control_point_locations(N_panels, nodes)
    print("\nControl Points:")
    print(control_points)

    # panel lengths
    print("\nPanel Length:")
    lengths = calc_panel_lengths(N_panels, nodes)
    print(lengths)

    # P matrix
    print("\n P Matrix:")
    P = calc_P_matrix(nodes, lengths, 1, control_points[1,0], control_points[1,1])
    print(P)

    # A matrix
    print("\n A Matrix:")
    A = calc_A_matrix(N_nodes, N_panels, nodes, lengths, control_points)
    print(A)

    # B vector
    print("\n B Vector:")
    B = calc_B_vector(N_nodes, N_panels, nodes, lengths, alpha_rad, vel_inf)
    print(B)

    # gamma vector
    print("\n Gamma Vector:")
    gamma = np.linalg.solve(A, B)
    print(gamma)

    # set variable A2 equal to true to test multiple alpha values and get a table instead of one set of coefficient outputs
    A2 = False
    if A2:
        # table stuff
        print("\nCoefficients for multiple alphas from", geometry)
        print("alpha [deg]       CL       CmLE       Cmc4")
        for alpha_deg in range(-12, 18, 2):
            alpha_sweep_rad = np.deg2rad(alpha_deg)

            # re calculate B vector, gamma vector, and coefficients for table
            B_sweep = calc_B_vector(N_nodes, N_panels, nodes, lengths, alpha_sweep_rad, vel_inf)
            gamma_sweep = np.linalg.solve(A, B_sweep)
            CL, CmLE, Cmc4 = calc_coefficients(N_panels, nodes, lengths, gamma_sweep, alpha_sweep_rad, vel_inf)

            # display results
            print(f"{alpha_deg:6d}      {CL:9.5f}  {CmLE:9.5f}  {Cmc4:9.5f}")
    else:
        # coefficents
        print("\nCoefficients:")
        coefficients = calc_coefficients(N_panels, nodes, lengths, gamma, alpha_rad, vel_inf)
        #CL
        print("\nCL:")
        print(round(coefficients[0],5))
        # CmLE
        print("\nCmLE:")
        print(round(coefficients[1],5))
        # Cmc4
        print("\nCmc4:")
        print(round(coefficients[2],5))