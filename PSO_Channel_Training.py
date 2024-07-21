import numpy as np
import scipy
import random
import tqdm
import matplotlib.pyplot as plt

RIS_ref_pos     = np.array([0, 1, 1])
BS_ref_pos      = np.array([2, 0, 0])
n_RIS_elements  = 128   # ULA
n_BS_elements   = 64    # ULA
particle_num    = 8
max_spd         = 1
pos_range       = 16
iter_qt         = 35
c1              = 1.5     # Social coefficient
c2              = 1   # Cognitive coefficient
w               = 0.7   # Inertial coefficient

fc              = 30e9
wavelength_fc   = scipy.constants.c/fc
dB              = wavelength_fc/2
dR              = wavelength_fc/2

#np.linspace(particles_pos[i], particles_pos[i]+dB*n_BS_elements, n_BS_elements)

particles_pos_x = np.random.rand(1,particle_num)[0]*pos_range  # Initial position
particles_spd_x = np.random.rand(1,particle_num)[0]*max_spd    # Initial speed

def distance_matrix(BS_base_pos):
    matrix = np.zeros((n_RIS_elements, n_BS_elements))
    for i in range(n_RIS_elements):
        for j in range(n_BS_elements):
            x_particle_antenna = BS_base_pos + j*dB
            dist = np.linalg.norm(np.array([x_particle_antenna, 0, 0]) - (RIS_ref_pos + np.array([0, i*dR, 0])))
            matrix[i,j] = dist
    return matrix

def h_function(d):
    return (wavelength_fc/(4*scipy.constants.pi*d))*np.exp(1j*-2*scipy.constants.pi*d/wavelength_fc)

h_function_vec = np.vectorize(h_function)

hBR = distance_matrix(BS_ref_pos[0])
hBR_matrix = h_function_vec(hBR)

pbest_array = np.zeros(particle_num)
fitness_values = []
pos_values = np.zeros((iter_qt,particle_num))
best_fitness_particles = np.full(particle_num, -np.inf)
gbest_particle_index = 0
gbest = -np.inf

for iter in tqdm.tqdm(range(iter_qt)):
    for i in range(particle_num):
        h_matrix = h_function_vec(distance_matrix(particles_pos_x[i]))
        c_val = np.random.randint(0, n_BS_elements)
        ps_activation = np.zeros(n_BS_elements)
        ps_activation[c_val] = 1
        fitness_function = np.linalg.norm(h_matrix @ ps_activation) - np.linalg.norm(hBR_matrix @ ps_activation)
        fitness_values.append(fitness_function)

        if fitness_function > best_fitness_particles[i]:
            pbest_array[i] = particles_pos_x[i]
            best_fitness_particles[i] = fitness_function

        if fitness_function > gbest:
            gbest = fitness_function
            gbest_particle_index = i
        
    #print(particles_pos_x, particles_spd_x, fitness_function)    
    pos_values[iter] = particles_pos_x
    particles_spd_x = np.clip(w*particles_spd_x + c1*(np.ones(len(particles_pos_x))*particles_pos_x[gbest_particle_index] - particles_pos_x) + c2*(pbest_array - particles_pos_x), -max_spd, max_spd)
    particles_pos_x = np.clip(particles_pos_x + particles_spd_x, 0, pos_range)
        
x_final = particles_pos_x[gbest_particle_index]
final_matrix = h_function_vec(distance_matrix(x_final))
eigmatrix = np.matmul(final_matrix.transpose().conjugate(), final_matrix)

vals, vects = np.linalg.eig(eigmatrix)
maxcol = np.argmax(vals)
eigenvect = vects[:,maxcol]
f = eigenvect/np.linalg.norm(eigenvect)

normalized_bf_gain = np.linalg.norm(np.matmul(final_matrix, f))
print("Normalized Beamforming Gain:", normalized_bf_gain)

xpoints = np.arange(0, iter_qt, 1)
plt.plot(xpoints,pos_values[:,0])
plt.plot(xpoints,pos_values[:,1])
plt.plot(xpoints,pos_values[:,2])
plt.plot(xpoints,pos_values[:,3])
plt.plot(xpoints,pos_values[:,4])
plt.plot(xpoints,pos_values[:,5])
plt.plot(xpoints,pos_values[:,6])
plt.plot(xpoints,pos_values[:,7])
plt.show()

# Plotting fitness values
ypoints = np.arange(0, iter_qt * particle_num, 1)
plt.plot(ypoints, fitness_values)
plt.xlabel('Iterations')
plt.ylabel('Fitness Value')
plt.title('Fitness Value over Iterations')
plt.show()