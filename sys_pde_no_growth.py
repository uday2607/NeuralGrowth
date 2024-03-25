from numba import jit, float64, int8, int32, types
import numpy as np

@jit(float64(float64[:,:], int32, int32, float64), nopython=True, nogil=True, cache=True,
     error_model="numpy", fastmath=True)
def laplace(array_x, i, j, dx):
    
    #5 point stencil
    return (array_x[i+1, j] + array_x[i-1, j] + array_x[i, j+1] + array_x[i, j-1]
            - 4*array_x[i, j])/(dx**2)


@jit(float64(float64[:,:], float64, float64, float64[::1], float64[::1], int32, int32), 
     nopython=True, nogil=True, cache=True, error_model="numpy", fastmath=True)
def BMP_pde_rhs(array_b, array_s_ij, array_l_ij, sys_params, sol_params, i, j):

    D_b, d_b, a_b_s, a_b_l = sys_params[0], sys_params[5], sys_params[6], sys_params[7]
    dx, Lx, Ly = sol_params[1], np.int32(sol_params[2]), np.int32(sol_params[3])

    return (D_b*laplace(array_b, i, j, dx) + 
                (a_b_s*array_s_ij + a_b_l*array_l_ij) - d_b*array_b[i, j])
    
@jit(float64(float64[:,:], float64, float64[::1], float64[::1], int32, int32), 
     nopython=True, nogil=True, cache=True, error_model="numpy", fastmath=True)
def BMP_i_pde_rhs(array_i, array_s_ij, sys_params, sol_params, i, j):

    D_i, d_i  = sys_params[1], sys_params[3]
    dx, Lx, Ly = sol_params[1], np.int32(sol_params[2]), np.int32(sol_params[3])

    return (D_i*laplace(array_i, i, j, dx) + 
                array_s_ij - d_i*array_i[i, j])

@jit(float64(int32, int32, int32, int32, int32, float64, float64), 
     nopython=True, nogil=True, cache=True, error_model="numpy", fastmath=True)
def pSmad_delta_dist(i, j, Lx, Ly, R, A, sigma):

    dist_centre = np.int32(np.sqrt((i - Lx//2)**2 + (j - Ly//2)**2))
    return A*(np.exp(-(dist_centre - R)**2/(2*sigma**2)))
    
@jit(float64(float64, float64, float64, float64[::1], int32, int32, float64), 
     nopython=True, nogil=True, cache=True, error_model="numpy", fastmath=True)
def pSmad_pde_rhs(array_s_ij, array_b_ij, array_i_ij, sys_params, i, j, pSmad_delta_val):

    K_i, h, d_s  = sys_params[10], sys_params[13], sys_params[4]

    return ((array_b_ij**h/(array_b_ij**h + (array_i_ij/K_i)**h))*
            (1 + pSmad_delta_val) - d_s*array_s_ij)
    

@jit(float64(float64, float64, float64[::1], int32, int32), 
     nopython=True, nogil=True, cache=True, error_model="numpy", fastmath=True)
def Lmx1a_pde_rhs(array_l_ij, array_s_ij, sys_params, i, j):

    a_s_l, K_l, K_s, h, d_l  = sys_params[8], sys_params[11], sys_params[12], sys_params[13], sys_params[5]

    return ((a_s_l*array_s_ij + (array_l_ij**h/(K_l**h + array_l_ij**h))*
             (array_s_ij**h/(K_s**h + array_s_ij**h))) 
            - d_l*array_l_ij)

@jit(float64[:, :](float64[:, :], int32, int32), 
     nopython=True, nogil=True, cache=True, error_model="numpy", fastmath=True)
def impose_Neumann_boundary(array_x, Lx, Ly):

    # Update the ghost cell i -> 0 and Lx-1 and j -> 0 and Ly-1
    array_x[:, 0] = array_x[:, 2] # j-> 0 are ghost cells
    array_x[:, Ly-1] = array_x[:, Ly-3] # j-> Ly-1 are ghost cells
    array_x[0, :] = array_x[2, :] # j-> 0 are ghost cells
    array_x[Lx-1, :] = array_x[Lx-3, :] # j-> Ly-1 are ghost cells

    return array_x

@jit(types.Tuple((float64[:,:],float64[:,:],float64[:,:],float64[:,:]))(float64[:,:], 
    float64[:,:], float64[:,:], float64[:,:], float64[::1], float64[::1], int8[:,::1],
    float64[:,:]), 
    nopython=True, nogil=True, cache=True, error_model="numpy", fastmath=True)
def solve_sys_pde_lhs(array_b, array_i, array_s, array_l, sys_params, sol_params, in_circle_array,
                      pSmad_delta_values):

    # Update the conccentration rates of all the chemical species acc to their PDEs
    dt, Lx, Ly = sol_params[0], np.int32(sol_params[2]), np.int32(sol_params[3])

    # make copies of old data sets
    new_array_b, new_array_i, new_array_s, new_array_l = (array_b.copy(), array_i.copy(), array_s.copy(),
                                                          array_l.copy())

    # Iterate through all the non-ghost mesh cell and update them
    for i in range(1, Lx-1):
        for j in range(1, Ly-1):
            # check if the mesh cell is in the circle
            in_circle = in_circle_array[i, j]

            # Update the conc on each mesh cell
            if in_circle:
                # Update intra-cellular species too
                new_array_b[i, j] += BMP_pde_rhs(array_b, array_s[i, j], array_l[i, j], sys_params, sol_params,
                                             i, j)*dt
                new_array_i[i, j] += BMP_i_pde_rhs(array_i, array_s[i, j], sys_params, sol_params, i, j)*dt
                new_array_s[i, j] += pSmad_pde_rhs(array_s[i,j], array_b[i, j], array_i[i, j], sys_params, 
                                            i, j, pSmad_delta_values[i, j])*dt
                new_array_l[i, j] += Lmx1a_pde_rhs(array_l[i,j], array_s[i, j], sys_params, i, j)*dt
            else:
                # Update only the diffusing species conc
                new_array_b[i, j] += BMP_pde_rhs(array_b, 0.0, 0.0, sys_params, sol_params,
                                             i, j)*dt
                new_array_i[i, j] += BMP_i_pde_rhs(array_i, 0.0, sys_params, sol_params, i, j)*dt

    # Impose the Neumann (No flux) boundary conditions on the ghost cells
    new_array_b = impose_Neumann_boundary(new_array_b, Lx, Ly)
    new_array_i = impose_Neumann_boundary(new_array_i, Lx, Ly)

    return new_array_b , new_array_i, new_array_s, new_array_l

@jit(types.Tuple((float64[:,:,:],float64[:,:,:],float64[:,:,:],float64[:,:,:]))(float64[:,:], 
    float64[:,:], float64[:,:], float64[:,:], float64[::1], float64[::1]), 
    nopython=True, nogil=True, cache=True, error_model="numpy", fastmath=True)
def run_and_save_sys(array_b, array_i, array_s, array_l, sys_params, sol_params):
    """Pass the initial conditions and parameters -> Returns concs at specific time points"""
    
    T, delta_t = np.int32(sol_params[5]), np.int32(sol_params[6])
    dt, Lx, Ly, R = sol_params[0], np.int32(sol_params[2]), np.int32(sol_params[3]), np.int32(sol_params[4])

    # Create arrays to save the data 
    data_b_time = np.zeros((T//delta_t, Lx, Ly))
    data_i_time = np.zeros((T//delta_t, Lx, Ly))
    data_s_time = np.zeros((T//delta_t, Lx, Ly))
    data_l_time = np.zeros((T//delta_t, Lx, Ly))

    # NOTE: To reduce number of computations, compute the arrays before hand
    in_circle_array = np.zeros((Lx, Ly), dtype=np.int8)
    # pSmad delta function variables
    A, sigma = sys_params[14], sys_params[15]
    pSmad_delta_values = np.zeros((Lx, Ly), dtype=np.float64)
    for i in range(1, Lx-1):
        for j in range(1, Ly-1):
            # check if the mesh cell is in the circle
            in_circle_array[i, j] = 1 if ((i - Lx//2)**2 + (j - Ly//2)**2) <= R**2 else 0
            
            # Also calculate the pSmad delta function so we can reduce the number of computations
            pSmad_delta_values[i, j] = pSmad_delta_dist(i, j, Lx, Ly, R, A, sigma)

    save_snap_time = 0
    for i in range(T):

        # save the concs data
        if i % delta_t == 0:
            data_b_time[save_snap_time][:, :] = array_b
            data_i_time[save_snap_time][:, :] = array_i
            data_s_time[save_snap_time][:, :] = array_s
            data_l_time[save_snap_time][:, :] = array_l

            save_snap_time += 1

        array_b, array_i, array_s, array_l = solve_sys_pde_lhs(array_b, array_i, array_s, 
                                                                array_l, sys_params, sol_params, 
                                                                in_circle_array, pSmad_delta_values)
        
    return data_b_time, data_i_time, data_s_time, data_l_time

@jit(types.Tuple((float64[:,:],float64[:,:],float64[:,:],float64[:,:]))(float64[::1], float64[::1]), 
     nopython=True, nogil=True, cache=True, error_model="numpy", fastmath=True)
def create_IC_1(sys_params, sol_params):

    Lx, Ly, R = np.int32(sol_params[2]), np.int32(sol_params[3]), np.int32(sol_params[4])

    # BMP is initally only present in the circle (make it smooth for solving purposes)
    array_b = np.zeros((Lx, Ly))
    # Set the initial concentration
    for i in range(Lx):
        for j in range(Ly):
            dist = np.int32((i - Lx//2)**2 + (j - Ly//2)**2)
            if (dist <= R**2):
                array_b[i, j] = sys_params[9]
            #make it smooth
            elif (dist <= (1.1*R)**2):
                array_b[i, j] = sys_params[9]*np.exp(-(dist - R**2))

    # Everyother species is set to zero 
    array_i = np.zeros((Lx, Ly))
    array_s = np.zeros((Lx, Ly))
    array_l = np.zeros((Lx, Ly))

    return array_b, array_i, array_s, array_l


