import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

def build_forward_model_2d(nx, nz, dx, dz, sigma_model, source_idx, sink_idx, I=1.0):
    """
    Construye y resuelve el forward model 2D A*phi = q.
    """
    N = nx * nz
    
    # Inicializar diagonales para la matriz rala A
    main_diag = np.zeros(N)
    off_diag_x = np.zeros(N-1)
    off_diag_z = np.zeros(N-nx)
    
    # Ensamblaje iterativo de la matriz A (Diferencias Finitas)
    for j in range(nz):
        for i in range(nx):
            n = j * nx + i
            sigma_c = sigma_model[j, i]
            
            # Coeficientes para X
            if i > 0: # Izquierda
                sigma_l = (sigma_c + sigma_model[j, i-1]) / 2.0
                cx = sigma_l / (dx**2)
                main_diag[n] += cx
                off_diag_x[n-1] = -cx
            if i < nx - 1: # Derecha
                sigma_r = (sigma_c + sigma_model[j, i+1]) / 2.0
                cx = sigma_r / (dx**2)
                main_diag[n] += cx
                
            # Coeficientes para Z
            if j > 0: # Arriba (Superficie, Neumann manejado implícitamente si no se resta)
                sigma_u = (sigma_c + sigma_model[j-1, i]) / 2.0
                cz = sigma_u / (dz**2)
                main_diag[n] += cz
                off_diag_z[n-nx] = -cz
            if j < nz - 1: # Abajo
                sigma_d = (sigma_c + sigma_model[j+1, i]) / 2.0
                cz = sigma_d / (dz**2)
                main_diag[n] += cz

    # Construir la matriz dispersa
    diagonals = [main_diag, off_diag_x, off_diag_x, off_diag_z, off_diag_z]
    offsets = [0, -1, 1, -nx, nx]
    A = sp.diags(diagonals, offsets, format='csr')

    # Vector fuente q (Dirac deltas)
    q = np.zeros(N)
    q[source_idx] = I   # r_t
    q[sink_idx] = -I    # r_s

    # Imponer condición de Dirichlet en el nodo del fondo para anclar la solución (referencia 0V)
    # y evitar que la matriz A sea singular (infinidad de soluciones agregando una constante)
    ref_node = N - 1
    A.data[A.indptr[ref_node]:A.indptr[ref_node+1]] = 0
    A[ref_node, ref_node] = 1.0
    q[ref_node] = 0.0

    # Resolver A * phi = q
    phi = spla.spsolve(A, q)
    
    return phi.reshape((nz, nx))

# --- Parámetros de prueba ---
nx, nz = 50, 30
dx, dz = 1.0, 1.0
sigma_bg = 0.01  # Conductividad base (100 Ohm.m)

# Crear un modelo de conductividad con una anomalía (geometría simple)
sigma_model = np.full((nz, nx), sigma_bg)
sigma_model[10:18, 20:30] = 0.1 # Anomalía conductiva (10 Ohm.m)

# Ubicación de electrodos (índices 1D: j * nx + i)
# Superficie (j=0), x=10 y x=40
source_idx = 0 * nx + 10
sink_idx = 0 * nx + 40

# Calcular potencial
phi_2d = build_forward_model_2d(nx, nz, dx, dz, sigma_model, source_idx, sink_idx)

# Visualización
plt.figure(figsize=(10, 4))
plt.contourf(phi_2d, levels=50, cmap='RdBu_r')
plt.colorbar(label=r'Potencial Eléctrico $\phi$ (V)')
plt.title('Distribución de Potencial - Forward Model 2D')
plt.gca().invert_yaxis() # Profundidad hacia abajo
plt.xlabel('X (Nodos)')
plt.ylabel('Z (Profundidad - Nodos)')
plt.savefig("resultado_potencial.png", dpi=300, bbox_inches='tight')