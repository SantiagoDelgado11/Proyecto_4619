# ==========================================
# PINN para ERT 2D con fuente tipo delta (Corregido)
# ==========================================

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# RED NEURONAL
# ==========================================

class PINN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.activation = nn.Tanh()
        self.layers = nn.ModuleList()

        for i in range(len(layers)-1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))

    def forward(self, x):
        for i in range(len(self.layers)-1):
            x = self.activation(self.layers[i](x))
        return self.layers[-1](x)

# ==========================================
# PARÁMETROS FÍSICOS
# ==========================================

# Electrodos EN LA SUPERFICIE (y = 1.0)
x_plus, y_plus = 0.3, 1.0
x_minus, y_minus = 0.7, 1.0

I = 1.0
eps = 0.02

# Conductividad
def sigma_fn(X):
    return X[:, 0:1] * 0.0 + 1.0

# ==========================================
# FUENTE (delta aproximada)
# ==========================================

def gaussian(x, y, x0, y0, eps):
    return torch.exp(-((x - x0)**2 + (y - y0)**2)/(2*eps**2)) / (2 * np.pi * eps**2)

def source(X):
    x = X[:, 0:1]
    y = X[:, 1:2]

    delta_plus = gaussian(x, y, x_plus, y_plus, eps)
    delta_minus = gaussian(x, y, x_minus, y_minus, eps)

    return I * (delta_plus - delta_minus)

# ==========================================
# PUNTOS DE ENTRENAMIENTO (Modificado para ERT)
# ==========================================

def sample_interior(n):
    x = torch.rand(n,1)
    y = torch.rand(n,1)
    return torch.cat([x,y], dim=1).to(device)

def sample_boundary(n):
    x = torch.rand(n,1)
    y = torch.rand(n,1)

    # 1. Fronteras Dirichlet (Fondo, Izquierda, Derecha -> phi = 0)
    xb_d = torch.cat([x, torch.zeros(n,1), torch.ones(n,1)], dim=0)
    yb_d = torch.cat([torch.zeros(n,1), y, y], dim=0)
    Xb_dirichlet = torch.cat([xb_d, yb_d], dim=1).to(device)

    # 2. Frontera Neumann (Superficie y=1 -> d(phi)/dy = 0)
    xb_n = x
    yb_n = torch.ones(n,1)
    Xb_neumann = torch.cat([xb_n, yb_n], dim=1).to(device)

    return Xb_dirichlet, Xb_neumann

# ==========================================
# LOSS PDE (Corrección de polaridad)
# ==========================================

def pde_loss(model, X):

    X.requires_grad_(True)
    phi = model(X)

    grad_phi = torch.autograd.grad(phi, X, grad_outputs=torch.ones_like(phi), create_graph=True)[0]
    phi_x = grad_phi[:, 0:1]
    phi_y = grad_phi[:, 1:2]

    phi_xx = torch.autograd.grad(phi_x, X, grad_outputs=torch.ones_like(phi_x), create_graph=True)[0][:, 0:1]
    phi_yy = torch.autograd.grad(phi_y, X, grad_outputs=torch.ones_like(phi_y), create_graph=True)[0][:, 1:2]

    sigma = sigma_fn(X)
    sigma_grad = torch.autograd.grad(sigma, X, grad_outputs=torch.ones_like(sigma), create_graph=True, allow_unused=True)[0]

    if sigma_grad is not None:
        sigma_x, sigma_y = sigma_grad[:, 0:1], sigma_grad[:, 1:2]
    else:
        sigma_x, sigma_y = torch.zeros_like(phi_x), torch.zeros_like(phi_y)

    div_term = sigma * (phi_xx + phi_yy) + sigma_x * phi_x + sigma_y * phi_y
    f = source(X)

    # CORRECCIÓN DE POLARIDAD: + f
    residual = div_term + f
    return torch.mean(residual**2)

# ==========================================
# LOSS DE FRONTERA (Mix Dirichlet/Neumann)
# ==========================================

def boundary_loss(model, Xb_d, Xb_n):
    # Loss Dirichlet (Lados y fondo = 0)
    phi_d = model(Xb_d)
    loss_dirichlet = torch.mean(phi_d**2)

    # Loss Neumann (Aislante en la superficie: d(phi)/dy = 0)
    Xb_n.requires_grad_(True)
    phi_n = model(Xb_n)
    grad_phi_n = torch.autograd.grad(phi_n, Xb_n, grad_outputs=torch.ones_like(phi_n), create_graph=True)[0]
    phi_y_n = grad_phi_n[:, 1:2] # Derivada normal (respecto a Y)
    loss_neumann = torch.mean(phi_y_n**2)

    return loss_dirichlet + loss_neumann

# ==========================================
# ENTRENAMIENTO
# ==========================================

model = PINN([2, 64, 64, 64, 1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 5000

for epoch in range(epochs):

    X = sample_interior(2000)
    Xb_dirichlet, Xb_neumann = sample_boundary(500)

    loss_pde_val = pde_loss(model, X)
    loss_bc_val = boundary_loss(model, Xb_dirichlet, Xb_neumann)

    # Damos un poco más de peso a la física y las fronteras
    loss = loss_pde_val + (10.0 * loss_bc_val)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch} | Loss Total: {loss.item():.6f}")

# ==========================================
# VISUALIZACIÓN, GROUND TRUTH Y DENSIDAD
# ==========================================

n = 100
x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)
X, Y = np.meshgrid(x, y)

XY = np.vstack([X.flatten(), Y.flatten()]).T
XY_t = torch.tensor(XY, dtype=torch.float32, device=device)

XY_t.requires_grad_(True)
phi_pred_t = model(XY_t)

grad_phi = torch.autograd.grad(phi_pred_t, XY_t, grad_outputs=torch.ones_like(phi_pred_t), create_graph=False)[0]
phi_pred = phi_pred_t.detach().cpu().numpy().reshape(n, n)

sigma_val = sigma_fn(XY_t)
J_t = -sigma_val * grad_phi
J_x = J_t[:, 0].detach().cpu().numpy().reshape(n, n)
J_y = J_t[:, 1].detach().cpu().numpy().reshape(n, n)

r_plus = np.sqrt((X - x_plus)**2 + (Y - y_plus)**2) + 1e-8
r_minus = np.sqrt((X - x_minus)**2 + (Y - y_minus)**2) + 1e-8

# NOTA: Al estar en la superficie (Medio espacio infinito), la corriente se duplica
# en comparación al espacio completo, por lo que la fórmula analítica cambia a "1 / pi"
sigma_cte = 1.0
phi_true = (-I / (np.pi * sigma_cte)) * np.log(r_plus) + (I / (np.pi * sigma_cte)) * np.log(r_minus)

error = np.abs(phi_true - phi_pred)

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

im0 = axs[0, 0].contourf(X, Y, phi_true, 50, cmap='viridis')
axs[0, 0].scatter([x_plus, x_minus], [y_plus, y_minus], c='red', edgecolor='k')
axs[0, 0].set_title("Ground Truth (Semi-espacio Infinito)")
fig.colorbar(im0, ax=axs[0, 0])

im1 = axs[0, 1].contourf(X, Y, phi_pred, 50, cmap='viridis')
axs[0, 1].scatter([x_plus, x_minus], [y_plus, y_minus], c='red', edgecolor='k')
axs[0, 1].set_title("Reconstrucción Potencial (PINN)")
fig.colorbar(im1, ax=axs[0, 1])

strm = axs[1, 1].streamplot(X, Y, J_x, J_y, color=np.sqrt(J_x**2 + J_y**2), cmap='autumn', linewidth=1)
axs[1, 1].scatter([x_plus, x_minus], [y_plus, y_minus], c='red', edgecolor='k')
axs[1, 1].set_title("Densidad de Corriente (Líneas de flujo)")
fig.colorbar(strm.lines, ax=axs[1, 1])

plt.tight_layout()
plt.show()