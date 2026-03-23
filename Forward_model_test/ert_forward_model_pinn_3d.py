import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Running on:", device)

# ============================================================
# DOMAIN DEFINITION
# ============================================================

Nx, Ny, Nz = 30, 30, 30

x = np.linspace(-1,1,Nx)
y = np.linspace(-1,1,Ny)
z = np.linspace(-1,1,Nz)

X,Y,Z = np.meshgrid(x,y,z,indexing="ij")

XYZ = np.vstack([
    X.flatten(),
    Y.flatten(),
    Z.flatten()
]).T

xyz = torch.tensor(XYZ,dtype=torch.float32).to(device)

# ============================================================
# ELECTRODE POSITIONS
# ============================================================

source = np.array([0.3,0.0,0.0])
sink   = np.array([-0.3,0.0,0.0])

source = torch.tensor(source,dtype=torch.float32).to(device)
sink   = torch.tensor(sink,dtype=torch.float32).to(device)

# ============================================================
# GAUSSIAN CURRENT SOURCE
# ============================================================

def gaussian_source(xyz,center,eps=0.1):

    r2 = torch.sum((xyz-center)**2,dim=1)

    return torch.exp(-r2/(2*eps**2))


def current_distribution(xyz):

    s = gaussian_source(xyz,source)
    t = gaussian_source(xyz,sink)

    return s - t

# ============================================================
# PINN ARCHITECTURE
# ============================================================

class PINN(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(3,128),
            nn.Tanh(),

            nn.Linear(128,128),
            nn.Tanh(),

            nn.Linear(128,128),
            nn.Tanh(),

            nn.Linear(128,1)
        )

    def forward(self,x):

        return self.net(x)

model = PINN().to(device)

# ============================================================
# PDE RESIDUAL
# ============================================================

def pde_residual(model,xyz):

    xyz.requires_grad_(True)

    phi = model(xyz)

    grad_phi = torch.autograd.grad(
        phi,
        xyz,
        grad_outputs=torch.ones_like(phi),
        create_graph=True
    )[0]

    phi_x = grad_phi[:,0]
    phi_y = grad_phi[:,1]
    phi_z = grad_phi[:,2]

    phi_xx = torch.autograd.grad(
        phi_x,
        xyz,
        grad_outputs=torch.ones_like(phi_x),
        create_graph=True
    )[0][:,0]

    phi_yy = torch.autograd.grad(
        phi_y,
        xyz,
        grad_outputs=torch.ones_like(phi_y),
        create_graph=True
    )[0][:,1]

    phi_zz = torch.autograd.grad(
        phi_z,
        xyz,
        grad_outputs=torch.ones_like(phi_z),
        create_graph=True
    )[0][:,2]

    laplacian = phi_xx + phi_yy + phi_zz

    return laplacian

# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)

# ============================================================
# TRAINING
# ============================================================

epochs = 2000

for e in range(epochs):

    optimizer.zero_grad()

    laplacian = pde_residual(model,xyz)

    f = current_distribution(xyz)

    loss = torch.mean((laplacian - f)**2)

    loss.backward()

    optimizer.step()

    if e % 200 == 0:

        print(f"Epoch {e} | Loss {loss.item():.6f}")

# ============================================================
# EVALUATION
# ============================================================

with torch.no_grad():

    phi = model(xyz).cpu().numpy()

phi = phi.reshape(Nx,Ny,Nz)

# ============================================================
# VISUALIZATION
# ============================================================

mid = Nz//2

plt.figure(figsize=(6,5))

plt.imshow(phi[:,:,mid],origin="lower")

plt.colorbar()

plt.title("Potential Slice (z mid-plane)")

plt.show()


# ============================================================
# OTHER SLICES
# ============================================================

fig,ax = plt.subplots(1,3,figsize=(15,4))

ax[0].imshow(phi[:,:,Nz//2],origin="lower")
ax[0].set_title("XY slice")

ax[1].imshow(phi[:,Ny//2,:],origin="lower")
ax[1].set_title("XZ slice")

ax[2].imshow(phi[Nx//2,:,:],origin="lower")
ax[2].set_title("YZ slice")

plt.show()

