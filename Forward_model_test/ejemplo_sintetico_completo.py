import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import wandb

WANDB_API_KEY = "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" 

if WANDB_API_KEY:
    wandb.login(key=WANDB_API_KEY)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. GENERACIÓN DE DATOS SINTÉTICOS (El "Forward" exacto)
# ==========================================
x_plus, y_plus = 0.3, 1.0
x_minus, y_minus = 0.7, 1.0
I_source = 1.0

# ¡ESTA ES LA REALIDAD OCULTA QUE LA PINN NO CONOCE Y DEBE DESCUBRIR!
sigma_true_val = 1.0  

def analytical_solution(X):
    """Solución analítica exacta para el potencial en un semi-espacio homogéneo."""
    x = X[:, 0]
    y = X[:, 1]
    r_plus = torch.sqrt((x - x_plus)**2 + (y - y_plus)**2) + 1e-8
    r_minus = torch.sqrt((x - x_minus)**2 + (y - y_minus)**2) + 1e-8
    
    phi = (-I_source / (np.pi * sigma_true_val)) * torch.log(r_plus) + \
          (I_source / (np.pi * sigma_true_val)) * torch.log(r_minus)
    return phi.unsqueeze(1)

def generate_synthetic_measurements(n_electrodes=40, noise_level=0.01):
    """Extrae datos falsos empíricos en la superficie como un ensayo geofísico real."""
    x_obs = torch.linspace(0.05, 0.95, n_electrodes).view(-1, 1).to(device)
    y_obs = torch.ones_like(x_obs).to(device)
    X_obs = torch.cat([x_obs, y_obs], dim=1)
    
    V_true = analytical_solution(X_obs)
    noise = noise_level * torch.randn_like(V_true) * torch.std(V_true)
    V_noisy = V_true + noise
    
    return x_obs, y_obs, V_noisy

# ==========================================
# 2. DEFINICIÓN DEL TÉRMINO FUENTE f(r)
# ==========================================
eps = 0.02
def gaussian(x, y, x0, y0, eps):
    """Aproximación suave de la inyección de corriente."""
    return torch.exp(-((x - x0)**2 + (y - y0)**2)/(2*eps**2)) / (2 * np.pi * eps**2)

def compute_source(X):
    """Calcula f(r) requerido para el lado derecho de la ecuación diferencial."""
    x = X[:, 0:1]
    y = X[:, 1:2]
    delta_plus = gaussian(x, y, x_plus, y_plus, eps)
    delta_minus = gaussian(x, y, x_minus, y_minus, eps)
    return I_source * (delta_plus - delta_minus)

# ==========================================
# 3. RED NEURONAL INVERSA (Descubre la Física)
# ==========================================
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, layers, neurons):
        super(MLP, self).__init__()
        modules = [nn.Linear(in_dim, neurons), nn.Tanh()]
        for _ in range(layers - 1):
            modules += [nn.Linear(neurons, neurons), nn.Tanh()]
        modules += [nn.Linear(neurons, out_dim)]
        self.net = nn.Sequential(*modules)
        
    def forward(self, x):
        return self.net(x)

class InversePINN(nn.Module):
    def __init__(self):
        super(InversePINN, self).__init__()
        self.net_V = MLP(2, 1, layers=4, neurons=64)
        self.net_sigma = MLP(2, 1, layers=3, neurons=32)
        self.softplus = nn.Softplus() 
        
    def forward_V(self, x, y):
        r = torch.cat([x, y], dim=1)
        return self.net_V(r)
        
    def forward_sigma(self, x, y):
        r = torch.cat([x, y], dim=1)
        return self.softplus(self.net_sigma(r))
    
    def pde_residual(self, x, y):
        """Implementa las matemáticas: div(sigma * grad V) + f = 0"""
        x.requires_grad_(True)
        y.requires_grad_(True)
        
        V = self.forward_V(x, y)
        sigma = self.forward_sigma(x, y)
        
        V_x = torch.autograd.grad(V, x, grad_outputs=torch.ones_like(V), create_graph=True)[0]
        V_y = torch.autograd.grad(V, y, grad_outputs=torch.ones_like(V), create_graph=True)[0]
        
        Jx = sigma * V_x
        Jy = sigma * V_y
        
        Jx_x = torch.autograd.grad(Jx, x, grad_outputs=torch.ones_like(Jx), create_graph=True)[0]
        Jy_y = torch.autograd.grad(Jy, y, grad_outputs=torch.ones_like(Jy), create_graph=True)[0]
        
        div_term = Jx_x + Jy_y
        
        X = torch.cat([x, y], dim=1)
        f_val = compute_source(X)
        
        return div_term + f_val

# ==========================================
# 4. ORQUESTACIÓN Y ENTRENAMIENTO
# ==========================================
def run_inverse_experiment():
    print(f"Buscando hardware: Entrenando en {device} \n")
    
    # Hiperparámetros matemáticos ajustados
    epochs = 5000
    n_colloc = 10000      # 🔧 CAMBIO 1: Subimos de 2500 a 10000. Más "policías" matemáticos en el subsuelo.
    lambda_data = 50.0    # 🔧 CAMBIO 2: Bajamos de 100 a 50. Obligamos a la red a darle más peso a la física.
    lr_initial = 1e-3     # 🔧 CAMBIO 3: Iniciamos un poco más rápido...
    noise_level = 0.01
    n_electrodes = 40
    
    # Inicializar Weights & Biases
    wandb.init(
        project="PINN_ERT",
        name="Inversion_Homogenea_Mejorada",
        config={
            "epochs": epochs,
            "n_colloc": n_colloc,
            "lambda_data": lambda_data,
            "learning_rate_initial": lr_initial,
            "noise_level": noise_level,
            "n_electrodes": n_electrodes
        }
    )
    
    print(f"1. Simulando adquisición de datos sintéticos en la frontera con {noise_level*100}% de ruido...")
    x_obs, y_obs, V_obs = generate_synthetic_measurements(n_electrodes=n_electrodes, noise_level=noise_level)
    
    model = InversePINN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_initial)
    
    # 🔧 CAMBIO 4: Implementación del Scheduler. 
    # Multiplica el Learning Rate por 0.999 en cada época (va frenando suavemente).
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    
    print("\n2. Iniciando la inversión matemática oculta (Entrenamiento)...")
    loss_history = []
    
    lambda_tv = 10.0  # Hiperparámetro de la penalización Total Variation
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # --- LOSS DATA ---
        V_pred_obs = model.forward_V(x_obs, y_obs)
        loss_data = torch.mean((V_pred_obs - V_obs)**2)
        
        # --- LOSS PHYSICS & REGULARIZACIÓN TV ---
        x_c = torch.rand(n_colloc, 1, device=device)
        y_c = torch.rand(n_colloc, 1, device=device)
        
        # 1. Habilitamos los gradientes espaciales en los puntos de colocación
        x_c.requires_grad_(True)
        y_c.requires_grad_(True)
        
        # 2. Pérdida Física (Residuo de la Ecuación Diferencial)
        residual = model.pde_residual(x_c, y_c)
        loss_pde = torch.mean(residual**2)
        
        # 3. Regularización Total Variation (TV) en la red de Conductividad
        sigma_c = model.forward_sigma(x_c, y_c)
        
        # Derivadas espaciales de primer orden respecto a la conductividad (∇σ)
        dsigma_dx = torch.autograd.grad(sigma_c, x_c, grad_outputs=torch.ones_like(sigma_c), create_graph=True)[0]
        dsigma_dy = torch.autograd.grad(sigma_c, y_c, grad_outputs=torch.ones_like(sigma_c), create_graph=True)[0]
        
        # Penalizamos las oscilaciones abruptas o artificiales en la superficie
        loss_tv = torch.mean(dsigma_dx**2 + dsigma_dy**2)
        
        # --- LOSS TOTAL ---
        # Sumamos la penalidad de suavizado fuertemente castigada por lambda_tv
        loss = lambda_data * loss_data + loss_pde + lambda_tv * loss_tv
        
        loss.backward()
        optimizer.step()
        
        # 🔧 CAMBIO 5: Actualizamos el scheduler en cada paso
        scheduler.step()
        
        loss_history.append(loss.item())
        
        # Registrar métricas en wandb (agregando Loss_TV)
        wandb.log({
            "Loss_Total": loss.item(),
            "Loss_Data": loss_data.item(),
            "Loss_PDE": loss_pde.item(),
            "Loss_TV": loss_tv.item(),
            "Learning_Rate": scheduler.get_last_lr()[0],
            "epoch": epoch
        })
        
        if epoch % 500 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:04d} | L_Total: {loss.item():.4e} | L_Data: {loss_data.item():.4e} | L_PDE: {loss_pde.item():.4e} | L_TV: {loss_tv.item():.4e} | LR: {scheduler.get_last_lr()[0]:.2e}")

    # ==========================
    # 5. VISUALIZACIÓN GRÁFICA GEOFÍSICA
    # ==========================
    print("\n3. Visualización de los descubrimientos generada.")
    model.eval()
    
    x_dim = np.linspace(0, 1, 100)
    y_dim = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x_dim, y_dim)
    
    x_tensor = torch.tensor(X.flatten(), dtype=torch.float32).view(-1, 1).to(device)
    y_tensor = torch.tensor(Y.flatten(), dtype=torch.float32).view(-1, 1).to(device)
    
    with torch.no_grad():
        sigma_pred = model.forward_sigma(x_tensor, y_tensor).cpu().numpy().reshape(100, 100)
        V_pred_map = model.forward_V(x_tensor, y_tensor).cpu().numpy().reshape(100, 100)
    
    fig, axs = plt.subplots(1, 4, figsize=(22, 5))
    
    axs[0].plot(loss_history, color='navy')
    axs[0].set_yscale('log')
    axs[0].set_title("Evolución del Error")
    axs[0].set_xlabel("Epoch")
    
    im1 = axs[1].contourf(X, Y, V_pred_map, 50, cmap='viridis')
    axs[1].scatter([x_plus, x_minus], [y_plus, y_minus], c='red', edgecolor='k', label="Electrodos (A, B)")
    axs[1].set_title("Potencial V(x,y) Recuperado")
    fig.colorbar(im1, ax=axs[1])
    axs[1].legend(loc="lower left")
    
    im2 = axs[2].contourf(X, Y, sigma_pred, 50, cmap='inferno')
    axs[2].contour(X, Y, sigma_pred, levels=[0.9, 0.95, 1.05, 1.1], colors='black', alpha=0.3)
    
    # 🔧 CAMBIO 6: Agregué la letra 'r' antes de las comillas para arreglar el SyntaxWarning de Python
    axs[2].set_title(rf"Conductividad Estudiada $\sigma_{{pred}}$\n(Debería ser homogénea ~ {sigma_true_val})")
    fig.colorbar(im2, ax=axs[2])
    
    V_obs_np = V_obs.cpu().numpy()
    x_obs_np = x_obs.cpu().numpy()
    axs[3].scatter(x_obs_np, V_obs_np, c='red', s=15, label="Datos de Campo (Ruidosos)", zorder=5)
    
    V_superficie = V_pred_map[-1, :] 
    axs[3].plot(x_dim, V_superficie, 'b-', label="Voltaje superficial PINN", zorder=4)
    axs[3].set_title("Ajuste Inverso en la Superficie")
    axs[3].legend()
    
    plt.tight_layout()
    
    wandb.log({"Grafica_Resultados_Inversion": wandb.Image(fig)})
    wandb.finish()
    
    plt.savefig("resultado_pinn.png", dpi=300, bbox_inches='tight')
    print("Figura guardada como resultado_pinn.png")

if __name__ == "__main__":
    run_inverse_experiment()
