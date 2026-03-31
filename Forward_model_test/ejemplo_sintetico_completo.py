import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. GENERACIÓN DE DATOS SINTÉTICOS (El "Forward" exacto)
# ==========================================
# Posiciones de inyección delta desde tu archivo original
x_plus, y_plus = 0.3, 1.0
x_minus, y_minus = 0.7, 1.0
I_source = 1.0

# ¡ESTA ES LA REALIDAD OCULTA QUE LA PINN NO CONOCE Y DEBE DESCUBRIR!
sigma_true_val = 1.0  

def analytical_solution(X):
    """Solución analítica exacta para el potencial en un semi-espacio homogéneo."""
    x = X[:, 0]
    y = X[:, 1]
    # Usamos 1e-8 para evitar división por cero o log(0) justo en el electrodo
    r_plus = torch.sqrt((x - x_plus)**2 + (y - y_plus)**2) + 1e-8
    r_minus = torch.sqrt((x - x_minus)**2 + (y - y_minus)**2) + 1e-8
    
    phi = (-I_source / (np.pi * sigma_true_val)) * torch.log(r_plus) + \
          (I_source / (np.pi * sigma_true_val)) * torch.log(r_minus)
    return phi.unsqueeze(1)

def generate_synthetic_measurements(n_electrodes=40, noise_level=0.01):
    """Extrae datos falsos empíricos en la superficie como un ensayo geofísico real."""
    # Desplegamos electrodos de medición en la superficie (y=1)
    x_obs = torch.linspace(0.05, 0.95, n_electrodes).view(-1, 1).to(device)
    y_obs = torch.ones_like(x_obs).to(device)
    X_obs = torch.cat([x_obs, y_obs], dim=1)
    
    # Simulación del voltaje perfecto en el subsuelo
    V_true = analytical_solution(X_obs)
    
    # Añadimos ruido gaussiano ("instrumental") para simular la cruda realidad del ensayo
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
    # Misma corrección de polaridad vista en tu código original
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
        # Red Principal: Estima el Potencial Eléctrico V(x,y)
        self.net_V = MLP(2, 1, layers=4, neurons=64)
        
        # Red Segundaria (El Inverso): Estima la Conductividad sigma(x,y) oculta
        # Nota: Usamos menos neuronas porque sigma suele ser mucho más suave geométricamente que el potencial
        self.net_sigma = MLP(2, 1, layers=3, neurons=32)
        # Softplus garantiza que sigma siempre devuelva valores estrictamente positivos > 0
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
        
        # Evaluar redes
        V = self.forward_V(x, y)
        sigma = self.forward_sigma(x, y)
        
        # 1. Gradiente del potencial: grad(V)
        V_x = torch.autograd.grad(V, x, grad_outputs=torch.ones_like(V), create_graph=True)[0]
        V_y = torch.autograd.grad(V, y, grad_outputs=torch.ones_like(V), create_graph=True)[0]
        
        # 2. Flujo / Densidad de Corriente: J = sigma * grad(V)
        Jx = sigma * V_x
        Jy = sigma * V_y
        
        # 3. Divergencia: div(J)
        Jx_x = torch.autograd.grad(Jx, x, grad_outputs=torch.ones_like(Jx), create_graph=True)[0]
        Jy_y = torch.autograd.grad(Jy, y, grad_outputs=torch.ones_like(Jy), create_graph=True)[0]
        
        div_term = Jx_x + Jy_y
        
        # 4. Fuente física
        X = torch.cat([x, y], dim=1)
        f_val = compute_source(X)
        
        # Retorna el residuo (divergencia + inyección = debe ser 0)
        return div_term + f_val

# ==========================================
# 4. ORQUESTACIÓN Y ENTRENAMIENTO
# ==========================================
def run_inverse_experiment():
    print(f"Buscando hardware: Entrenando en {device} \n")
    
    # 1. Generar los datos (Imitar el trabajo de campo)
    print("1. Simulando adquisición de datos sintéticos en la frontera con 1% de ruido ambiental...")
    x_obs, y_obs, V_obs = generate_synthetic_measurements(n_electrodes=40, noise_level=0.01)
    
    # 2. Instanciar el Solucionador
    model = InversePINN().to(device)
    # Adam suele funcionar excelente para arrancar PINNs
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Hiperparámetros matemáticos
    epochs = 5000
    n_colloc = 2500       # Puntos para chequear las leyes de la física en el subsuelo
    lambda_data = 100.0   # Peso crítico: Obligamos a la red a no ignorar las matemáticas experimentales (d_obs)
    
    print("\n2. Iniciando la inversión matemática oculta (Entrenamiento)...")
    loss_history = []
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # --- LOSS DATA (Se aplica estrictamente donde hay datos: x_obs, y_obs) ---
        V_pred_obs = model.forward_V(x_obs, y_obs)
        loss_data = torch.mean((V_pred_obs - V_obs)**2)
        
        # --- LOSS PHYSICS (Muestras de colocación al interior del dominio) ---
        x_c = torch.rand(n_colloc, 1, device=device)
        y_c = torch.rand(n_colloc, 1, device=device)
        
        residual = model.pde_residual(x_c, y_c)
        loss_pde = torch.mean(residual**2)
        
        # L = λ*MSE_data + MSE_fisica 
        loss = lambda_data * loss_data + loss_pde
        
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        
        if epoch % 500 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:04d} | L_Total: {loss.item():.4e} | L_Data: {loss_data.item():.4e} | L_PDE: {loss_pde.item():.4e}")

    # ==========================
    # 5. VISUALIZACIÓN GRÁFICA GEOFÍSICA
    # ==========================
    print("\n3. Visualización de los descubrimientos generada.")
    model.eval()
    
    # Evaluar malla fina para visualización
    x_dim = np.linspace(0, 1, 100)
    y_dim = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x_dim, y_dim)
    
    x_tensor = torch.tensor(X.flatten(), dtype=torch.float32).view(-1, 1).to(device)
    y_tensor = torch.tensor(Y.flatten(), dtype=torch.float32).view(-1, 1).to(device)
    
    with torch.no_grad():
        sigma_pred = model.forward_sigma(x_tensor, y_tensor).cpu().numpy().reshape(100, 100)
        V_pred_map = model.forward_V(x_tensor, y_tensor).cpu().numpy().reshape(100, 100)
    
    fig, axs = plt.subplots(1, 4, figsize=(22, 5))
    
    # Gráfico 0: Curva del optimizador
    axs[0].plot(loss_history, color='navy')
    axs[0].set_yscale('log')
    axs[0].set_title("Evolución del Error")
    axs[0].set_xlabel("Epoch")
    
    # Gráfico 1: Trazado 2D del Potencial recuperado
    im1 = axs[1].contourf(X, Y, V_pred_map, 50, cmap='viridis')
    axs[1].scatter([x_plus, x_minus], [y_plus, y_minus], c='red', edgecolor='k', label="Electrodos (A, B)")
    axs[1].set_title("Potencial V(x,y) Recuperado")
    fig.colorbar(im1, ax=axs[1])
    axs[1].legend(loc="lower left")
    
    # Gráfico 2: Trazado 2D de la Propiedad Física "descubierta"
    im2 = axs[2].contourf(X, Y, sigma_pred, 50, cmap='inferno')
    # Añadir lineas de contorno para mejor lectura
    axs[2].contour(X, Y, sigma_pred, levels=[0.9, 0.95, 1.05, 1.1], colors='black', alpha=0.3)
    axs[2].set_title(f"Conductividad Estudiada $\sigma_{{pred}}$\n(Debería ser homogénea ~ {sigma_true_val})")
    fig.colorbar(im2, ax=axs[2])
    
    # Gráfico 3: Data vs Predicción en los Electrodos Superficiales
    V_obs_np = V_obs.cpu().numpy()
    x_obs_np = x_obs.cpu().numpy()
    axs[3].scatter(x_obs_np, V_obs_np, c='red', s=15, label="Datos de Campo (Ruidosos)", zorder=5)
    
    V_superficie = V_pred_map[-1, :] # El voltaje deducido por la red en la cima (y=1)
    axs[3].plot(x_dim, V_superficie, 'b-', label="Voltaje superficial PINN", zorder=4)
    axs[3].set_title("Ajuste Inverso en la Superficie")
    axs[3].legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_inverse_experiment()
