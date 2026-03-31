import torch
import torch.nn as nn
import numpy as np

class MLP(nn.Module):
    """
    Multilayer Perceptron genérico para aproximar funciones continuas.
    """
    def __init__(self, in_dim, out_dim, hidden_layers, hidden_dim, activation=nn.Tanh()):
        super(MLP, self).__init__()
        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(activation)
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation)
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

class PINN_ERT(nn.Module):
    """
    Physics-Informed Neural Network para Tomografía de Resistividad Eléctrica (TRE).
    Resuelve el problema inverso para estimar simultáneamente el potencial V(x,y)
    y la conductividad sigma(x,y).
    """
    def __init__(self, hidden_layers=4, hidden_dim=64):
        super(PINN_ERT, self).__init__()
        # 1. Red para el potencial V_theta(r)
        self.net_V = MLP(in_dim=2, out_dim=1, hidden_layers=hidden_layers, hidden_dim=hidden_dim)
        
        # 2. Red para la conductividad sigma_psi(r)
        self.net_sigma = MLP(in_dim=2, out_dim=1, hidden_layers=hidden_layers, hidden_dim=hidden_dim)
        
        # Función de activación final para la conductividad para garantizar que sea estrictamente positiva
        # Softplus(x) = ln(1 + e^x) > 0 para todo x.
        self.softplus = nn.Softplus()
        
    def forward_V(self, x, y):
        # r = (x, y)
        r = torch.cat([x, y], dim=1)
        return self.net_V(r)
        
    def forward_sigma(self, x, y):
        r = torch.cat([x, y], dim=1)
        # Aseguramos que sigma sea estrictamente positiva
        return self.softplus(self.net_sigma(r))
    
    def calculate_pde_residual(self, x, y, f):
        """
        Calcula el residuo de la EDP: div(sigma(r) * grad(V(r))) - f(r)
        usando diferenciación automática.
        """
        # Habilitar el cálculo de gradientes respecto a las coordenadas espaciales
        # (Esto es crítico para poder derivar respecto a x e y)
        x.requires_grad_(True)
        y.requires_grad_(True)
        
        # Calcular V_theta y sigma_psi en las coordenadas solicitadas
        V = self.forward_V(x, y)
        sigma = self.forward_sigma(x, y)
        
        # -- Cálculo de derivadas espaciales usando autograd --
        
        # Paso 1: grad(V) = (dV/dx, dV/dy)
        # create_graph=True permite que podamos calcular derivadas de orden superior después.
        dV_dx = torch.autograd.grad(
            V, x, 
            grad_outputs=torch.ones_like(V), 
            create_graph=True, 
            retain_graph=True
        )[0]
        
        dV_dy = torch.autograd.grad(
            V, y, 
            grad_outputs=torch.ones_like(V), 
            create_graph=True, 
            retain_graph=True
        )[0]
        
        # Paso 2: Multiplicar por sigma para obtener la corriente de densidad: J = sigma(x,y) * grad(V)
        Jx = sigma * dV_dx
        Jy = sigma * dV_dy
        
        # Paso 3: Calcular divergencia de J: div(J) = dJx/dx + dJy/dy
        dJx_dx = torch.autograd.grad(
            Jx, x, 
            grad_outputs=torch.ones_like(Jx), 
            create_graph=True, 
            retain_graph=True
        )[0]
        
        dJy_dy = torch.autograd.grad(
            Jy, y, 
            grad_outputs=torch.ones_like(Jy), 
            create_graph=True, 
            retain_graph=True
        )[0]
        
        # Divergencia del producto (lado izquierdo de la EDP)
        div_J = dJx_dx + dJy_dy
        
        # Paso 4: Residuo final -> div(J) - f(r)
        pde_residual = div_J - f
        
        return pde_residual

def train_pinn_ert():
    """
    Función principal de entrenamiento y orquestación del modelo.
    """
    # Configuraciones
    n_collocation = 5000 # Puntos de colocación en el dominio para satisfacer la física.
    n_obs = 100          # Número de puntos de observación (electrodos en TRE).
    epochs = 10000
    learning_rate = 1e-3
    lambda_pde = 1.0     # Peso relativo (hiperparámetro) de la pérdida física en la sumatoria total
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 1. Inicializar clase estructurada, optimizador Adam
    model = PINN_ERT(hidden_layers=4, hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 2. Inicializar puntos de colocación (collocation points) r_colloc para la EDP.
    # Uniformemente distribuidos en el dominio de simulación de 2D, ej: [-1, 1]x[-1, 1]
    x_colloc = (2.0 * torch.rand(n_collocation, 1) - 1.0).to(device)
    y_colloc = (2.0 * torch.rand(n_collocation, 1) - 1.0).to(device)
    
    # Definir el término fuente volumétrico f(r). 
    # En muchos problemas TRE, f(r) es equivalente a las fuentes puntuales de inyección de corriente
    # (podemos aproximarlas como distribuciones gaussianas estrechas o ceros donde no hay fuente).
    f_colloc = torch.zeros_like(x_colloc).to(device)
    
    # --- INTEGRACION DE OBSERVACIONES (d_obs) EN LA FRONTERA ---
    # Para TRE interactuamos principalmente con datos empíricos obtenidos en los electrodos
    # típicamente situados en la superficie (frontera z=0 o y=1 en nuestro sistema coordenado local).
    # d_obs corresponde a las lecturas de voltaje ante un patrón de inyección de corriente conocido.
    
    # Supongamos que generamos electrodos sobre la superficie superior del dominio (y = 1)
    x_obs = (2.0 * torch.rand(n_obs, 1) - 1.0).to(device)
    y_obs = torch.ones_like(x_obs).to(device)  # Superficie constante
    
    # En un escenario sin simulación, 'd_obs' se cargaría desde un archivo medido experimentalmente,
    # el cual puede consistir en tensores (n_obs, 1) de potencial eléctrico V empírico.
    # Aquí creamos un d_obs de ejemplo sintético (Voltajes medidos)
    d_obs = torch.sin(np.pi * x_obs).to(device) 
    
    # Main training loop
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        # IMPORTANTE: Aseguramos que requieren gradientes en cada iteración por si se regeneran batchs iterativamente
        x_colloc.requires_grad_(True)
        y_colloc.requires_grad_(True)
        
        # --- Cálculo de la Componente Experimental (Data Loss) ---
        # Calculamos la predicción de V usando nuestras coordenadas de las observaciones empíricas
        V_pred = model.forward_V(x_obs, y_obs)
        # Error Cuadrático Medio con relación a las observaciones reales
        loss_data = torch.mean((V_pred - d_obs) ** 2)
        
        # --- Cálculo de la Componente Física (PDE Loss) ---
        # Computamos el residuo evaluando la gobernante física en todo el dominio interior
        # Este componente restringe las posibles soluciones al espacio físico válido regido por V y sigma
        pde_residual = model.calculate_pde_residual(x_colloc, y_colloc, f_colloc)
        loss_pde = torch.mean(pde_residual ** 2)
        
        # --- Pérdida Total (Total Loss) ---
        # L(theta, psi) = loss_data + lambda * loss_pde
        loss = loss_data + lambda_pde * loss_pde
        
        # Retropropagación y optimización
        loss.backward()
        optimizer.step()
        
        if epoch % 1000 == 0 or epoch == 1:
            print(f'Epoch [{epoch}/{epochs}] \t '
                  f'Total Loss: {loss.item():.6e} \t '
                  f'Data Loss: {loss_data.item():.6e} \t '
                  f'PDE Loss: {loss_pde.item():.6e}')

if __name__ == "__main__":
    train_pinn_ert()
