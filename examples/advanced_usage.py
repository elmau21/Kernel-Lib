"""
Ejemplos avanzados de uso de Kernel ML Engine.

Incluye:
- Entrenamiento de SVM con diferentes kernels
- Reducción de dimensionalidad con KPCA
- Regresión con Gaussian Processes
- Optimización de hiperparámetros
- Visualizaciones avanzadas
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression, make_circles
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

from kernel.kernels.rbf import RBFKernel
from kernel.kernels.polynomial import PolynomialKernel
from kernel.kernels.matern import MaternKernel
from kernel.kernels.composite import CompositeKernel
from kernel.methods.svm import KernelSVM
from kernel.methods.kpca import KernelPCA
from kernel.methods.gaussian_process import GaussianProcess


def example_svm_classification():
    """Ejemplo completo de clasificación con SVM."""
    print("=" * 60)
    print("Ejemplo: Clasificación con SVM")
    print("=" * 60)
    
    # Genera datos no linealmente separables
    X, y = make_circles(n_samples=200, noise=0.1, factor=0.5, random_state=42)
    y = np.where(y == 0, -1, 1)  # Convierte a -1/1
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Entrena SVM con kernel RBF
    kernel = RBFKernel(gamma=1.0)
    svm = KernelSVM(kernel=kernel, C=1.0, tol=1e-3, max_iter=1000, verbose=True)
    svm.fit(X_train, y_train)
    
    # Predice
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nPrecisión: {accuracy:.4f}")
    print(f"Número de vectores de soporte: {svm.n_support_}")
    
    # Visualiza
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Datos originales
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', alpha=0.6)
    plt.title("Datos Originales")
    plt.xlabel("x1")
    plt.ylabel("x2")
    
    # Subplot 2: Frontera de decisión
    plt.subplot(1, 2, 2)
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='RdYlBu', 
               edgecolors='black', s=50)
    if svm.support_vectors_ is not None:
        plt.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1],
                   s=100, facecolors='none', edgecolors='black', linewidths=2,
                   label='Vectores de Soporte')
    plt.title(f"Frontera de Decisión SVM (Accuracy: {accuracy:.3f})")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('svm_classification.png', dpi=150)
    print("\nVisualización guardada en 'svm_classification.png'")


def example_kpca_dimensionality_reduction():
    """Ejemplo de reducción de dimensionalidad con KPCA."""
    print("\n" + "=" * 60)
    print("Ejemplo: Reducción de Dimensionalidad con KPCA")
    print("=" * 60)
    
    # Genera datos en forma de espiral (no lineal)
    n_samples = 300
    t = np.linspace(0, 4 * np.pi, n_samples)
    X = np.zeros((n_samples, 3))
    X[:, 0] = t * np.cos(t)
    X[:, 1] = t * np.sin(t)
    X[:, 2] = np.random.randn(n_samples) * 0.1
    
    # Aplica KPCA
    kernel = RBFKernel(gamma=0.1)
    kpca = KernelPCA(kernel=kernel, n_components=2, center_kernel=True)
    X_kpca = kpca.fit_transform(X)
    
    explained_var = kpca.explained_variance_ratio_()
    
    print(f"Varianza explicada por componente:")
    for i, var in enumerate(explained_var):
        print(f"  Componente {i+1}: {var:.4f} ({var*100:.2f}%)")
    
    # Visualiza
    fig = plt.figure(figsize=(15, 5))
    
    # Subplot 1: Datos originales (3D)
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=t, cmap='viridis', alpha=0.6)
    ax1.set_title("Datos Originales (3D)")
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")
    ax1.set_zlabel("x3")
    
    # Subplot 2: KPCA (2D)
    ax2 = fig.add_subplot(1, 3, 2)
    scatter = ax2.scatter(X_kpca[:, 0], X_kpca[:, 1], c=t, cmap='viridis', alpha=0.6)
    ax2.set_title("KPCA (2 componentes)")
    ax2.set_xlabel("Primera Componente Principal")
    ax2.set_ylabel("Segunda Componente Principal")
    plt.colorbar(scatter, ax=ax2)
    
    # Subplot 3: Varianza explicada
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.bar(range(1, len(explained_var) + 1), explained_var * 100)
    ax3.set_title("Varianza Explicada")
    ax3.set_xlabel("Componente")
    ax3.set_ylabel("Varianza Explicada (%)")
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('kpca_reduction.png', dpi=150)
    print("\nVisualización guardada en 'kpca_reduction.png'")


def example_gaussian_process_regression():
    """Ejemplo de regresión con Gaussian Process."""
    print("\n" + "=" * 60)
    print("Ejemplo: Regresión con Gaussian Process")
    print("=" * 60)
    
    # Genera datos con ruido
    X = np.linspace(0, 10, 50).reshape(-1, 1)
    y_true = np.sin(X.ravel()) + 0.1 * X.ravel()
    y = y_true + np.random.randn(len(X)) * 0.1
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Entrena GP
    kernel = RBFKernel(gamma=1.0)
    gp = GaussianProcess(kernel=kernel, alpha=0.1, normalize_y=True)
    gp.fit(X_train, y_train)
    
    # Predice
    X_pred = np.linspace(0, 10, 100).reshape(-1, 1)
    y_mean, y_std = gp.predict(X_pred, return_std=True)
    
    # Calcula score
    y_pred_test = gp.predict(X_test, return_std=False)
    mse = mean_squared_error(y_test, y_pred_test)
    score = gp.score(X_test, y_test)
    
    print(f"\nMSE: {mse:.4f}")
    print(f"R² Score: {score:.4f}")
    
    # Visualiza
    plt.figure(figsize=(12, 6))
    
    # Datos de entrenamiento
    plt.scatter(X_train, y_train, c='blue', s=50, alpha=0.7, label='Datos Entrenamiento', zorder=3)
    plt.scatter(X_test, y_test, c='red', s=50, alpha=0.7, label='Datos Prueba', zorder=3)
    
    # Predicción con incertidumbre
    plt.plot(X_pred, y_mean, 'k-', linewidth=2, label='Media Predicha')
    plt.fill_between(X_pred.ravel(), 
                     y_mean - 2 * y_std, 
                     y_mean + 2 * y_std,
                     alpha=0.3, color='gray', label='Intervalo de Confianza (2σ)')
    
    # Función verdadera
    y_true_pred = np.sin(X_pred.ravel()) + 0.1 * X_pred.ravel()
    plt.plot(X_pred, y_true_pred, 'g--', linewidth=2, alpha=0.7, label='Función Verdadera')
    
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Regresión con Gaussian Process (R² = {score:.3f})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gp_regression.png', dpi=150)
    print("\nVisualización guardada en 'gp_regression.png'")


def example_composite_kernels():
    """Ejemplo de kernels compuestos."""
    print("\n" + "=" * 60)
    print("Ejemplo: Kernels Compuestos")
    print("=" * 60)
    
    X = np.random.randn(20, 2)
    
    # Kernel compuesto: RBF + Polynomial
    rbf = RBFKernel(gamma=1.0)
    poly = PolynomialKernel(degree=2, gamma=0.1)
    composite = CompositeKernel([rbf, poly], operation="sum")
    
    K_composite = composite.gram_matrix(X)
    
    print(f"Kernel compuesto (RBF + Polynomial):")
    print(f"  Shape: {K_composite.shape}")
    print(f"  ¿Es PSD? {composite.is_psd(X)}")
    
    # Kernel escalado
    from kernel.kernels.composite import ScaledKernel
    scaled = ScaledKernel(rbf, scale=2.0)
    K_scaled = scaled.gram_matrix(X)
    
    print(f"\nKernel escalado (2.0 * RBF):")
    print(f"  Ratio con RBF original: {np.mean(K_scaled / rbf.gram_matrix(X)):.4f}")


def example_hyperparameter_optimization():
    """Ejemplo de optimización de hiperparámetros."""
    print("\n" + "=" * 60)
    print("Ejemplo: Optimización de Hiperparámetros")
    print("=" * 60)
    
    # Genera datos
    X, y = make_classification(n_samples=100, n_features=2, n_redundant=0,
                              n_informative=2, n_clusters_per_class=1,
                              random_state=42)
    y = np.where(y == 0, -1, 1)
    
    # Grid search para gamma en RBF
    gamma_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    C_values = [0.1, 1.0, 10.0]
    
    best_score = -np.inf
    best_params = None
    
    print("Grid Search:")
    print("  gamma    C      Accuracy")
    print("  " + "-" * 30)
    
    for gamma in gamma_values:
        for C in C_values:
            kernel = RBFKernel(gamma=gamma)
            svm = KernelSVM(kernel=kernel, C=C, tol=1e-3, max_iter=500)
            svm.fit(X, y)
            
            # Score en datos de entrenamiento (en producción usar validación cruzada)
            score = svm.score(X, y)
            
            print(f"  {gamma:6.1f}  {C:6.1f}  {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_params = {"gamma": gamma, "C": C}
    
    print(f"\nMejores parámetros: {best_params}")
    print(f"Mejor score: {best_score:.4f}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("EJEMPLOS AVANZADOS - KERNEL ML ENGINE")
    print("=" * 60)
    
    # Ejecuta ejemplos
    try:
        example_svm_classification()
    except Exception as e:
        print(f"Error en ejemplo SVM: {e}")
    
    try:
        example_kpca_dimensionality_reduction()
    except Exception as e:
        print(f"Error en ejemplo KPCA: {e}")
    
    try:
        example_gaussian_process_regression()
    except Exception as e:
        print(f"Error en ejemplo GP: {e}")
    
    try:
        example_composite_kernels()
    except Exception as e:
        print(f"Error en ejemplo kernels compuestos: {e}")
    
    try:
        example_hyperparameter_optimization()
    except Exception as e:
        print(f"Error en ejemplo optimización: {e}")
    
    print("\n" + "=" * 60)
    print("Ejemplos completados")
    print("=" * 60)

