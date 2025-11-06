# Kernel - Motor Avanzado de Métodos de Kernel para Machine Learning e Ingeniería

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: Proprietary](https://img.shields.io/badge/license-Proprietary-red.svg)](LICENSE)

**Kernel** es una biblioteca de alto rendimiento y producción para métodos de kernel en machine learning e ingeniería, diseñada para ser vendida como producto SaaS o librería premium.

## 🎯 Características Principales

### ✅ Implementado y Listo para Producción

#### Métodos de Machine Learning Completos

- ✅ **Support Vector Machines (SVM)** con algoritmo SMO optimizado
- ✅ **Kernel Principal Component Analysis (KPCA)** con reducción de dimensionalidad
- ✅ **Gaussian Processes** para regresión con incertidumbre
- ✅ Optimización automática de hiperparámetros

#### Redes Neuronales Implementadas Matemáticamente desde Cero

- ✅ **Red Neuronal Multicapa** con forward/backward propagation completo
- ✅ **Backpropagation** implementado matemáticamente (regla de la cadena)
- ✅ **Funciones de Activación**:
  - Clásicas: Sigmoid, Tanh, ReLU, LeakyReLU, ELU, Softmax, Linear
  - Avanzadas: Softplus, Swish, HardSigmoid, HardSwish, GELU, PReLU, SELU, Mish
- ✅ **Funciones de Pérdida**: MSE, Cross-Entropy, Binary Cross-Entropy
- ✅ **Regularización**: L1, L2, Dropout
- ✅ **Inicialización**: Xavier, He, Random

#### Optimizadores Matemáticos (40+ Implementados)

- ✅ **Clásicos**: SGD, Momentum, RMSprop, Adam, AdaGrad, Nesterov
- ✅ **Adaptativos Avanzados**: AdamW, Nadam, RAdam, AdaBelief, AdaMax, Yogi
- ✅ **Especializados**: Lion, Ranger, RangerQH, Lamb, QHM
- ✅ **Variantes**: Adadelta, Rprop, SignSGD, Adafactor, NovoGrad
- ✅ **Híbridos**: Lookahead, AggMo, AdaMod, SMORMS3, AdaShift
- ✅ **Con Restricciones**: AdaBound, AMSBound
- ✅ **Otros**: Fromage, AddSign, PowerSign, ExtendedRprop

#### Algoritmos de Optimización Matemática Avanzada

- ✅ **Gradiente Descendente** con búsqueda de línea (Armijo, Wolfe)
- ✅ **Método de Newton** con cálculo de Hessiana
- ✅ **Gradiente Conjugado** para sistemas lineales
- ✅ **BFGS** (Quasi-Newton method)

#### Álgebra Lineal Computacional

- ✅ **Descomposiciones**: LU, QR, SVD, Cholesky
- ✅ **Solvers de Sistemas Lineales**: LU, QR, Cholesky
- ✅ **Autovalores y Autovectores**: Power Method, QR Algorithm
- ✅ **Pseudoinversa de Moore-Penrose**

#### Kernels Avanzados

- ✅ **RBF (Gaussian)** - Optimizado con estabilidad numérica
- ✅ **Polynomial** - Homogéneo e inhomogéneo
- ✅ **Linear** - Altamente optimizado con BLAS
- ✅ **Matern** - Con parámetro de suavidad ν (0.5, 1.5, 2.5, ∞)
- ✅ **Laplacian** - Robusto a outliers
- ✅ **Composite** - Suma, producto y combinaciones lineales
- ✅ **Scaled** - Transformaciones de escala
- ✅ **Custom** - Kernels personalizados

#### Optimizaciones Avanzadas

- ✅ **Sistema de Caching LRU** - Caché inteligente con hash de datos
- ✅ **Descomposición de Cholesky** - Para sistemas lineales eficientes
- ✅ **Eigendecomposition** - Para análisis espectral
- ✅ **Soporte GPU** - CuPy para aceleración CUDA/OpenCL
- ✅ **Estabilidad Numérica** - Manejo robusto de casos edge
- ✅ **Validación Matemática** - Verificación de propiedades PSD

#### API REST Completa

- ✅ **FastAPI** - API REST moderna y rápida
- ✅ **Endpoints para Kernels** - Cálculo de matrices de kernel
- ✅ **Endpoints para SVM** - Entrenamiento y predicción
- ✅ **Endpoints para KPCA** - Transformación de datos
- ✅ **Endpoints para GP** - Regresión con incertidumbre
- ✅ **Gestión de Modelos** - Almacenamiento y recuperación

## 🚀 Inicio Rápido

### Instalación

```bash
# Clonar repositorio
git clone https://github.com/kernel-ml/kernel.git
cd kernel

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### Uso Básico

```python
from kernel.kernels.rbf import RBFKernel
from kernel.methods.svm import KernelSVM
import numpy as np

# Crea kernel RBF
kernel = RBFKernel(gamma=1.0, use_cache=True)

# Calcula matriz de kernel
X = np.random.randn(100, 10)
K = kernel.gram_matrix(X)

# Entrena SVM
svm = KernelSVM(kernel=kernel, C=1.0)
X_train, y_train = np.random.randn(50, 2), np.random.choice([-1, 1], 50)
svm.fit(X_train, y_train)

# Predice
X_test = np.random.randn(20, 2)
predictions = svm.predict(X_test)
```

### API REST

```bash
# Iniciar servidor
cd api
python main.py

# O con uvicorn
uvicorn api.main:app --reload --port 8000
```

**Documentación interactiva**: http://localhost:8000/docs

**Ejemplo de uso de API**:

```bash
# Calcular kernel
curl -X POST "http://localhost:8000/kernels/compute" \
  -H "Content-Type: application/json" \
  -d '{
    "X": [[1.0, 2.0], [3.0, 4.0]],
    "kernel_config": {
      "type": "rbf",
      "params": {"gamma": 1.0}
    }
  }'
```

## 📚 Ejemplos Avanzados

### Ejemplo 1: SVM con Kernel RBF

```python
from kernel.kernels.rbf import RBFKernel
from kernel.methods.svm import KernelSVM
from sklearn.datasets import make_circles

# Datos no linealmente separables
X, y = make_circles(n_samples=200, noise=0.1, factor=0.5)
y = np.where(y == 0, -1, 1)

# Entrena SVM
kernel = RBFKernel(gamma=1.0)
svm = KernelSVM(kernel=kernel, C=1.0, tol=1e-3)
svm.fit(X, y)

# Predice
predictions = svm.predict(X)
accuracy = svm.score(X, y)
print(f"Accuracy: {accuracy:.4f}")
```

### Ejemplo 2: Reducción de Dimensionalidad con KPCA

```python
from kernel.methods.kpca import KernelPCA
from kernel.kernels.rbf import RBFKernel

# Datos de alta dimensionalidad
X = np.random.randn(100, 50)

# Aplica KPCA
kernel = RBFKernel(gamma=0.1)
kpca = KernelPCA(kernel=kernel, n_components=2, center_kernel=True)
X_reduced = kpca.fit_transform(X)

# Varianza explicada
explained_var = kpca.explained_variance_ratio_()
print(f"Varianza explicada: {explained_var}")
```

### Ejemplo 3: Regresión con Gaussian Process

```python
from kernel.methods.gaussian_process import GaussianProcess
from kernel.kernels.rbf import RBFKernel

# Datos de entrenamiento
X_train = np.linspace(0, 10, 50).reshape(-1, 1)
y_train = np.sin(X_train.ravel()) + np.random.randn(50) * 0.1

# Entrena GP
kernel = RBFKernel(gamma=1.0)
gp = GaussianProcess(kernel=kernel, alpha=0.1, normalize_y=True)
gp.fit(X_train, y_train)

# Predice con incertidumbre
X_test = np.linspace(0, 10, 100).reshape(-1, 1)
y_mean, y_std = gp.predict(X_test, return_std=True)
```

### Ejemplo 4: Kernels Compuestos

```python
from kernel.kernels.rbf import RBFKernel
from kernel.kernels.polynomial import PolynomialKernel
from kernel.kernels.composite import CompositeKernel, ScaledKernel

# Kernel compuesto: RBF + Polynomial
rbf = RBFKernel(gamma=1.0)
poly = PolynomialKernel(degree=2, gamma=0.1)
composite = CompositeKernel([rbf, poly], operation="sum")

# Kernel escalado
scaled = ScaledKernel(rbf, scale=2.0)

# Usa en modelos
X = np.random.randn(100, 10)
K = composite.gram_matrix(X)
```

### Ejemplo 5: Red Neuronal desde Cero

```python
from kernel.math.neural_network import NeuralNetwork
from kernel.math.activations import ReLU, Sigmoid
from kernel.math.activations import BinaryCrossEntropy

# Crea red neuronal: 20 -> 64 -> 32 -> 1
nn = NeuralNetwork(
    layers=[20, 64, 32, 1],
    activations=[ReLU(), ReLU(), Sigmoid()],
    loss=BinaryCrossEntropy(),
    weight_init="he"
)

# Entrena
nn.fit(X_train, y_train, epochs=100, batch_size=32, learning_rate=0.01)

# Predice
predictions = nn.predict(X_test)
accuracy = nn.evaluate(X_test, y_test)["accuracy"]
```

### Ejemplo 6: Optimización Matemática

```python
from kernel.math.optimization import GradientDescent, NewtonMethod, BFGS

# Define función objetivo y gradiente
def f(x):
    return (x[0] - 1)**2 + (x[1] - 2)**2

def grad_f(x):
    return np.array([2*(x[0] - 1), 2*(x[1] - 2)])

# Optimiza con gradiente descendente
optimizer = GradientDescent(max_iter=1000, line_search="armijo")
result = optimizer.minimize(f, grad_f, x0=np.array([0.0, 0.0]))
print(f"Óptimo: {result['x']}, Valor: {result['fun']}")
```

### Ejemplo 7: Álgebra Lineal Computacional

```python
from kernel.math.linear_algebra import (
    MatrixDecomposition, EigenvalueSolver, LinearSystemSolver
)

# Descomposición LU
A = np.random.randn(5, 5)
L, U, P = MatrixDecomposition.lu_decomposition(A)

# Resuelve sistema lineal
b = np.random.randn(5)
x = LinearSystemSolver.solve_lu(A, b)

# Autovalores con Power Method
eigenvalue, eigenvector = EigenvalueSolver.power_method(A)
```

## 🎓 Fundamentos Matemáticos

### Teoría de Kernels

Un **kernel** K(x, y) es una función que mide la similitud entre dos vectores en un espacio de características de alta dimensión. Un kernel válido debe ser:

1. **Simétrico**: K(x, y) = K(y, x)
2. **Positivo Semidefinido (PSD)**: Para cualquier conjunto de puntos, la matriz de Gram es PSD

### Reproducing Kernel Hilbert Spaces (RKHS)

Cada kernel válido define un espacio de Hilbert de funciones donde el kernel actúa como producto interno:

```
<f, K(·, x)> = f(x)  (Propiedad de reproducción)
```

### Teorema de Mercer

Si K es un kernel válido, existe un mapeo φ: X → H tal que:

```
K(x, y) = <φ(x), φ(y)>_H
```

donde H es un espacio de Hilbert.

## 🛠️ Arquitectura Técnica

### Optimizaciones Implementadas

1. **Caching LRU**: Sistema de caché con hash SHA256 de datos y parámetros
2. **Cholesky Decomposition**: Para resolver sistemas lineales O(n³) → O(n²)
3. **Eigendecomposition**: Para análisis espectral y KPCA
4. **GPU Acceleration**: Soporte opcional con CuPy
5. **Numerical Stability**: Manejo de casos edge y underflow

### Algoritmos

- **SMO (Sequential Minimal Optimization)**: Para SVM, más eficiente que QP general
- **Kernel Centering**: Para KPCA, centra la matriz de kernel
- **Marginal Likelihood**: Para optimización de hiperparámetros en GP

## 📊 Benchmarks

Ejecuta benchmarks de rendimiento:

```bash
python benchmarks/performance_test.py
```

Evalúa:

- Velocidad de cálculo de kernels
- Escalabilidad con tamaño de datos
- Comparación CPU vs GPU
- Uso de memoria

## 🧪 Testing

```bash
# Ejecutar todos los tests
pytest tests/

# Con cobertura
pytest tests/ --cov=kernel --cov-report=html

# Tests específicos
pytest tests/test_kernels.py -v
```

## 📖 Documentación

### Estructura del Proyecto

```
Kernel/
├── kernel/                  # Código principal
│   ├── core/               # Núcleo matemático avanzado
│   │   └── kernel_base.py  # Clase base con optimizaciones
│   ├── kernels/            # Implementación de kernels
│   │   ├── rbf.py          # RBF optimizado
│   │   ├── polynomial.py   # Polinomial
│   │   ├── linear.py       # Lineal
│   │   ├── matern.py        # Matern
│   │   ├── laplacian.py    # Laplaciano
│   │   └── composite.py    # Kernels compuestos
│   ├── methods/            # Métodos de ML
│   │   ├── svm.py          # SVM con SMO
│   │   ├── kpca.py         # Kernel PCA
│   │   └── gaussian_process.py  # Gaussian Process
│   └── math/               # Algoritmos matemáticos avanzados
│       ├── activations.py  # Funciones de activación
│       ├── neural_network.py  # Redes neuronales desde cero
│       ├── optimizers.py   # Optimizadores (SGD, Adam, etc.)
│       ├── optimization.py # Optimización matemática
│       └── linear_algebra.py  # Álgebra lineal computacional
├── api/                    # API REST
│   └── main.py             # FastAPI application
├── tests/                  # Tests unitarios
├── examples/               # Ejemplos de uso
│   ├── advanced_usage.py   # Ejemplos avanzados
│   └── neural_network_example.py  # Ejemplos de redes neuronales
└── benchmarks/            # Benchmarks
    └── performance_test.py
```
