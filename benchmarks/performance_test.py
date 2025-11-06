"""
Benchmarks de rendimiento para Kernel ML Engine.

Evalúa:
- Velocidad de cálculo de kernels
- Escalabilidad con tamaño de datos
- Comparación CPU vs GPU
- Uso de memoria
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import sys

from kernel.kernels.rbf import RBFKernel
from kernel.kernels.polynomial import PolynomialKernel
from kernel.kernels.linear import LinearKernel
from kernel.methods.svm import KernelSVM
from kernel.methods.kpca import KernelPCA


def benchmark_kernel_computation(kernel_class, sizes: List[int], 
                                 use_cache: bool = True, 
                                 enable_gpu: bool = False) -> Dict[str, List[float]]:
    """
    Benchmark de cálculo de kernels.
    
    Parámetros:
    -----------
    kernel_class : class
        Clase del kernel a evaluar.
    sizes : List[int]
        Tamaños de datasets a probar.
    use_cache : bool
        Usa caching o no.
    enable_gpu : bool
        Habilita GPU.
    
    Retorna:
    --------
    results : dict
        Tiempos de ejecución por tamaño.
    """
    results = {"sizes": sizes, "times": [], "throughput": []}
    
    print(f"\nBenchmark: {kernel_class.__name__}")
    print(f"  Caching: {use_cache}, GPU: {enable_gpu}")
    print("  Size      Time (s)    Throughput (samples²/s)")
    print("  " + "-" * 50)
    
    for size in sizes:
        # Genera datos
        X = np.random.randn(size, 10)
        
        # Crea kernel
        if kernel_class == RBFKernel:
            kernel = kernel_class(gamma=1.0, use_cache=use_cache, enable_gpu=enable_gpu)
        elif kernel_class == PolynomialKernel:
            kernel = kernel_class(degree=3, use_cache=use_cache, enable_gpu=enable_gpu)
        else:
            kernel = kernel_class(use_cache=use_cache, enable_gpu=enable_gpu)
        
        # Warm-up
        _ = kernel.gram_matrix(X[:min(100, size)])
        
        # Mide tiempo
        start = time.time()
        K = kernel.gram_matrix(X)
        elapsed = time.time() - start
        
        throughput = (size ** 2) / elapsed if elapsed > 0 else 0
        
        results["times"].append(elapsed)
        results["throughput"].append(throughput)
        
        print(f"  {size:6d}  {elapsed:10.4f}  {throughput:15.2e}")
    
    return results


def benchmark_svm_training(sizes: List[int]) -> Dict[str, List[float]]:
    """Benchmark de entrenamiento de SVM."""
    results = {"sizes": sizes, "times": []}
    
    print(f"\nBenchmark: KernelSVM Training")
    print("  Size      Time (s)")
    print("  " + "-" * 30)
    
    for size in sizes:
        # Genera datos
        X, y = np.random.randn(size, 5), np.random.choice([-1, 1], size)
        
        kernel = RBFKernel(gamma=1.0)
        svm = KernelSVM(kernel=kernel, C=1.0, max_iter=1000)
        
        start = time.time()
        svm.fit(X, y)
        elapsed = time.time() - start
        
        results["times"].append(elapsed)
        
        print(f"  {size:6d}  {elapsed:10.4f}")
    
    return results


def benchmark_memory_usage():
    """Evalúa uso de memoria."""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    
    sizes = [100, 500, 1000, 2000]
    results = {"sizes": sizes, "memory_mb": []}
    
    print(f"\nBenchmark: Memory Usage")
    print("  Size      Memory (MB)")
    print("  " + "-" * 30)
    
    for size in sizes:
        # Limpia memoria
        import gc
        gc.collect()
        
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        # Calcula kernel grande
        X = np.random.randn(size, 10)
        kernel = RBFKernel(gamma=1.0, use_cache=True)
        K = kernel.gram_matrix(X)
        
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_used = final_memory - initial_memory
        
        results["memory_mb"].append(memory_used)
        
        print(f"  {size:6d}  {memory_used:10.2f}")
        
        del K, X, kernel
        gc.collect()
    
    return results


def plot_benchmark_results(results_dict: Dict[str, Dict]):
    """Visualiza resultados de benchmarks."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Subplot 1: Tiempo de cálculo de kernels
    ax1 = axes[0, 0]
    for name, results in results_dict.items():
        if "kernel" in name.lower():
            ax1.loglog(results["sizes"], results["times"], marker='o', label=name)
    ax1.set_xlabel("Tamaño del Dataset")
    ax1.set_ylabel("Tiempo (s)")
    ax1.set_title("Tiempo de Cálculo de Kernels")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Throughput
    ax2 = axes[0, 1]
    for name, results in results_dict.items():
        if "kernel" in name.lower() and "throughput" in results:
            ax2.semilogx(results["sizes"], results["throughput"], marker='o', label=name)
    ax2.set_xlabel("Tamaño del Dataset")
    ax2.set_ylabel("Throughput (samples²/s)")
    ax2.set_title("Throughput de Kernels")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Entrenamiento SVM
    ax3 = axes[1, 0]
    if "svm" in results_dict:
        results = results_dict["svm"]
        ax3.loglog(results["sizes"], results["times"], marker='o', color='red')
    ax3.set_xlabel("Tamaño del Dataset")
    ax3.set_ylabel("Tiempo (s)")
    ax3.set_title("Tiempo de Entrenamiento SVM")
    ax3.grid(True, alpha=0.3)
    
    # Subplot 4: Uso de memoria
    ax4 = axes[1, 1]
    if "memory" in results_dict:
        results = results_dict["memory"]
        ax4.plot(results["sizes"], results["memory_mb"], marker='o', color='green')
    ax4.set_xlabel("Tamaño del Dataset")
    ax4.set_ylabel("Memoria (MB)")
    ax4.set_title("Uso de Memoria")
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('benchmarks.png', dpi=150)
    print("\nGráficos guardados en 'benchmarks.png'")


def run_all_benchmarks():
    """Ejecuta todos los benchmarks."""
    print("=" * 60)
    print("BENCHMARKS DE RENDIMIENTO - KERNEL ML ENGINE")
    print("=" * 60)
    
    sizes = [100, 200, 500, 1000, 2000]
    
    results = {}
    
    # Benchmarks de kernels
    try:
        results["RBF Kernel (CPU, Cache)"] = benchmark_kernel_computation(
            RBFKernel, sizes, use_cache=True, enable_gpu=False
        )
    except Exception as e:
        print(f"Error en benchmark RBF: {e}")
    
    try:
        results["RBF Kernel (CPU, No Cache)"] = benchmark_kernel_computation(
            RBFKernel, sizes, use_cache=False, enable_gpu=False
        )
    except Exception as e:
        print(f"Error en benchmark RBF sin cache: {e}")
    
    try:
        results["Polynomial Kernel"] = benchmark_kernel_computation(
            PolynomialKernel, sizes, use_cache=True, enable_gpu=False
        )
    except Exception as e:
        print(f"Error en benchmark Polynomial: {e}")
    
    try:
        results["Linear Kernel"] = benchmark_kernel_computation(
            LinearKernel, sizes, use_cache=True, enable_gpu=False
        )
    except Exception as e:
        print(f"Error en benchmark Linear: {e}")
    
    # Benchmark SVM
    try:
        svm_sizes = [50, 100, 200, 500]
        results["svm"] = benchmark_svm_training(svm_sizes)
    except Exception as e:
        print(f"Error en benchmark SVM: {e}")
    
    # Benchmark memoria
    try:
        results["memory"] = benchmark_memory_usage()
    except Exception as e:
        print(f"Error en benchmark memoria: {e}")
    
    # Visualiza
    try:
        plot_benchmark_results(results)
    except Exception as e:
        print(f"Error al generar gráficos: {e}")
    
    print("\n" + "=" * 60)
    print("Benchmarks completados")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    results = run_all_benchmarks()

