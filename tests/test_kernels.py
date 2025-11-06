"""
Tests para los kernels implementados.
"""

import numpy as np
import pytest
from kernel.kernels.rbf import RBFKernel
from kernel.kernels.polynomial import PolynomialKernel
from kernel.kernels.linear import LinearKernel


class TestRBFKernel:
    """Tests para el kernel RBF."""
    
    def test_rbf_basic(self):
        """Test básico del kernel RBF."""
        X = np.random.randn(10, 3)
        rbf = RBFKernel(gamma=1.0)
        K = rbf.gram_matrix(X)
        
        assert K.shape == (10, 10)
        assert np.allclose(K, K.T)  # Simetría
        assert np.all(K >= 0)  # Valores no negativos
    
    def test_rbf_psd(self):
        """Test de que el kernel RBF es PSD."""
        X = np.random.randn(20, 5)
        rbf = RBFKernel(gamma=1.0)
        assert rbf.is_psd(X)
    
    def test_rbf_sigma_parameter(self):
        """Test del parámetro sigma."""
        rbf1 = RBFKernel(gamma=1.0)
        rbf2 = RBFKernel(sigma=np.sqrt(0.5))
        
        X = np.random.randn(5, 2)
        K1 = rbf1.gram_matrix(X)
        K2 = rbf2.gram_matrix(X)
        
        assert np.allclose(K1, K2)
    
    def test_rbf_diagonal(self):
        """Test de que la diagonal es 1 (K(x, x) = 1)."""
        X = np.random.randn(10, 3)
        rbf = RBFKernel(gamma=1.0)
        K = rbf.gram_matrix(X)
        
        assert np.allclose(np.diag(K), 1.0)


class TestPolynomialKernel:
    """Tests para el kernel polinomial."""
    
    def test_polynomial_basic(self):
        """Test básico del kernel polinomial."""
        X = np.random.randn(10, 3)
        poly = PolynomialKernel(degree=2, gamma=1.0, coef0=0.0)
        K = poly.gram_matrix(X)
        
        assert K.shape == (10, 10)
        assert np.allclose(K, K.T)  # Simetría
    
    def test_polynomial_psd(self):
        """Test de que el kernel polinomial es PSD."""
        X = np.random.randn(20, 5)
        poly = PolynomialKernel(degree=3)
        assert poly.is_psd(X)


class TestLinearKernel:
    """Tests para el kernel lineal."""
    
    def test_linear_basic(self):
        """Test básico del kernel lineal."""
        X = np.random.randn(10, 3)
        linear = LinearKernel()
        K = linear.gram_matrix(X)
        
        assert K.shape == (10, 10)
        assert np.allclose(K, K.T)  # Simetría
    
    def test_linear_psd(self):
        """Test de que el kernel lineal es PSD."""
        X = np.random.randn(20, 5)
        linear = LinearKernel()
        assert linear.is_psd(X)
    
    def test_linear_is_dot_product(self):
        """Test de que el kernel lineal es el producto punto."""
        X = np.random.randn(10, 3)
        linear = LinearKernel()
        K = linear.gram_matrix(X)
        K_expected = np.dot(X, X.T)
        
        assert np.allclose(K, K_expected)

