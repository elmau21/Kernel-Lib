"""
Tests para optimizadores matemáticos.

Incluye tests para todos los optimizadores implementados.
"""

import numpy as np
import pytest
from kernel.math.optimizers import (
    Optimizer, SGD, Momentum, RMSprop, Adam, AdaGrad, Nesterov,
    AdamW, Nadam, Adadelta, Rprop, SignSGD, Yogi, Adafactor, NovoGrad,
    AdaMax, Lion, Lookahead, Lamb, QHM, Fromage, AddSign, PowerSign,
    RAdam, AdaBelief, AMSBound, AdaBound, Ranger, RangerQH, AggMo,
    AdaMod, SMORMS3, AdaShift, ExtendedRprop
)


class TestOptimizerBase:
    """Tests para clase base Optimizer."""
    
    def test_optimizer_initialization(self):
        """Test inicialización básica."""
        optimizer = Optimizer(learning_rate=0.01)
        assert optimizer.learning_rate == 0.01
        assert optimizer.iterations == 0


class TestSGD:
    """Tests para SGD."""
    
    def test_sgd_basic(self):
        """Test SGD básico sin momentum."""
        sgd = SGD(learning_rate=0.1, momentum=0.0)
        params = [np.array([1.0, 2.0])]
        grads = [np.array([0.5, -0.3])]
        
        updated = sgd.update(params, grads)
        expected = params[0] - 0.1 * grads[0]
        assert np.allclose(updated[0], expected)
    
    def test_sgd_with_momentum(self):
        """Test SGD con momentum."""
        sgd = SGD(learning_rate=0.1, momentum=0.9)
        params = [np.array([1.0])]
        grads = [np.array([0.5])]
        
        # Primera iteración
        updated1 = sgd.update(params, grads)
        # Segunda iteración
        updated2 = sgd.update(updated1, grads)
        
        assert sgd.iterations == 2
        assert updated2[0] != updated1[0]


class TestAdam:
    """Tests para Adam."""
    
    def test_adam_basic(self):
        """Test Adam básico."""
        adam = Adam(learning_rate=0.001, beta1=0.9, beta2=0.999)
        params = [np.array([1.0, 2.0])]
        grads = [np.array([0.5, -0.3])]
        
        updated = adam.update(params, grads)
        assert updated[0].shape == params[0].shape
        assert adam.iterations == 1
    
    def test_adam_multiple_iterations(self):
        """Test Adam con múltiples iteraciones."""
        adam = Adam(learning_rate=0.001)
        params = [np.array([1.0])]
        grads = [np.array([0.5])]
        
        for _ in range(5):
            params = adam.update(params, grads)
        
        assert adam.iterations == 5
        assert adam.m is not None
        assert adam.v is not None


class TestRMSprop:
    """Tests para RMSprop."""
    
    def test_rmsprop_basic(self):
        """Test RMSprop básico."""
        rmsprop = RMSprop(learning_rate=0.001, decay=0.9)
        params = [np.array([1.0])]
        grads = [np.array([0.5])]
        
        updated = rmsprop.update(params, grads)
        assert updated[0].shape == params[0].shape
        assert rmsprop.cache is not None


class TestAdaGrad:
    """Tests para AdaGrad."""
    
    def test_adagrad_basic(self):
        """Test AdaGrad básico."""
        adagrad = AdaGrad(learning_rate=0.01)
        params = [np.array([1.0])]
        grads = [np.array([0.5])]
        
        updated = adagrad.update(params, grads)
        assert updated[0].shape == params[0].shape
        assert adagrad.G is not None


class TestAdamW:
    """Tests para AdamW."""
    
    def test_adamw_basic(self):
        """Test AdamW básico."""
        adamw = AdamW(learning_rate=0.001, weight_decay=0.01)
        params = [np.array([1.0])]
        grads = [np.array([0.5])]
        
        updated = adamw.update(params, grads)
        assert updated[0].shape == params[0].shape


class TestNadam:
    """Tests para Nadam."""
    
    def test_nadam_basic(self):
        """Test Nadam básico."""
        nadam = Nadam(learning_rate=0.001)
        params = [np.array([1.0])]
        grads = [np.array([0.5])]
        
        updated = nadam.update(params, grads)
        assert updated[0].shape == params[0].shape


class TestRAdam:
    """Tests para RAdam (Rectified Adam)."""
    
    def test_radam_basic(self):
        """Test RAdam básico."""
        radam = RAdam(learning_rate=0.001)
        params = [np.array([1.0])]
        grads = [np.array([0.5])]
        
        updated = radam.update(params, grads)
        assert updated[0].shape == params[0].shape


class TestAdaBelief:
    """Tests para AdaBelief."""
    
    def test_adabelief_basic(self):
        """Test AdaBelief básico."""
        adabelief = AdaBelief(learning_rate=0.001)
        params = [np.array([1.0])]
        grads = [np.array([0.5])]
        
        updated = adabelief.update(params, grads)
        assert updated[0].shape == params[0].shape


class TestLion:
    """Tests para Lion."""
    
    def test_lion_basic(self):
        """Test Lion básico."""
        lion = Lion(learning_rate=0.001)
        params = [np.array([1.0])]
        grads = [np.array([0.5])]
        
        updated = lion.update(params, grads)
        assert updated[0].shape == params[0].shape


class TestRanger:
    """Tests para Ranger."""
    
    def test_ranger_basic(self):
        """Test Ranger básico."""
        ranger = Ranger(learning_rate=0.001)
        params = [np.array([1.0])]
        grads = [np.array([0.5])]
        
        updated = ranger.update(params, grads)
        assert updated[0].shape == params[0].shape


class TestOptimizerConvergence:
    """Tests de convergencia para optimizadores."""
    
    @pytest.mark.parametrize("optimizer_class, kwargs", [
        (SGD, {"learning_rate": 0.01}),
        (Adam, {"learning_rate": 0.001}),
        (RMSprop, {"learning_rate": 0.001}),
        (AdaGrad, {"learning_rate": 0.01}),
        (AdamW, {"learning_rate": 0.001}),
        (RAdam, {"learning_rate": 0.001}),
        (AdaBelief, {"learning_rate": 0.001}),
    ])
    def test_optimizer_converges(self, optimizer_class, kwargs):
        """Test que optimizadores convergen en función simple."""
        optimizer = optimizer_class(**kwargs)
        
        # Función objetivo: f(x) = x² (mínimo en x=0)
        params = [np.array([5.0])]
        
        for _ in range(100):
            # Gradiente: df/dx = 2x
            grads = [2.0 * params[0]]
            params = optimizer.update(params, grads)
        
        # Debe estar cerca de 0
        assert abs(params[0][0]) < 0.1


class TestOptimizerMultipleParams:
    """Tests para optimizadores con múltiples parámetros."""
    
    @pytest.mark.parametrize("optimizer_class", [
        SGD, Adam, RMSprop, AdaGrad, AdamW, RAdam
    ])
    def test_multiple_parameters(self, optimizer_class):
        """Test que optimizadores funcionan con múltiples parámetros."""
        optimizer = optimizer_class(learning_rate=0.01)
        
        params = [
            np.array([1.0, 2.0]),
            np.array([3.0]),
            np.array([[1.0, 2.0], [3.0, 4.0]])
        ]
        grads = [
            np.array([0.1, 0.2]),
            np.array([0.3]),
            np.array([[0.1, 0.2], [0.3, 0.4]])
        ]
        
        updated = optimizer.update(params, grads)
        
        assert len(updated) == len(params)
        for up, p in zip(updated, params):
            assert up.shape == p.shape


class TestOptimizerState:
    """Tests para estado interno de optimizadores."""
    
    def test_adam_state(self):
        """Test que Adam mantiene estado correctamente."""
        adam = Adam(learning_rate=0.001)
        params = [np.array([1.0])]
        grads = [np.array([0.5])]
        
        # Primera actualización
        adam.update(params, grads)
        m1 = adam.m[0].copy()
        v1 = adam.v[0].copy()
        
        # Segunda actualización
        adam.update(params, grads)
        m2 = adam.m[0]
        v2 = adam.v[0]
        
        # Estado debe cambiar
        assert not np.allclose(m1, m2)
        assert not np.allclose(v1, v2)
    
    def test_rmsprop_state(self):
        """Test que RMSprop mantiene estado correctamente."""
        rmsprop = RMSprop(learning_rate=0.001)
        params = [np.array([1.0])]
        grads = [np.array([0.5])]
        
        # Primera actualización
        rmsprop.update(params, grads)
        cache1 = rmsprop.cache[0].copy()
        
        # Segunda actualización
        rmsprop.update(params, grads)
        cache2 = rmsprop.cache[0]
        
        # Cache debe cambiar
        assert not np.allclose(cache1, cache2)

