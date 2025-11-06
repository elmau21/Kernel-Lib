"""
Tests para funciones de activación.

Incluye tests para todas las funciones de activación implementadas.
"""

import numpy as np
import pytest
from kernel.math.activations import (
    Sigmoid, Tanh, ReLU, LeakyReLU, ELU, Softmax, Linear,
    Softplus, Swish, HardSigmoid, HardSwish, GELU, PReLU, SELU, Mish
)


class TestSigmoid:
    """Tests para Sigmoid."""
    
    def test_forward(self):
        """Test forward pass."""
        sigmoid = Sigmoid()
        x = np.array([0.0, 1.0, -1.0])
        result = sigmoid.forward(x)
        assert np.allclose(result, [0.5, 0.731, 0.269], atol=0.01)
        assert np.all(result >= 0) and np.all(result <= 1)
    
    def test_derivative(self):
        """Test derivada."""
        sigmoid = Sigmoid()
        x = np.array([0.0])
        s = sigmoid.forward(x)
        deriv = sigmoid.derivative(x)
        assert np.isclose(deriv[0], s[0] * (1 - s[0]))
    
    def test_backward(self):
        """Test backward pass."""
        sigmoid = Sigmoid()
        x = np.array([1.0])
        grad_output = np.array([2.0])
        grad_input = sigmoid.backward(x, grad_output)
        expected = sigmoid.derivative(x) * grad_output
        assert np.allclose(grad_input, expected)


class TestTanh:
    """Tests para Tanh."""
    
    def test_forward(self):
        """Test forward pass."""
        tanh = Tanh()
        x = np.array([0.0, 1.0, -1.0])
        result = tanh.forward(x)
        assert np.isclose(result[0], 0.0)
        assert np.all(result >= -1) and np.all(result <= 1)
    
    def test_derivative(self):
        """Test derivada."""
        tanh = Tanh()
        x = np.array([0.0])
        t = tanh.forward(x)
        deriv = tanh.derivative(x)
        assert np.isclose(deriv[0], 1.0 - t[0]**2)


class TestReLU:
    """Tests para ReLU."""
    
    def test_forward(self):
        """Test forward pass."""
        relu = ReLU()
        x = np.array([-1.0, 0.0, 1.0, 2.0])
        result = relu.forward(x)
        expected = np.array([0.0, 0.0, 1.0, 2.0])
        assert np.allclose(result, expected)
    
    def test_derivative(self):
        """Test derivada."""
        relu = ReLU()
        x = np.array([-1.0, 0.0, 1.0])
        deriv = relu.derivative(x)
        expected = np.array([0.0, 0.0, 1.0])
        assert np.allclose(deriv, expected)


class TestLeakyReLU:
    """Tests para LeakyReLU."""
    
    def test_forward(self):
        """Test forward pass."""
        leaky = LeakyReLU(alpha=0.01)
        x = np.array([-1.0, 0.0, 1.0])
        result = leaky.forward(x)
        assert np.isclose(result[0], -0.01)
        assert np.isclose(result[2], 1.0)
    
    def test_derivative(self):
        """Test derivada."""
        leaky = LeakyReLU(alpha=0.01)
        x = np.array([-1.0, 1.0])
        deriv = leaky.derivative(x)
        assert np.isclose(deriv[0], 0.01)
        assert np.isclose(deriv[1], 1.0)


class TestELU:
    """Tests para ELU."""
    
    def test_forward(self):
        """Test forward pass."""
        elu = ELU(alpha=1.0)
        x = np.array([-1.0, 0.0, 1.0])
        result = elu.forward(x)
        assert result[1] == 0.0
        assert result[2] == 1.0
        assert result[0] < 0  # Negativo para x < 0
    
    def test_derivative(self):
        """Test derivada."""
        elu = ELU(alpha=1.0)
        x = np.array([1.0])
        deriv = elu.derivative(x)
        assert np.isclose(deriv[0], 1.0)


class TestSoftplus:
    """Tests para Softplus."""
    
    def test_forward(self):
        """Test forward pass."""
        softplus = Softplus()
        x = np.array([0.0, 1.0])
        result = softplus.forward(x)
        assert result[0] > 0
        assert result[1] > result[0]
    
    def test_derivative(self):
        """Test derivada (debe ser sigmoid)."""
        softplus = Softplus()
        x = np.array([0.0])
        deriv = softplus.derivative(x)
        sigmoid_val = 1.0 / (1.0 + np.exp(-0.0))
        assert np.isclose(deriv[0], sigmoid_val)


class TestSwish:
    """Tests para Swish."""
    
    def test_forward(self):
        """Test forward pass."""
        swish = Swish(beta=1.0)
        x = np.array([0.0, 1.0])
        result = swish.forward(x)
        assert np.isclose(result[0], 0.0)
        assert result[1] > 0
    
    def test_derivative(self):
        """Test derivada."""
        swish = Swish(beta=1.0)
        x = np.array([0.0])
        deriv = swish.derivative(x)
        assert deriv[0] > 0


class TestHardSigmoid:
    """Tests para HardSigmoid."""
    
    def test_forward(self):
        """Test forward pass."""
        hard_sigmoid = HardSigmoid()
        x = np.array([-10.0, 0.0, 10.0])
        result = hard_sigmoid.forward(x)
        assert np.isclose(result[0], 0.0)
        assert np.isclose(result[2], 1.0)
        assert np.all(result >= 0) and np.all(result <= 1)
    
    def test_derivative(self):
        """Test derivada."""
        hard_sigmoid = HardSigmoid()
        x = np.array([0.0, 5.0])
        deriv = hard_sigmoid.derivative(x)
        assert np.isclose(deriv[0], 0.2)
        assert np.isclose(deriv[1], 0.0)  # Fuera del rango activo


class TestHardSwish:
    """Tests para HardSwish."""
    
    def test_forward(self):
        """Test forward pass."""
        hard_swish = HardSwish()
        x = np.array([0.0, 1.0])
        result = hard_swish.forward(x)
        assert np.isclose(result[0], 0.0)
        assert result[1] > 0


class TestGELU:
    """Tests para GELU."""
    
    def test_forward(self):
        """Test forward pass."""
        gelu = GELU()
        x = np.array([0.0, 1.0])
        result = gelu.forward(x)
        assert np.isclose(result[0], 0.0, atol=0.01)
        assert result[1] > 0
    
    def test_derivative(self):
        """Test derivada."""
        gelu = GELU()
        x = np.array([0.0])
        deriv = gelu.derivative(x)
        assert deriv[0] > 0


class TestPReLU:
    """Tests para PReLU."""
    
    def test_forward(self):
        """Test forward pass."""
        prelu = PReLU(a=0.25)
        x = np.array([-1.0, 1.0])
        result = prelu.forward(x)
        assert np.isclose(result[0], -0.25)
        assert np.isclose(result[1], 1.0)
    
    def test_derivative(self):
        """Test derivada."""
        prelu = PReLU(a=0.25)
        x = np.array([-1.0, 1.0])
        deriv = prelu.derivative(x)
        assert np.isclose(deriv[0], 0.25)
        assert np.isclose(deriv[1], 1.0)


class TestSELU:
    """Tests para SELU."""
    
    def test_forward(self):
        """Test forward pass."""
        selu = SELU()
        x = np.array([0.0, 1.0])
        result = selu.forward(x)
        assert result[1] > 0
    
    def test_derivative(self):
        """Test derivada."""
        selu = SELU()
        x = np.array([1.0])
        deriv = selu.derivative(x)
        assert deriv[0] > 0


class TestMish:
    """Tests para Mish."""
    
    def test_forward(self):
        """Test forward pass."""
        mish = Mish()
        x = np.array([0.0, 1.0])
        result = mish.forward(x)
        assert np.isclose(result[0], 0.0, atol=0.01)
        assert result[1] > 0
    
    def test_derivative(self):
        """Test derivada."""
        mish = Mish()
        x = np.array([0.0])
        deriv = mish.derivative(x)
        assert deriv[0] > 0


class TestSoftmax:
    """Tests para Softmax."""
    
    def test_forward(self):
        """Test forward pass."""
        softmax = Softmax()
        x = np.array([1.0, 2.0, 3.0])
        result = softmax.forward(x)
        assert np.isclose(np.sum(result), 1.0)
        assert np.all(result >= 0) and np.all(result <= 1)
    
    def test_forward_batch(self):
        """Test forward con batch."""
        softmax = Softmax()
        x = np.array([[1.0, 2.0], [3.0, 1.0]])
        result = softmax.forward(x)
        assert result.shape == (2, 2)
        assert np.allclose(np.sum(result, axis=1), [1.0, 1.0])
    
    def test_derivative(self):
        """Test derivada (Jacobiano)."""
        softmax = Softmax()
        x = np.array([1.0, 2.0])
        jacobian = softmax.derivative(x)
        assert jacobian.shape == (2, 2)


class TestLinear:
    """Tests para Linear."""
    
    def test_forward(self):
        """Test forward pass."""
        linear = Linear()
        x = np.array([1.0, 2.0, 3.0])
        result = linear.forward(x)
        assert np.allclose(result, x)
    
    def test_derivative(self):
        """Test derivada."""
        linear = Linear()
        x = np.array([1.0, 2.0])
        deriv = linear.derivative(x)
        assert np.allclose(deriv, [1.0, 1.0])


class TestActivationProperties:
    """Tests de propiedades generales de activaciones."""
    
    @pytest.mark.parametrize("activation_class", [
        Sigmoid, Tanh, ReLU, LeakyReLU, ELU, Softplus, Swish,
        HardSigmoid, HardSwish, GELU, PReLU, SELU, Mish, Linear
    ])
    def test_backward_consistency(self, activation_class):
        """Test que backward es consistente con derivative."""
        if activation_class == PReLU:
            activation = activation_class(a=0.25)
        elif activation_class == LeakyReLU:
            activation = activation_class(alpha=0.01)
        elif activation_class == ELU:
            activation = activation_class(alpha=1.0)
        elif activation_class == Swish:
            activation = activation_class(beta=1.0)
        elif activation_class == SELU:
            activation = activation_class()
        else:
            activation = activation_class()
        
        x = np.random.randn(5)
        grad_output = np.random.randn(5)
        
        # backward debe ser igual a derivative * grad_output
        grad_backward = activation.backward(x, grad_output)
        grad_expected = activation.derivative(x) * grad_output
        
        assert np.allclose(grad_backward, grad_expected, atol=1e-6)
    
    @pytest.mark.parametrize("activation_class", [
        Sigmoid, Tanh, ReLU, LeakyReLU, ELU, Softplus, Swish,
        HardSigmoid, HardSwish, GELU, PReLU, SELU, Mish, Linear
    ])
    def test_call_method(self, activation_class):
        """Test que __call__ funciona igual que forward."""
        if activation_class == PReLU:
            activation = activation_class(a=0.25)
        elif activation_class == LeakyReLU:
            activation = activation_class(alpha=0.01)
        elif activation_class == ELU:
            activation = activation_class(alpha=1.0)
        elif activation_class == Swish:
            activation = activation_class(beta=1.0)
        elif activation_class == SELU:
            activation = activation_class()
        else:
            activation = activation_class()
        
        x = np.random.randn(5)
        result_call = activation(x)
        result_forward = activation.forward(x)
        
        assert np.allclose(result_call, result_forward)

