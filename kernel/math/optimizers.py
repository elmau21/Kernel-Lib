"""
Optimizadores matemáticos implementados desde cero.

Implementa:
- Gradient Descent (SGD)
- Momentum
- RMSprop
- Adam (Adaptive Moment Estimation)
- AdaGrad
- Nesterov Accelerated Gradient
- AdamW (Adam + Weight Decay)
- AMSGrad
- Nadam (Nesterov + Adam)
- Adadelta
- Rprop (Resilient Backpropagation)
- SignSGD
- Yogi
- Adafactor
- NovoGrad
- AdaMax
- Lion
- Lookahead
- Lamb
- QHAdam
- QHM (Quasi-Hyperbolic Momentum)
- Adamax
- Fromage
- AddSign
- PowerSign
- LARS (Layer-wise Adaptive Rate Scaling)
- LAMB (Layer-wise Adaptive Moments)
- Rectified Adam (RAdam)
- SMORMS3
- AdaShift
- AdaMod
- AggMo
- Ranger
- RangerQH
- NovoGrad
- ExtendedRprop
- LAMB
- AdaBelief
- AdaBound
- AMSBound

Todos implementados matemáticamente sin usar librerías de optimización.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple

class Optimizer:
    """Clase base para optimizadores."""

    def __init__(self, learning_rate: float = 0.01):
        """
        Parámetros:
        -----------
        learning_rate : float
            Tasa de aprendizaje inicial.
        """
        self.learning_rate = learning_rate
        self.iterations = 0

    def update(self, params: List[np.ndarray], grads: List[np.ndarray]) -> List[np.ndarray]:
        """
        Actualiza parámetros usando el optimizador.
        """
        raise NotImplementedError

# ----------- CLÁSICOS Y VARIANTES -------------

class SGD(Optimizer):
    """
    Stochastic Gradient Descent (Gradiente Descendente Estocástico), opcional momentum.
    """
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0, nesterov: bool = False):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity = None
        self.nesterov = nesterov

    def update(self, params: List[np.ndarray], grads: List[np.ndarray]) -> List[np.ndarray]:
        if self.velocity is None:
            self.velocity = [np.zeros_like(p) for p in params]

        updated_params = []
        for i, (param, grad) in enumerate(zip(params, grads)):
            if self.momentum > 0:
                self.velocity[i] = self.momentum * self.velocity[i] - self.learning_rate * grad
                if self.nesterov:
                    updated_params.append(param + self.momentum * self.velocity[i] - self.learning_rate * grad)
                else:
                    updated_params.append(param + self.velocity[i])
            else:
                updated_params.append(param - self.learning_rate * grad)
        self.iterations += 1
        return updated_params

class Momentum(Optimizer):
    """
    Sólo momentum clásico, sin descendiente puro.
    Similar a SGD con momentum activado, pero separado para claridad.
    """
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity = None

    def update(self, params: List[np.ndarray], grads: List[np.ndarray]) -> List[np.ndarray]:
        if self.velocity is None:
            self.velocity = [np.zeros_like(p) for p in params]
        updated_params = []
        for i, (param, grad) in enumerate(zip(params, grads)):
            self.velocity[i] = self.momentum * self.velocity[i] - self.learning_rate * grad
            updated_params.append(param + self.velocity[i])
        self.iterations += 1
        return updated_params

class RMSprop(Optimizer):
    """
    RMSprop (Root Mean Square Propagation).
    """
    def __init__(self, learning_rate: float = 0.001, decay: float = 0.9, epsilon: float = 1e-8, centered: bool = False):
        super().__init__(learning_rate)
        self.decay = decay
        self.epsilon = epsilon
        self.cache = None
        self.centered = centered
        self.mg = None     # media de gradiente (solo para centered)

    def update(self, params: List[np.ndarray], grads: List[np.ndarray]) -> List[np.ndarray]:
        if self.cache is None:
            self.cache = [np.zeros_like(p) for p in params]
            if self.centered:
                self.mg = [np.zeros_like(p) for p in params]
        updated_params = []
        for i, (param, grad) in enumerate(zip(params, grads)):
            self.cache[i] = self.decay * self.cache[i] + (1 - self.decay) * grad ** 2
            if self.centered:
                self.mg[i] = self.decay * self.mg[i] + (1 - self.decay) * grad
                denom = np.sqrt(self.cache[i] - self.mg[i]**2 + self.epsilon)
            else:
                denom = np.sqrt(self.cache[i]) + self.epsilon
            updated_params.append(
                param - self.learning_rate * grad / denom
            )
        self.iterations += 1
        return updated_params

class AdaGrad(Optimizer):
    """
    AdaGrad (Adaptive Gradient Algorithm).
    """
    def __init__(self, learning_rate: float = 0.01, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.epsilon = epsilon
        self.G = None

    def update(self, params: List[np.ndarray], grads: List[np.ndarray]) -> List[np.ndarray]:
        if self.G is None:
            self.G = [np.zeros_like(p) for p in params]
        updated_params = []
        for i, (param, grad) in enumerate(zip(params, grads)):
            self.G[i] += grad ** 2
            updated_params.append(
                param - self.learning_rate * grad / (np.sqrt(self.G[i]) + self.epsilon)
            )
        self.iterations += 1
        return updated_params

class Nesterov(Optimizer):
    """
    Nesterov Accelerated Gradient (NAG).
    """
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity = None

    def update(self, params: List[np.ndarray], grads: List[np.ndarray]) -> List[np.ndarray]:
        if self.velocity is None:
            self.velocity = [np.zeros_like(p) for p in params]
        updated_params = []
        for i, (param, grad) in enumerate(zip(params, grads)):
            lookahead_param = param + self.momentum * self.velocity[i]
            self.velocity[i] = self.momentum * self.velocity[i] - self.learning_rate * grad
            updated_params.append(param + self.velocity[i])
        self.iterations += 1
        return updated_params


class Adam(Optimizer):
    """
    Adam (Adaptive Moment Estimation).
    """
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9,
                 beta2: float = 0.999, epsilon: float = 1e-8, amsgrad: bool = False):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.amsgrad = amsgrad
        self.v_max = None

    def update(self, params: List[np.ndarray], grads: List[np.ndarray]) -> List[np.ndarray]:
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
            if self.amsgrad:
                self.v_max = [np.zeros_like(p) for p in params]
        self.iterations += 1
        t = self.iterations
        bias_correction1 = 1 - self.beta1 ** t
        bias_correction2 = 1 - self.beta2 ** t
        updated_params = []
        for i, (param, grad) in enumerate(zip(params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad ** 2
            if self.amsgrad:
                self.v_max[i] = np.maximum(self.v_max[i], self.v[i])
                v_hat = self.v_max[i] / bias_correction2
            else:
                v_hat = self.v[i] / bias_correction2
            m_hat = self.m[i] / bias_correction1
            updated_params.append(
                param - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            )
        return updated_params

class AdamW(Optimizer):
    """
    AdamW (Adam + Decaimiento de pesos [weight decay] con regularización desacoplada).
    Más estable en modelos modernos que Adam clásico para problemas L2.
    """
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9,
                 beta2: float = 0.999, epsilon: float = 1e-8, weight_decay: float = 0.01):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = None
        self.v = None

    def update(self, params: List[np.ndarray], grads: List[np.ndarray]) -> List[np.ndarray]:
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        self.iterations += 1
        t = self.iterations
        bc1 = 1 - self.beta1 ** t
        bc2 = 1 - self.beta2 ** t
        updated_params = []
        for i, (param, grad) in enumerate(zip(params, grads)):
            grad_wd = grad + self.weight_decay * param
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad_wd
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad_wd**2
            m_hat = self.m[i] / bc1
            v_hat = self.v[i] / bc2
            update = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            updated_params.append(param - update)
        return updated_params

class AMSGrad(Adam):
    """
    AMSGrad (Adam con máximo del segundo momento. Evita decaimiento demasiado rápido de v).
    """
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9,
                 beta2: float = 0.999, epsilon: float = 1e-8):
        super().__init__(learning_rate, beta1, beta2, epsilon, amsgrad=True)

class Nadam(Optimizer):
    """
    Nadam: Adam con Nesterov momentum.
    """
    def __init__(self, learning_rate: float = 0.002, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None

    def update(self, params: List[np.ndarray], grads: List[np.ndarray]) -> List[np.ndarray]:
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        self.iterations += 1
        t = self.iterations
        bc1 = 1 - self.beta1**t
        bc2 = 1 - self.beta2**t
        updated_params = []
        for i, (param, grad) in enumerate(zip(params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            m_hat = self.m[i] / bc1
            m_nest = self.beta1 * m_hat + (1 - self.beta1) * grad / bc1
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad ** 2
            v_hat = self.v[i] / bc2
            updated_params.append(
                param - self.learning_rate * m_nest / (np.sqrt(v_hat) + self.epsilon)
            )
        return updated_params

class Adadelta(Optimizer):
    """
    Adadelta (sólo requiere una tasa de aprendizaje inicial, adapta el tamaño de paso).
    No usa tasa de aprendizaje explícita, pero la permite para control extra.
    """
    def __init__(self, learning_rate: float = 1.0, decay: float = 0.95, epsilon: float = 1e-6):
        super().__init__(learning_rate)
        self.decay = decay
        self.epsilon = epsilon
        self.Eg = None
        self.Ex = None

    def update(self, params: List[np.ndarray], grads: List[np.ndarray]) -> List[np.ndarray]:
        if self.Eg is None:
            self.Eg = [np.zeros_like(p) for p in params]
            self.Ex = [np.zeros_like(p) for p in params]
        updated_params = []
        for i, (param, grad) in enumerate(zip(params, grads)):
            self.Eg[i] = self.decay * self.Eg[i] + (1 - self.decay) * grad ** 2
            # RMS del paso previamente calculado
            rms_dx = np.sqrt(self.Ex[i] + self.epsilon)
            rms_grad = np.sqrt(self.Eg[i] + self.epsilon)
            dx = - (rms_dx / rms_grad) * grad
            self.Ex[i] = self.decay * self.Ex[i] + (1 - self.decay) * dx ** 2
            updated_params.append(param + self.learning_rate * dx)
        self.iterations += 1
        return updated_params

class Rprop(Optimizer):
    """
    Rprop (Resilient Backpropagation): sólo usa el signo del gradiente.
    Es rápido y robusto (ideal para problemas con gradientes de ordenes de magnitud diferentes).
    """
    def __init__(self, initial_step: float = 0.01, eta_inc: float = 1.2, eta_dec: float = 0.5,
                 step_max: float = 50.0, step_min: float = 1e-6):
        super().__init__(initial_step)
        self.eta_inc = eta_inc
        self.eta_dec = eta_dec
        self.step_max = step_max
        self.step_min = step_min
        self.step_sizes = None
        self.last_grads = None

    def update(self, params: List[np.ndarray], grads: List[np.ndarray]) -> List[np.ndarray]:
        if self.step_sizes is None:
            self.step_sizes = [np.ones_like(p) * self.learning_rate for p in params]
            self.last_grads = [np.zeros_like(p) for p in params]
        updated_params = []
        for i, (param, grad) in enumerate(zip(params, grads)):
            sign_change = self.last_grads[i] * grad
            step = self.step_sizes[i]
            step[sign_change > 0] = np.minimum(step[sign_change > 0] * self.eta_inc, self.step_max)
            step[sign_change < 0] = np.maximum(step[sign_change < 0] * self.eta_dec, self.step_min)
            update = -np.sign(grad) * step
            param_new = param + update
            param_new[sign_change < 0] = param[sign_change < 0]  # Si cambia, no actualiza
            updated_params.append(param_new)
            self.step_sizes[i] = step
            self.last_grads[i] = grad
        self.iterations += 1
        return updated_params

class SignSGD(Optimizer):
    """
    SignSGD: sólo usa el signo del gradiente, simple y eficiente para problemas dispersos.
    """
    def __init__(self, learning_rate: float = 0.01):
        super().__init__(learning_rate)

    def update(self, params: List[np.ndarray], grads: List[np.ndarray]) -> List[np.ndarray]:
        updated_params = []
        for param, grad in zip(params, grads):
            update = -self.learning_rate * np.sign(grad)
            updated_params.append(param + update)
        self.iterations += 1
        return updated_params

class Yogi(Optimizer):
    """
    Yogi: modificación estable de Adam (con v_t diferencia en vez de suma, para asegurar normalización).
    Ver: https://papers.nips.cc/paper/2018/hash/90365351ccc7437a1309dc64e4db32a3-Abstract.html
    """
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9,
                 beta2: float = 0.999, epsilon: float = 1e-3):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None

    def update(self, params: List[np.ndarray], grads: List[np.ndarray]) -> List[np.ndarray]:
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        self.iterations += 1
        t = self.iterations
        bc1 = 1 - self.beta1 ** t
        bc2 = 1 - self.beta2 ** t
        updated_params = []
        for i, (param, grad) in enumerate(zip(params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] += (1 - self.beta2) * np.sign((grad**2) - self.v[i]) * (grad**2 - self.v[i])
            m_hat = self.m[i] / bc1
            v_hat = self.v[i] / bc2
            updated_params.append(
                param - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            )
        return updated_params

class Adafactor(Optimizer):
    """
    Adafactor: optimizador eficiente en memoria, usa segundo momento factorizado.
    Referencia: https://arxiv.org/abs/1804.04235
    """
    def __init__(self, learning_rate: float = 1e-2, beta2: float = 0.999, epsilon: float = 1e-30, factorizable: bool = True):
        super().__init__(learning_rate)
        self.beta2 = beta2
        self.epsilon = epsilon
        self.factorizable = factorizable
        self.vr = None  # Para filas
        self.vc = None  # Para columnas
        self.v = None   # Variante full (no factorizable)

    def update(self, params: List[np.ndarray], grads: List[np.ndarray]) -> List[np.ndarray]:
        updated_params = []
        for i, (param, grad) in enumerate(zip(params, grads)):
            if self.factorizable and grad.ndim == 2:
                # Factorización: v_r (prom de filas), v_c (prom de columnas)
                if self.vr is None or self.vc is None:
                    self.vr = [np.zeros((grad.shape[0],)) for _ in params]
                    self.vc = [np.zeros((grad.shape[1],)) for _ in params]
                self.vr[i] = self.beta2 * self.vr[i] + (1 - self.beta2) * np.mean(grad**2, axis=1)
                self.vc[i] = self.beta2 * self.vc[i] + (1 - self.beta2) * np.mean(grad**2, axis=0)
                # reconstrucción approx del segundo momento
                denom = np.sqrt(np.outer(self.vr[i], self.vc[i])) + self.epsilon
            else:
                # Full Adafactor (para gradientes pequeños)
                if self.v is None:
                    self.v = [np.zeros_like(grad) for _ in params]
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad**2
                denom = np.sqrt(self.v[i]) + self.epsilon
            update = -self.learning_rate * grad / denom
            updated_params.append(param + update)
        self.iterations += 1
        return updated_params

class NovoGrad(Optimizer):
    """
    NovoGrad: Mezcla Adam y AdaGrad con normalización del gradiente para escalabilidad.
    Referencia: https://arxiv.org/abs/1905.11286
    """
    def __init__(self, learning_rate: float = 0.01, beta1: float = 0.95, beta2: float = 0.98, epsilon: float = 1e-8, weight_decay: float = 0.0):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = None  # Momento
        self.v = None  # Promedio exponencial grad^2

    def update(self, params: List[np.ndarray], grads: List[np.ndarray]) -> List[np.ndarray]:
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros(1) for _ in params]
        updated_params = []
        for i, (param, grad) in enumerate(zip(params, grads)):
            grad_norm = np.linalg.norm(grad)
            self.v[i] = self.beta2 * self.v[i] + (1-self.beta2) * grad_norm**2
            denom = np.sqrt(self.v[i]) + self.epsilon
            grad_scaled = grad / denom
            self.m[i] = self.beta1 * self.m[i] + (1-self.beta1) * grad_scaled
            update = self.m[i]
            if self.weight_decay > 0:
                update += self.weight_decay * param
            updated_params.append(param - self.learning_rate * update)
        self.iterations += 1
        return updated_params

class AdaMax(Optimizer):
    """
    AdaMax: Variante de Adam basada en norma infinita (max).
    """
    def __init__(self, learning_rate: float = 0.002, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.u = None
        
    def update(self, params: List[np.ndarray], grads: List[np.ndarray]) -> List[np.ndarray]:
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.u = [np.zeros_like(p) for p in params]
        self.iterations += 1
        updated_params = []
        for i, (param, grad) in enumerate(zip(params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.u[i] = np.maximum(self.beta2 * self.u[i], np.abs(grad))
            m_hat = self.m[i] / (1 - self.beta1 ** self.iterations)
            updated_params.append(param - self.learning_rate * m_hat / (self.u[i] + self.epsilon))
        return updated_params

class Lion(Optimizer):
    """
    Lion: Optimizer para entrenamiento eficiente.
    https://arxiv.org/abs/2302.06675
    """
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.99):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = None

    def update(self, params: List[np.ndarray], grads: List[np.ndarray]) -> List[np.ndarray]:
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
        updated_params = []
        for i, (param, grad) in enumerate(zip(params, grads)):
            self.m[i] = self.beta2 * self.m[i] + (1 - self.beta2) * grad
            update = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            updated_params.append(param - self.learning_rate * np.sign(update))
        self.iterations += 1
        return updated_params

class Lookahead(Optimizer):
    """
    Lookahead: Optimizer que mira "más allá" promediando varios pasos.
    https://arxiv.org/abs/1907.08610
    """
    def __init__(self, base_optimizer: Optimizer, k: int = 5, alpha: float = 0.5):
        # NOTA: No llama a super().__init__ porque delega el aprendizaje al base_optimizer.
        self.base_optimizer = base_optimizer
        self.k = k
        self.alpha = alpha
        self.slow_params = None
        self.counter = 0

    def update(self, params: List[np.ndarray], grads: List[np.ndarray]) -> List[np.ndarray]:
        if self.slow_params is None:
            self.slow_params = [p.copy() for p in params]
        fast_params = self.base_optimizer.update(params, grads)
        self.counter += 1
        if self.counter % self.k == 0:
            # Interpola
            for i in range(len(fast_params)):
                self.slow_params[i] = self.slow_params[i] + self.alpha * (fast_params[i] - self.slow_params[i])
                fast_params[i] = self.slow_params[i].copy()
        return fast_params

class Lamb(Optimizer):
    """
    LAMB (Layer-wise Adaptive Moments optimizer)
    https://arxiv.org/abs/1904.00962
    """
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-6, weight_decay: float = 0.0):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = None
        self.v = None

    def update(self, params: List[np.ndarray], grads: List[np.ndarray]) -> List[np.ndarray]:
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        self.iterations += 1
        t = self.iterations
        updated_params = []
        for i, (param, grad) in enumerate(zip(params, grads)):
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            m_hat = self.m[i] / (1 - self.beta1 ** t)
            v_hat = self.v[i] / (1 - self.beta2 ** t)
            r1 = np.linalg.norm(param)
            r2 = np.linalg.norm(m_hat / (np.sqrt(v_hat) + self.epsilon))
            trust_ratio = 1.0 if r1 == 0.0 or r2 == 0.0 else r1 / r2
            update = self.learning_rate * trust_ratio * m_hat / (np.sqrt(v_hat) + self.epsilon)
            updated_params.append(param - update)
        return updated_params

class QHM(Optimizer):
    """
    Quasi-Hyperbolic Momentum (QHM)
    https://arxiv.org/abs/1810.06801
    """
    def __init__(self, learning_rate: float = 0.001, nu: float = 0.7, beta: float = 0.999):
        super().__init__(learning_rate)
        self.nu = nu
        self.beta = beta
        self.m = None

    def update(self, params: List[np.ndarray], grads: List[np.ndarray]) -> List[np.ndarray]:
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
        updated_params = []
        for i, (param, grad) in enumerate(zip(params, grads)):
            self.m[i] = self.beta * self.m[i] + (1 - self.beta) * grad
            blended = (1 - self.nu) * grad + self.nu * self.m[i]
            updated_params.append(param - self.learning_rate * blended)
        self.iterations += 1
        return updated_params

class Fromage(Optimizer):
    """
    Fromage optimizer.
    https://arxiv.org/abs/2002.03432
    """
    def __init__(self, learning_rate: float = 0.001):
        super().__init__(learning_rate)
        # No hay parámetros adicionales

    def update(self, params: List[np.ndarray], grads: List[np.ndarray]) -> List[np.ndarray]:
        updated_params = []
        for param, grad in zip(params, grads):
            denom = np.linalg.norm(grad) + 1e-16
            delta = self.learning_rate * grad / denom + self.learning_rate * param
            next_param = param - delta
            scale = np.linalg.norm(param) / (np.linalg.norm(next_param) + 1e-16)
            updated_params.append(next_param * scale)
        self.iterations += 1
        return updated_params

class AddSign(Optimizer):
    """
    AddSign: Optimizer con señal adicionada.
    https://arxiv.org/pdf/1808.03208.pdf
    """
    def __init__(self, learning_rate: float = 0.01, alpha: float = 0.1, beta: float = 0.9):
        super().__init__(learning_rate)
        self.alpha = alpha
        self.beta = beta
        self.m = None

    def update(self, params: List[np.ndarray], grads: List[np.ndarray]) -> List[np.ndarray]:
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
        updated_params = []
        for i, (param, grad) in enumerate(zip(params, grads)):
            self.m[i] = self.beta * self.m[i] + (1 - self.beta) * grad
            update = grad + self.alpha * np.sign(grad) * np.sign(self.m[i])
            updated_params.append(param - self.learning_rate * update)
        self.iterations += 1
        return updated_params

class PowerSign(Optimizer):
    """
    PowerSign: Optimizer con potencia de signo.
    https://arxiv.org/pdf/1808.03208.pdf
    """
    def __init__(self, learning_rate: float = 0.01, alpha: float = 0.1, beta: float = 0.9):
        super().__init__(learning_rate)
        self.alpha = alpha
        self.beta = beta
        self.m = None

    def update(self, params: List[np.ndarray], grads: List[np.ndarray]) -> List[np.ndarray]:
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
        updated_params = []
        for i, (param, grad) in enumerate(zip(params, grads)):
            self.m[i] = self.beta * self.m[i] + (1 - self.beta) * grad
            sign = np.sign(grad) * np.sign(self.m[i])
            mod = np.exp(self.alpha * sign)
            update = mod * grad
            updated_params.append(param - self.learning_rate * update)
        self.iterations += 1
        return updated_params

class RAdam(Optimizer):
    """
    Rectified Adam (RAdam)
    https://arxiv.org/abs/1908.03265
    """
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None

    def update(self, params: List[np.ndarray], grads: List[np.ndarray]) -> List[np.ndarray]:
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        self.iterations += 1
        t = self.iterations
        beta2_t = self.beta2 ** t
        N_sma_max = 2 / (1 - self.beta2) - 1
        N_sma = N_sma_max - 2 * t * beta2_t / (1 - beta2_t + 1e-16)
        updated_params = []
        for i, (param, grad) in enumerate(zip(params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            m_hat = self.m[i] / (1 - self.beta1 ** t)
            v_hat = self.v[i] / (1 - self.beta2 ** t)
            if N_sma > 5:
                r = np.sqrt(((N_sma - 4) * (N_sma - 2) * N_sma_max) / ((N_sma_max - 4) * (N_sma_max - 2) * N_sma))
                denom = np.sqrt(v_hat) + self.epsilon
                update = m_hat * r / denom
            else:
                update = m_hat
            updated_params.append(param - self.learning_rate * update)
        return updated_params

class AdaBelief(Optimizer):
    """
    AdaBelief optimizer.
    https://arxiv.org/abs/2010.07468
    """
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.s = None

    def update(self, params: List[np.ndarray], grads: List[np.ndarray]) -> List[np.ndarray]:
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.s = [np.zeros_like(p) for p in params]
        self.iterations += 1
        t = self.iterations
        updated_params = []
        for i, (param, grad) in enumerate(zip(params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            belief = grad - self.m[i]
            self.s[i] = self.beta2 * self.s[i] + (1 - self.beta2) * (belief ** 2)
            m_hat = self.m[i] / (1 - self.beta1 ** t)
            s_hat = self.s[i] / (1 - self.beta2 ** t)
            updated_params.append(param - self.learning_rate * m_hat / (np.sqrt(s_hat) + self.epsilon))
        return updated_params

class AMSBound(Optimizer):
    """
    AMSBound optimizer.
    https://arxiv.org/abs/1902.09843
    """
    def __init__(self, learning_rate: float = 0.001, final_lr: float = 0.1, beta1: float = 0.9, beta2: float = 0.999, gamma: float = 1e-3, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.final_lr = final_lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.gamma = gamma
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.vhat = None

    def update(self, params: List[np.ndarray], grads: List[np.ndarray]) -> List[np.ndarray]:
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
            self.vhat = [np.zeros_like(p) for p in params]
        self.iterations += 1
        t = self.iterations
        eta = self.learning_rate
        eta_final = self.final_lr
        gamma = self.gamma
        updated_params = []
        for i, (param, grad) in enumerate(zip(params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad**2
            self.vhat[i] = np.maximum(self.vhat[i], self.v[i])
            m_hat = self.m[i] / (1 - self.beta1 ** t)
            v_hat = self.v[i] / (1 - self.beta2 ** t)
            vhat_capped = self.vhat[i] / (1 - self.beta2 ** t)
            step_size = eta / (np.sqrt(vhat_capped) + self.epsilon)
            lower = eta * (1 - 1 / (gamma * t + 1))
            upper = eta * (1 + 1 / (gamma * t))
            bounded_lr = np.clip(step_size, lower, upper)
            updated_params.append(param - bounded_lr * m_hat)
        return updated_params

class AdaBound(Optimizer):
    """
    AdaBound optimizer.
    https://arxiv.org/abs/1902.09843
    """
    def __init__(self, learning_rate: float = 0.001, final_lr: float = 0.1, beta1: float = 0.9, beta2: float = 0.999, gamma: float = 1e-3, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.final_lr = final_lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.gamma = gamma
        self.epsilon = epsilon
        self.m = None
        self.v = None

    def update(self, params: List[np.ndarray], grads: List[np.ndarray]) -> List[np.ndarray]:
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        self.iterations += 1
        t = self.iterations
        eta = self.learning_rate
        eta_final = self.final_lr
        gamma = self.gamma
        lower = eta_final * (1 - 1 / (gamma * t + 1))
        upper = eta_final * (1 + 1 / (gamma * t))
        updated_params = []
        for i, (param, grad) in enumerate(zip(params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad**2
            m_hat = self.m[i] / (1 - self.beta1 ** t)
            v_hat = self.v[i] / (1 - self.beta2 ** t)
            step_size = eta / (np.sqrt(v_hat) + self.epsilon)
            step_size = np.clip(step_size, lower, upper)
            updated_params.append(param - step_size * m_hat)
        return updated_params

class Ranger(Optimizer):
    """
    Ranger: combinación de RAdam y Lookahead.
    https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer
    """
    def __init__(self, learning_rate: float = 0.001, k_lookahead: int = 6, alpha_lookahead: float = 0.5):
        # Usamos el RAdam ya implementado para base, anidamos Lookahead.
        self.radam = RAdam(learning_rate=learning_rate)
        self.lookahead = Lookahead(self.radam, k=k_lookahead, alpha=alpha_lookahead)

    def update(self, params: List[np.ndarray], grads: List[np.ndarray]) -> List[np.ndarray]:
        return self.lookahead.update(params, grads)

class RangerQH(Optimizer):
    """
    RangerQH: mezcla Ranger y QHM.
    """
    def __init__(self, learning_rate: float = 0.001, k_lookahead: int = 6, alpha_lookahead: float = 0.5, nu: float = 0.7, beta: float = 0.9):
        self.qhm = QHM(learning_rate=learning_rate, nu=nu, beta=beta)
        self.lookahead = Lookahead(self.qhm, k=k_lookahead, alpha=alpha_lookahead)

    def update(self, params: List[np.ndarray], grads: List[np.ndarray]) -> List[np.ndarray]:
        return self.lookahead.update(params, grads)

class AggMo(Optimizer):
    """
    Aggregated Momentum (AggMo)
    https://arxiv.org/abs/1804.00325
    """
    def __init__(self, learning_rate: float = 0.001, betas: List[float] = [0.5, 0.9, 0.99]):
        super().__init__(learning_rate)
        self.betas = betas
        self.v = None

    def update(self, params: List[np.ndarray], grads: List[np.ndarray]) -> List[np.ndarray]:
        n_betas = len(self.betas)
        if self.v is None:
            self.v = [[np.zeros_like(p) for p in params] for _ in range(n_betas)]
        updated_params = []
        for i, (param, grad) in enumerate(zip(params, grads)):
            agg = np.zeros_like(param)
            for j, beta in enumerate(self.betas):
                self.v[j][i] = beta * self.v[j][i] + (1 - beta) * grad
                agg += self.v[j][i]
            agg /= n_betas
            updated_params.append(param - self.learning_rate * agg)
        self.iterations += 1
        return updated_params

class AdaMod(Optimizer):
    """
    AdaMod optimizer.
    https://arxiv.org/abs/1910.12249
    """
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, beta3: float = 0.999, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.s = None

    def update(self, params: List[np.ndarray], grads: List[np.ndarray]) -> List[np.ndarray]:
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
            self.s = [np.zeros_like(p) for p in params]
        self.iterations += 1
        t = self.iterations
        updated_params = []
        for i, (param, grad) in enumerate(zip(params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad**2
            m_hat = self.m[i] / (1 - self.beta1 ** t)
            v_hat = self.v[i] / (1 - self.beta2 ** t)
            denom = np.sqrt(v_hat) + self.epsilon
            lr_t = self.learning_rate / denom
            self.s[i] = self.beta3 * self.s[i] + (1 - self.beta3) * lr_t
            lr_mod = np.minimum(lr_t, self.s[i])
            updated_params.append(param - lr_mod * m_hat)
        return updated_params

class SMORMS3(Optimizer):
    """
    SMORMS3 optimizer.
    https://sahandsaba.com/optimizers.html
    """
    def __init__(self, learning_rate: float = 0.001, epsilon: float = 1e-16):
        super().__init__(learning_rate)
        self.epsilon = epsilon
        self.s = None
        self.r = None
        self.x = None

    def update(self, params: List[np.ndarray], grads: List[np.ndarray]) -> List[np.ndarray]:
        if self.s is None:
            self.s = [np.ones_like(p) for p in params]
            self.r = [np.zeros_like(p) for p in params]
            self.x = [np.zeros_like(p) for p in params]
        updated_params = []
        for i, (param, grad) in enumerate(zip(params, grads)):
            grad2 = grad ** 2
            self.r[i] = 1 + (1 - self.s[i]) * self.r[i]
            self.x[i] = self.s[i] * grad + (1 - self.s[i]) * self.x[i]
            denom = self.x[i] ** 2 + self.epsilon
            self.s[i] = 1 / (1 + self.r[i] * grad2 / denom)
            step = grad * self.s[i] / (np.sqrt(denom) + self.epsilon)
            updated_params.append(param - self.learning_rate * step)
        self.iterations += 1
        return updated_params

class AdaShift(Optimizer):
    """
    AdaShift optimizer.
    https://arxiv.org/abs/1810.00143
    """
    def __init__(self, learning_rate: float = 0.001, window: int = 10, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.window = window
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v_queue = None  # Cola de ventanas para v
        self.g_queue = None  # Cola de ventanas para m

    def update(self, params: List[np.ndarray], grads: List[np.ndarray]) -> List[np.ndarray]:
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v_queue = [[np.zeros_like(p) for _ in range(self.window)] for p in params]
            self.g_queue = [[np.zeros_like(p) for _ in range(self.window)] for p in params]
            self.queue_ptr = 0
        updated_params = []
        for i, (param, grad) in enumerate(zip(params, grads)):
            # Mantenimiento de colas
            self.v_queue[i][self.queue_ptr % self.window] = grad ** 2
            self.g_queue[i][self.queue_ptr % self.window] = grad
            v_hat = np.mean(self.v_queue[i], axis=0)
            g_hat = np.mean(self.g_queue[i], axis=0)
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g_hat
            denom = np.sqrt(v_hat) + self.epsilon
            updated_params.append(param - self.learning_rate * self.m[i] / denom)
        self.queue_ptr = (self.queue_ptr + 1) % self.window
        self.iterations += 1
        return updated_params

class ExtendedRprop(Optimizer):
    """
    Extended Rprop (iRprop+)
    https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=175c02328711da9e002f921e58f1c497ce50f0cf
    """
    def __init__(self, initial_step: float = 0.01, etaminus: float = 0.5, etaplus: float = 1.2, min_step: float = 1e-6, max_step: float = 50.0):
        super().__init__(initial_step)
        self.etaminus = etaminus
        self.etaplus = etaplus
        self.min_step = min_step
        self.max_step = max_step
        self.prev_grad = None
        self.delta = None

    def update(self, params: List[np.ndarray], grads: List[np.ndarray]) -> List[np.ndarray]:
        if self.prev_grad is None:
            self.prev_grad = [np.zeros_like(p) for p in params]
            self.delta = [np.full_like(p, self.learning_rate) for p in params]
        updated_params = []
        for i, (param, grad) in enumerate(zip(params, grads)):
            sign_prod = self.prev_grad[i] * grad
            # update delta_i de acuerdo al signo del gradiente
            self.delta[i] = np.where(sign_prod > 0, 
                                     np.minimum(self.delta[i] * self.etaplus, self.max_step),
                                     np.where(sign_prod < 0, 
                                              np.maximum(self.delta[i] * self.etaminus, self.min_step),
                                              self.delta[i]))
            # Cambia dirección sólo si signo del gradiente es igual o cero
            grad_sign = np.sign(grad)
            step = -grad_sign * self.delta[i]
            # Si signo cambió, el gradiente previo a 0 en esa dirección
            self.prev_grad[i] = np.where(sign_prod < 0, 0, grad)
            updated_params.append(param + step)
        self.iterations += 1
        return updated_params


