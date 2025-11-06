"""
API REST completa para Kernel ML Engine.

Implementa endpoints para:
- Cálculo de kernels
- Entrenamiento de modelos (SVM, KPCA, GP)
- Predicciones
- Optimización de hiperparámetros
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import numpy as np
import json
import uuid
from datetime import datetime

from kernel.kernels.rbf import RBFKernel
from kernel.kernels.polynomial import PolynomialKernel
from kernel.kernels.linear import LinearKernel
from kernel.kernels.matern import MaternKernel
from kernel.kernels.laplacian import LaplacianKernel
from kernel.kernels.composite import CompositeKernel, ScaledKernel
from kernel.methods.svm import KernelSVM
from kernel.methods.kpca import KernelPCA
from kernel.methods.gaussian_process import GaussianProcess

app = FastAPI(
    title="Kernel ML Engine API",
    description="API REST para métodos de kernel en machine learning",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Almacenamiento en memoria (en producción usar Redis/DB)
models_store: Dict[str, Any] = {}
kernels_store: Dict[str, Any] = {}


# Schemas Pydantic
class KernelConfig(BaseModel):
    type: str = Field(..., description="Tipo de kernel: rbf, polynomial, linear, matern, laplacian")
    params: Dict[str, Any] = Field(default_factory=dict, description="Parámetros del kernel")
    use_cache: bool = True
    enable_gpu: bool = False


class KernelComputeRequest(BaseModel):
    X: List[List[float]] = Field(..., description="Primer conjunto de datos")
    Y: Optional[List[List[float]]] = None
    kernel_config: KernelConfig


class SVMTrainRequest(BaseModel):
    X: List[List[float]]
    y: List[float]
    kernel_config: KernelConfig
    C: float = 1.0
    tol: float = 1e-3
    max_iter: int = 1000


class SVMPredictRequest(BaseModel):
    model_id: str
    X: List[List[float]]


class KPCATrainRequest(BaseModel):
    X: List[List[float]]
    kernel_config: KernelConfig
    n_components: Optional[int] = None
    center_kernel: bool = True


class KPCATransformRequest(BaseModel):
    model_id: str
    X: List[List[float]]


class GPTrainRequest(BaseModel):
    X: List[List[float]]
    y: List[float]
    kernel_config: KernelConfig
    alpha: float = 1e-6
    normalize_y: bool = False


class GPPredictRequest(BaseModel):
    model_id: str
    X: List[List[float]]
    return_std: bool = True


def create_kernel(config: KernelConfig):
    """Crea un kernel a partir de la configuración."""
    kernel_type = config.type.lower()
    params = config.params.copy()
    params['use_cache'] = config.use_cache
    params['enable_gpu'] = config.enable_gpu
    
    if kernel_type == "rbf":
        return RBFKernel(**params)
    elif kernel_type == "polynomial":
        return PolynomialKernel(**params)
    elif kernel_type == "linear":
        return LinearKernel(**params)
    elif kernel_type == "matern":
        return MaternKernel(**params)
    elif kernel_type == "laplacian":
        return LaplacianKernel(**params)
    else:
        raise ValueError(f"Tipo de kernel desconocido: {kernel_type}")


@app.get("/")
async def root():
    """Endpoint raíz."""
    return {
        "name": "Kernel ML Engine API",
        "version": "1.0.0",
        "endpoints": {
            "kernels": "/kernels",
            "svm": "/svm",
            "kpca": "/kpca",
            "gaussian_process": "/gp"
        }
    }


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# Endpoints de Kernels
@app.post("/kernels/compute")
async def compute_kernel(request: KernelComputeRequest):
    """
    Calcula matriz de kernel.
    
    Retorna la matriz de kernel K(X, Y) o K(X, X) si Y no se proporciona.
    """
    try:
        X = np.array(request.X, dtype=np.float64)
        Y = np.array(request.Y, dtype=np.float64) if request.Y else None
        
        kernel = create_kernel(request.kernel_config)
        K = kernel(X, Y)
        
        return {
            "kernel_matrix": K.tolist(),
            "shape": list(K.shape),
            "kernel_type": request.kernel_config.type
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/kernels/gram_matrix")
async def compute_gram_matrix(request: KernelComputeRequest):
    """Calcula matriz de Gram K(X, X)."""
    try:
        X = np.array(request.X, dtype=np.float64)
        kernel = create_kernel(request.kernel_config)
        K = kernel.gram_matrix(X)
        
        return {
            "gram_matrix": K.tolist(),
            "shape": list(K.shape),
            "is_psd": kernel.is_psd(X).tolist() if hasattr(kernel.is_psd(X), 'tolist') else bool(kernel.is_psd(X))
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Endpoints de SVM
@app.post("/svm/train")
async def train_svm(request: SVMTrainRequest):
    """Entrena un modelo SVM."""
    try:
        X = np.array(request.X, dtype=np.float64)
        y = np.array(request.y, dtype=np.float64)
        
        kernel = create_kernel(request.kernel_config)
        svm = KernelSVM(kernel=kernel, C=request.C, tol=request.tol, 
                       max_iter=request.max_iter)
        svm.fit(X, y)
        
        model_id = str(uuid.uuid4())
        models_store[model_id] = {
            "type": "svm",
            "model": svm,
            "created_at": datetime.now().isoformat()
        }
        
        return {
            "model_id": model_id,
            "n_support_vectors": int(svm.n_support_),
            "support_vectors": svm.support_vectors_.tolist() if svm.support_vectors_ is not None else None
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/svm/predict")
async def predict_svm(request: SVMPredictRequest):
    """Predice usando un modelo SVM."""
    try:
        if request.model_id not in models_store:
            raise HTTPException(status_code=404, detail="Modelo no encontrado")
        
        model_data = models_store[request.model_id]
        if model_data["type"] != "svm":
            raise HTTPException(status_code=400, detail="Modelo no es SVM")
        
        svm = model_data["model"]
        X = np.array(request.X, dtype=np.float64)
        
        predictions = svm.predict(X)
        scores = svm.decision_function(X)
        
        return {
            "predictions": predictions.tolist(),
            "decision_scores": scores.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/svm/models/{model_id}")
async def get_svm_model(model_id: str):
    """Obtiene información de un modelo SVM."""
    if model_id not in models_store:
        raise HTTPException(status_code=404, detail="Modelo no encontrado")
    
    model_data = models_store[model_id]
    if model_data["type"] != "svm":
        raise HTTPException(status_code=400, detail="Modelo no es SVM")
    
    svm = model_data["model"]
    
    return {
        "model_id": model_id,
        "type": "svm",
        "n_support_vectors": int(svm.n_support_) if svm.n_support_ else 0,
        "created_at": model_data["created_at"]
    }


# Endpoints de KPCA
@app.post("/kpca/train")
async def train_kpca(request: KPCATrainRequest):
    """Entrena un modelo KPCA."""
    try:
        X = np.array(request.X, dtype=np.float64)
        
        kernel = create_kernel(request.kernel_config)
        kpca = KernelPCA(kernel=kernel, n_components=request.n_components,
                        center_kernel=request.center_kernel)
        kpca.fit(X)
        
        model_id = str(uuid.uuid4())
        models_store[model_id] = {
            "type": "kpca",
            "model": kpca,
            "created_at": datetime.now().isoformat()
        }
        
        explained_variance = kpca.explained_variance_ratio_()
        
        return {
            "model_id": model_id,
            "n_components": int(len(kpca.eigenvalues_)) if kpca.eigenvalues_ is not None else 0,
            "explained_variance_ratio": explained_variance.tolist() if explained_variance is not None else None
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/kpca/transform")
async def transform_kpca(request: KPCATransformRequest):
    """Transforma datos usando KPCA."""
    try:
        if request.model_id not in models_store:
            raise HTTPException(status_code=404, detail="Modelo no encontrado")
        
        model_data = models_store[request.model_id]
        if model_data["type"] != "kpca":
            raise HTTPException(status_code=400, detail="Modelo no es KPCA")
        
        kpca = model_data["model"]
        X = np.array(request.X, dtype=np.float64)
        
        X_transformed = kpca.transform(X)
        
        return {
            "transformed_data": X_transformed.tolist(),
            "shape": list(X_transformed.shape)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Endpoints de Gaussian Process
@app.post("/gp/train")
async def train_gp(request: GPTrainRequest):
    """Entrena un modelo Gaussian Process."""
    try:
        X = np.array(request.X, dtype=np.float64)
        y = np.array(request.y, dtype=np.float64)
        
        kernel = create_kernel(request.kernel_config)
        gp = GaussianProcess(kernel=kernel, alpha=request.alpha,
                           normalize_y=request.normalize_y)
        gp.fit(X, y)
        
        model_id = str(uuid.uuid4())
        models_store[model_id] = {
            "type": "gp",
            "model": gp,
            "created_at": datetime.now().isoformat()
        }
        
        log_likelihood = gp._log_marginal_likelihood()
        
        return {
            "model_id": model_id,
            "log_marginal_likelihood": float(log_likelihood)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/gp/predict")
async def predict_gp(request: GPPredictRequest):
    """Predice usando un modelo GP."""
    try:
        if request.model_id not in models_store:
            raise HTTPException(status_code=404, detail="Modelo no encontrado")
        
        model_data = models_store[request.model_id]
        if model_data["type"] != "gp":
            raise HTTPException(status_code=400, detail="Modelo no es GP")
        
        gp = model_data["model"]
        X = np.array(request.X, dtype=np.float64)
        
        if request.return_std:
            y_mean, y_std = gp.predict(X, return_std=True)
            return {
                "mean": y_mean.tolist(),
                "std": y_std.tolist()
            }
        else:
            y_mean = gp.predict(X, return_std=False)
            return {
                "mean": y_mean.tolist()
            }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/models")
async def list_models():
    """Lista todos los modelos almacenados."""
    return {
        "models": [
            {
                "model_id": model_id,
                "type": data["type"],
                "created_at": data["created_at"]
            }
            for model_id, data in models_store.items()
        ]
    }


@app.delete("/models/{model_id}")
async def delete_model(model_id: str):
    """Elimina un modelo."""
    if model_id not in models_store:
        raise HTTPException(status_code=404, detail="Modelo no encontrado")
    
    del models_store[model_id]
    return {"message": "Modelo eliminado"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

