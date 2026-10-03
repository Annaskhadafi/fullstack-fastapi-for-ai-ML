import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Union

import joblib
import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.ml_model import MLModel
from app.schemas.ml import FeatureDefinition, ModelInfo, PredictResponse
from app.services.model_hub_service import get_all_models_unified, load_local_registry

logger = logging.getLogger(__name__)
_model_cache: Dict[str, Any] = {}
_onnx_cache: Dict[str, Any] = {}
DEFAULT_IRIS_FEATURES = [
    {"name": "sepal_length", "label": "Sepal Length (cm)", "type": "number", "min_value": 4.0, "max_value": 8.0},
    {"name": "sepal_width", "label": "Sepal Width (cm)", "type": "number", "min_value": 2.0, "max_value": 5.0},
    {"name": "petal_length", "label": "Petal Length (cm)", "type": "number", "min_value": 1.0, "max_value": 7.0},
    {"name": "petal_width", "label": "Petal Width (cm)", "type": "number", "min_value": 0.1, "max_value": 3.0},
]
DEFAULT_IRIS_TARGETS = ["Iris Setosa", "Iris Versicolor", "Iris Virginica"]
DEFAULT_IRIS_METRICS = {"accuracy": 0.98, "algorithm": "RandomForestClassifier", "n_estimators": 50}


class ModelNotFoundError(ValueError): pass
class ModelInputError(ValueError): pass


def get_cached_model(file_path: str) -> Any:
    if file_path not in _model_cache:
        if not os.path.isfile(file_path): raise ModelNotFoundError("File model tidak ditemukan")
        _model_cache[file_path] = joblib.load(file_path)
    return _model_cache[file_path]


def _features(raw: Any) -> List[FeatureDefinition]:
    result = []
    for item in raw if isinstance(raw, list) else []:
        if isinstance(item, str): item = {"name": item}
        if isinstance(item, dict) and item.get("name"):
            item = dict(item); item.setdefault("label", item["name"]); result.append(FeatureDefinition(**item))
    return result


def _inferred_features(model: Any, path: str) -> List[FeatureDefinition]:
    names = getattr(model, "feature_names_in_", None)
    count = getattr(model, "n_features_in_", None)
    if names is not None: return [FeatureDefinition(name=str(x), label=str(x)) for x in names]
    if count is not None: return [FeatureDefinition(name=f"feature_{i+1}", label=f"Feature {i+1}") for i in range(int(count))]
    raise ModelInputError(f"Model '{os.path.basename(path)}' tidak memiliki schema fitur")


def _onnx_session(path: str):
    if path not in _onnx_cache:
        try:
            import onnxruntime as ort
            _onnx_cache[path] = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        except Exception as exc: raise ModelInputError(f"Model ONNX tidak dapat dibuka: {exc}") from exc
    return _onnx_cache[path]


def _onnx_features(session: Any) -> List[FeatureDefinition]:
    inputs = session.get_inputs()
    if len(inputs) != 1 or len(inputs[0].shape) != 2 or not isinstance(inputs[0].shape[1], int):
        raise ModelInputError("Model ONNX harus memiliki tepat satu input tensor numerik 2D dengan jumlah fitur tetap")
    return [FeatureDefinition(name=f"feature_{i+1}", label=f"Feature {i+1}") for i in range(inputs[0].shape[1])]


def _parse_json(value: Optional[str], default: Any) -> Any:
    try: return json.loads(value) if value else default
    except (TypeError, ValueError): return default


async def seed_demo_models(db: Optional[AsyncSession] = None) -> None:
    os.makedirs(settings.WEIGHTS_DIR, exist_ok=True); path = os.path.join(settings.WEIGHTS_DIR, "iris_classifier.joblib")
    if not os.path.exists(path):
        try:
            from sklearn.datasets import load_iris
            from sklearn.ensemble import RandomForestClassifier
            iris = load_iris(); joblib.dump(RandomForestClassifier(n_estimators=50, random_state=42).fit(iris.data, iris.target), path)
        except Exception as exc: logger.warning("Could not train demo Iris model: %s", exc)
    if db is not None:
        try:
            if not await db.scalar(select(MLModel).where(MLModel.name == "iris_classifier")):
                db.add(MLModel(name="iris_classifier", display_name="Iris Flower Classification", description="Klasifikasi spesies bunga Iris berdasarkan panjang & lebar sepal dan petal.", category="machine_learning", framework="scikit-learn", task_type="classification", file_path=path, features_json=json.dumps(DEFAULT_IRIS_FEATURES), target_names_json=json.dumps(DEFAULT_IRIS_TARGETS), metrics_json=json.dumps(DEFAULT_IRIS_METRICS))); await db.commit()
        except Exception as exc: logger.warning("Could not register demo model: %s", exc)


async def _metadata(db: Optional[AsyncSession], name: str) -> Dict[str, Any]:
    if db is not None:
        try:
            m = await db.scalar(select(MLModel).where(MLModel.name == name))
            if m: return {"id": m.id, "name": m.name, "display_name": m.display_name, "description": m.description, "task_type": m.task_type, "framework": m.framework, "file_path": m.file_path, "features": _parse_json(m.features_json, []), "target_names": _parse_json(m.target_names_json, []), "metrics": _parse_json(m.metrics_json, None)}
        except Exception as exc: logger.warning("DB model lookup failed for %s: %s", name, exc)
    for item in load_local_registry():
        if item.get("name") == name: return dict(item)
    if name == "iris_classifier":
        path = os.path.join(settings.WEIGHTS_DIR, "iris_classifier.joblib")
        if os.path.isfile(path): return {"id": "local_iris_classifier", "name": name, "display_name": "Iris Flower Classification", "description": "Klasifikasi spesies bunga Iris (RandomForest).", "task_type": "classification", "file_path": path, "features": DEFAULT_IRIS_FEATURES, "target_names": DEFAULT_IRIS_TARGETS, "metrics": DEFAULT_IRIS_METRICS}
    for item in (await get_all_models_unified(db))["ml_models"]:
        if item.get("name") == name: return dict(item._data)
    raise ModelNotFoundError(f"Model '{name}' tidak ditemukan")


async def list_available_models(db: Optional[AsyncSession] = None) -> List[ModelInfo]:
    result = []
    for item in (await get_all_models_unified(db))["ml_models"]:
        data = dict(item._data); path = data.get("file_path")
        if not path or not os.path.isfile(path): continue
        feats = _features(data.get("features"))
        try:
            if not feats: feats = _onnx_features(_onnx_session(path)) if str(path).lower().endswith(".onnx") else _inferred_features(get_cached_model(path), path)
        except ModelInputError: feats = []
        result.append(ModelInfo(id=data.get("id", data["name"]), name=data["name"], display_name=data.get("display_name", data["name"]), description=data.get("description"), task_type=data.get("task_type", "classification"), features=feats, target_names=data.get("target_names") or None, metrics=data.get("metrics")))
    return result


def _input_vector(features: Sequence[FeatureDefinition], inputs: Union[Dict[str, Any], List[Any]], count: int) -> np.ndarray:
    if isinstance(inputs, list):
        if len(inputs) != count: raise ModelInputError(f"Model membutuhkan {count} fitur, tetapi menerima {len(inputs)}")
        values = inputs
    elif isinstance(inputs, dict):
        names = [x.name for x in features]
        if not names: raise ModelInputError("Model tidak memiliki schema fitur untuk input bernama")
        missing = [x for x in names if x not in inputs]; unknown = [x for x in inputs if x not in names]
        if missing or unknown: raise ModelInputError("Input fitur tidak valid (" + "; ".join(([f"missing: {', '.join(missing)}"] if missing else []) + ([f"unknown: {', '.join(unknown)}"] if unknown else [])) + ")")
        values = [inputs[x] for x in names]
    else: raise ModelInputError("features harus berupa object atau ordered list")
    try: return np.asarray([[float(x) for x in values]], dtype=np.float32)
    except (TypeError, ValueError) as exc: raise ModelInputError("Semua fitur harus berupa angka") from exc


async def predict_with_model(db: Optional[AsyncSession], model_name: str, feature_inputs: Union[Dict[str, Any], List[Any]]) -> PredictResponse:
    started = time.time(); meta = await _metadata(db, model_name); path = meta.get("file_path")
    if not path or not os.path.isfile(path): raise ModelNotFoundError(f"Model '{model_name}' tidak ditemukan")
    targets = meta.get("target_names") or (DEFAULT_IRIS_TARGETS if model_name == "iris_classifier" else [])
    if str(path).lower().endswith(".onnx"):
        session = _onnx_session(path); feats = _features(meta.get("features")) or _onnx_features(session); X = _input_vector(feats, feature_inputs, len(feats)); outputs = session.run(None, {session.get_inputs()[0].name: X}); first = np.asarray(outputs[0]); raw = first[0] if first.ndim > 1 else first.reshape(-1)[0]; probs = next((np.asarray(x)[0] for x in outputs[1:] if np.asarray(x).ndim == 2), None)
        if np.asarray(raw).size > 1: probs = np.asarray(raw); raw = int(np.argmax(raw))
    else:
        model = get_cached_model(path); feats = _features(meta.get("features")) or _inferred_features(model, path); X = _input_vector(feats, feature_inputs, len(feats)); raw = model.predict(X)[0]; probs = model.predict_proba(X)[0] if hasattr(model, "predict_proba") else None
    raw = raw.item() if isinstance(raw, np.generic) else raw
    probabilities = {str(targets[i]) if i < len(targets) else f"Class {i}": round(float(x), 4) for i, x in enumerate(np.asarray(probs).reshape(-1))} if probs is not None else None
    label = str(targets[int(raw)]) if targets and isinstance(raw, (int, np.integer)) and 0 <= int(raw) < len(targets) else str(raw)
    return PredictResponse(success=True, model_name=meta.get("display_name", model_name), prediction=raw, prediction_label=label, probabilities=probabilities, execution_time_ms=round((time.time() - started) * 1000, 2))
