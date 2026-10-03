import os
import json
import uuid
import shutil
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.ml_model import MLModel

logger = logging.getLogger(__name__)

REGISTRY_FILENAME = "models_registry.json"


class UnifiedModelItem:
    """Wrapper that supports both attribute access (.name) and dict access (['name'])."""
    def __init__(self, data: Dict[str, Any]):
        self.__dict__["_data"] = data
        for k, v in data.items():
            self.__dict__[k] = v

    def __getattr__(self, item):
        return self._data.get(item)

    def __getitem__(self, item):
        return self._data.get(item)

    def get(self, item, default=None):
        return self._data.get(item, default)


def get_registry_path() -> str:
    os.makedirs(settings.WEIGHTS_DIR, exist_ok=True)
    return os.path.join(settings.WEIGHTS_DIR, REGISTRY_FILENAME)


def load_local_registry() -> List[Dict[str, Any]]:
    """Loads locally saved models metadata from weights/models_registry.json."""
    reg_path = get_registry_path()
    if not os.path.exists(reg_path):
        return []
    try:
        with open(reg_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception as e:
        logger.warning(f"Failed to read {reg_path}: {e}")
        return []


def save_local_registry(models: List[Dict[str, Any]]) -> None:
    """Saves local models metadata list to weights/models_registry.json."""
    reg_path = get_registry_path()
    try:
        with open(reg_path, "w", encoding="utf-8") as f:
            json.dump(models, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to write {reg_path}: {e}")


def scan_weights_directory() -> List[Dict[str, Any]]:
    """
    Scans weights/ folder for existing .onnx, .pt, .pkl, .joblib files
    and automatically builds metadata for any unindexed models.
    """
    scanned: List[Dict[str, Any]] = []
    if not os.path.exists(settings.WEIGHTS_DIR):
        return scanned

    builtin_names = {"yolov8n.pt", "yolov8n.onnx", "iris_classifier.joblib"}

    for filename in os.listdir(settings.WEIGHTS_DIR):
        if filename.startswith(".") or filename == REGISTRY_FILENAME or filename.endswith(".py"):
            continue

        file_path = os.path.join(settings.WEIGHTS_DIR, filename)
        if not os.path.isfile(file_path):
            continue

        ext = os.path.splitext(filename)[1].lower()
        if ext not in [".pt", ".onnx", ".pkl", ".joblib"]:
            continue

        # Inferred properties
        base_name = os.path.splitext(filename)[0]
        cleaned_display = base_name.replace("_", " ").title()
        
        if ext in [".pt", ".onnx"]:
            category = "computer_vision"
            framework = "pytorch_yolo" if ext == ".pt" else "onnx"
            task_type = "object_detection"
        else:
            category = "machine_learning"
            framework = "scikit-learn"
            task_type = "classification"

        is_builtin = filename in builtin_names

        scanned.append({
            "id": f"local_{base_name}",
            "name": base_name,
            "display_name": cleaned_display if not is_builtin else f"{cleaned_display} (Built-in)",
            "description": f"Model lokal di folder weights/ ({framework.upper()})",
            "category": category,
            "framework": framework,
            "task_type": task_type,
            "file_path": file_path,
            "is_builtin": is_builtin,
            "source": "local_storage",
            "created_at": datetime.fromtimestamp(os.path.getmtime(file_path), tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        })

    return scanned


async def get_all_models_unified(db: Optional[AsyncSession] = None) -> Dict[str, List[UnifiedModelItem]]:
    """
    Returns all models categorized into 'cv_models' and 'ml_models'.
    Merges PostgreSQL database models, local JSON registry, and loose weights files.
    Completely resilient: never throws 500 even if the database is offline or empty.
    """
    merged_map: Dict[str, Dict[str, Any]] = {}

    # 1. First populate with loose files in weights/ folder
    for item in scan_weights_directory():
        key = os.path.abspath(item["file_path"]).lower()
        merged_map[key] = item

    # 2. Overlay with local JSON registry (has custom display names and descriptions)
    for item in load_local_registry():
        if os.path.exists(item.get("file_path", "")):
            key = os.path.abspath(item["file_path"]).lower()
            merged_map[key] = {**merged_map.get(key, {}), **item, "source": "local_registry"}

    # 3. Overlay with PostgreSQL database models if reachable
    if db is not None:
        try:
            result = await db.execute(select(MLModel).order_by(MLModel.created_at.desc()))
            db_models = result.scalars().all()
            for m in db_models:
                p = m.file_path or ""
                key = os.path.abspath(p).lower() if p else m.id
                merged_map[key] = {
                    "id": m.id,
                    "name": m.name,
                    "display_name": m.display_name,
                    "description": m.description,
                    "category": m.category,
                    "framework": m.framework,
                    "task_type": m.task_type,
                    "file_path": m.file_path,
                    "features": json.loads(m.features_json or "[]"),
                    "target_names": json.loads(m.target_names_json or "[]"),
                    "metrics": json.loads(m.metrics_json or "{}"),
                    "created_at": m.created_at.strftime("%Y-%m-%d %H:%M") if hasattr(m.created_at, "strftime") else str(m.created_at),
                    "source": "database",
                    "is_builtin": False
                }
        except Exception as e:
            logger.warning(f"Could not load models from database (using local fallback): {e}")

    # Separate into CV and ML
    cv_list: List[UnifiedModelItem] = []
    ml_list: List[UnifiedModelItem] = []

    for item in merged_map.values():
        unified = UnifiedModelItem(item)
        if item.get("category") == "computer_vision":
            cv_list.append(unified)
        else:
            ml_list.append(unified)

    return {
        "cv_models": cv_list,
        "ml_models": ml_list,
        "all_models": cv_list + ml_list
    }


async def save_uploaded_model(
    file_bytes: bytes,
    original_filename: str,
    display_name: str,
    category: str,
    framework: str,
    task_type: str = "object_detection",
    description: Optional[str] = None,
    features_json: Optional[str] = None,
    target_names_json: Optional[str] = None,
    db: Optional[AsyncSession] = None
) -> Dict[str, Any]:
    """
    Saves an uploaded model to weights/, persists to local registry,
    and syncs to PostgreSQL database if possible.
    """
    os.makedirs(settings.WEIGHTS_DIR, exist_ok=True)
    ext = os.path.splitext(original_filename)[1].lower()

    # Create safe filename and safe name
    safe_name = "".join(c for c in display_name.lower().replace(" ", "_") if c.isalnum() or c in "_-")
    if not safe_name:
        safe_name = "model"
    
    unique_suffix = uuid.uuid4().hex[:6]
    unique_name = f"{safe_name}_{unique_suffix}"
    final_filename = f"{unique_name}{ext}"
    dest_path = os.path.join(settings.WEIGHTS_DIR, final_filename)

    # 1. Save file locally
    with open(dest_path, "wb") as f:
        f.write(file_bytes)
    
    logger.info(f"Model file saved locally to {dest_path}")

    model_id = str(uuid.uuid4())
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")

    def _json_list(raw: Optional[str]) -> list:
        if not raw:
            return []
        try:
            value = json.loads(raw)
            if not isinstance(value, list):
                raise ValueError("harus berupa array JSON")
            return value
        except (TypeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"Metadata model tidak valid: {exc}") from exc

    feature_schema = _json_list(features_json)
    target_names = _json_list(target_names_json)
    model_metadata = {
        "id": model_id,
        "name": unique_name,
        "display_name": display_name.strip(),
        "description": description.strip() if description else f"Model {framework} untuk {category}",
        "category": category,
        "framework": framework,
        "task_type": task_type,
        "file_path": dest_path,
        "created_at": now_str,
        "source": "local_storage",
        "features": feature_schema,
        "target_names": target_names
    }

    # 2. Persist to local JSON registry (local fallback guaranteed)
    local_registry = load_local_registry()
    local_registry.append(model_metadata)
    save_local_registry(local_registry)

    # 3. Attempt DB sync
    db_synced = False
    if db is not None:
        try:
            db_record = MLModel(
                id=model_id,
                name=unique_name,
                display_name=display_name.strip(),
                description=description.strip() if description else f"Model {framework} untuk {category}",
                category=category,
                framework=framework,
                task_type=task_type,
                file_path=dest_path,
                features_json=json.dumps(feature_schema),
                target_names_json=json.dumps(target_names),
                metrics_json="{}"
            )
            db.add(db_record)
            await db.commit()
            db_synced = True
            logger.info(f"Model {unique_name} successfully synced to database.")
        except Exception as e:
            logger.warning(f"Database sync failed for model (using local fallback): {e}")
            try:
                await db.rollback()
            except Exception:
                pass

    return {
        "success": True,
        "model_id": model_id,
        "name": unique_name,
        "file_path": dest_path,
        "db_synced": db_synced
    }


async def delete_model_unified(
    identifier: str,
    db: Optional[AsyncSession] = None
) -> bool:
    """
    Deletes model from database, local registry, and deletes physical file.
    """
    file_to_delete = None

    # 1. Try delete from DB
    if db is not None:
        try:
            model = await db.get(MLModel, identifier)
            if not model:
                model = await db.scalar(select(MLModel).where(MLModel.name == identifier))
            if model:
                file_to_delete = model.file_path
                await db.delete(model)
                await db.commit()
                logger.info(f"Deleted model {identifier} from database.")
        except Exception as e:
            logger.warning(f"Could not delete {identifier} from database: {e}")

    # 2. Delete from local registry
    local_registry = load_local_registry()
    kept = []
    for item in local_registry:
        if item.get("id") == identifier or item.get("name") == identifier:
            if not file_to_delete:
                file_to_delete = item.get("file_path")
        else:
            kept.append(item)
    save_local_registry(kept)

    # 3. Check scanned directory if file matches identifier
    if not file_to_delete:
        for f in os.listdir(settings.WEIGHTS_DIR):
            if f == identifier or os.path.splitext(f)[0] == identifier:
                file_to_delete = os.path.join(settings.WEIGHTS_DIR, f)
                break

    # 4. Remove physical file
    if file_to_delete and os.path.exists(file_to_delete):
        try:
            os.remove(file_to_delete)
            logger.info(f"Removed physical file: {file_to_delete}")
            return True
        except Exception as e:
            logger.warning(f"Failed to remove file {file_to_delete}: {e}")

    return True


async def resolve_cv_model_path(
    identifier: Optional[str],
    db: Optional[AsyncSession] = None
) -> Optional[str]:
    """
    Resolves the exact file path for a Computer Vision model by identifier (ID, name, or filename).
    """
    if not identifier or identifier in ["default", "yolov8n"]:
        pt_path = os.path.join(settings.WEIGHTS_DIR, "yolov8n.pt")
        onnx_path = os.path.join(settings.WEIGHTS_DIR, settings.CV_MODEL_NAME)
        if os.path.exists(pt_path):
            return pt_path
        if os.path.exists(onnx_path):
            return onnx_path
        return None

    if identifier == "haar_cascade":
        return "haar_cascade"

    # 1. Direct file check in weights/
    direct_path = os.path.join(settings.WEIGHTS_DIR, identifier)
    if os.path.exists(direct_path) and os.path.isfile(direct_path):
        return direct_path

    # Try adding common extensions
    for ext in [".pt", ".onnx"]:
        ext_path = os.path.join(settings.WEIGHTS_DIR, f"{identifier}{ext}")
        if os.path.exists(ext_path) and os.path.isfile(ext_path):
            return ext_path

    # 2. Check DB
    if db is not None:
        try:
            m = await db.scalar(select(MLModel).where((MLModel.id == identifier) | (MLModel.name == identifier)))
            if m and m.file_path and os.path.exists(m.file_path):
                return m.file_path
        except Exception as e:
            logger.warning(f"DB lookup failed for model {identifier}: {e}")

    # 3. Check local registry
    for item in load_local_registry():
        if item.get("id") == identifier or item.get("name") == identifier:
            fp = item.get("file_path")
            if fp and os.path.exists(fp):
                return fp

    # 4. Fuzzy filename search in weights/
    for fname in os.listdir(settings.WEIGHTS_DIR):
        if fname.lower().startswith(identifier.lower()):
            fp = os.path.join(settings.WEIGHTS_DIR, fname)
            if os.path.isfile(fp):
                return fp

    return None
