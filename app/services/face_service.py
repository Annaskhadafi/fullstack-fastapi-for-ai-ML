import os
import json
import uuid
import time
import base64
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.face import RegisteredFace

logger = logging.getLogger(__name__)

LOCAL_FACES_DIR = os.path.join("data", "faces")
REGISTRY_FILE = os.path.join(LOCAL_FACES_DIR, "faces_registry.json")
PHOTOS_DIR = os.path.join("app", "static", "faces")

os.makedirs(LOCAL_FACES_DIR, exist_ok=True)
os.makedirs(PHOTOS_DIR, exist_ok=True)

_analyzer = None
_spoofer = None


def _optimize_onnx_session(model_obj) -> None:
    """Optimizes ONNX Runtime session with multi-threading and maximum graph optimizations for CPU."""
    try:
        import onnxruntime as ort
        if hasattr(model_obj, "session") and hasattr(model_obj, "model_path"):
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = min(8, os.cpu_count() or 4)
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.log_severity_level = 3
            model_obj.session = ort.InferenceSession(
                model_obj.model_path,
                sess_options=opts,
                providers=["CPUExecutionProvider"]
            )
            logger.debug(f"Optimized ONNX session for {model_obj.__class__.__name__}")
    except Exception as e:
        logger.warning(f"Failed to optimize ONNX session: {e}")


def get_face_analyzer():
    """Lazily initializes and caches the UniFace FaceAnalyzer instance (SCRFD 500M + ArcFace MNET) with acceleration."""
    global _analyzer
    if _analyzer is None:
        try:
            import uniface
            _analyzer = uniface.FaceAnalyzer()
            _optimize_onnx_session(_analyzer.detector)
            if _analyzer.recognizer is not None:
                _optimize_onnx_session(_analyzer.recognizer)
            logger.info("UniFace FaceAnalyzer initialized (SCRFD + ArcFace accelerated).")
        except Exception as e:
            logger.error(f"Failed to initialize UniFace FaceAnalyzer: {e}")
            raise
    return _analyzer


def get_spoofer():
    """Lazily loads and caches the UniFace MiniFASNet anti-spoofing liveness model with acceleration."""
    global _spoofer
    if _spoofer is None:
        try:
            from uniface.spoofing import MiniFASNet
            _spoofer = MiniFASNet()
            _optimize_onnx_session(_spoofer)
            logger.info("UniFace MiniFASNet Anti-Spoofing initialized (accelerated).")
        except Exception as e:
            logger.error(f"Failed to load MiniFASNet: {e}")
            raise
    return _spoofer


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Decodes image bytes to OpenCV BGR numpy array."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Gagal membaca file gambar. Pastikan format JPG/PNG/WebP valid.")
    return img


def encode_image_to_base64(img: np.ndarray, ext: str = ".jpg") -> str:
    """Encodes OpenCV image to base64 data URI."""
    success, buffer = cv2.imencode(ext, img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not success:
        raise ValueError("Gagal melakukan encoding gambar ke JPEG.")
    b64_str = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"


def load_local_faces_registry() -> List[Dict[str, Any]]:
    """Loads registered faces from local JSON file."""
    if not os.path.exists(REGISTRY_FILE):
        return []
    try:
        with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception as e:
        logger.warning(f"Failed to read local faces registry: {e}")
        return []


def save_local_faces_registry(faces: List[Dict[str, Any]]) -> None:
    """Saves registered faces list to local JSON file."""
    try:
        with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
            json.dump(faces, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to write local faces registry: {e}")


async def get_all_registered_faces(db: Optional[AsyncSession] = None) -> List[Dict[str, Any]]:
    """
    Returns all registered faces, merging PostgreSQL database records and local JSON registry.
    Never throws 500 even if the database is offline.
    """
    faces_map: Dict[str, Dict[str, Any]] = {}

    # 1. Load local registry
    for item in load_local_faces_registry():
        faces_map[item["id"]] = item

    # 2. Overlay with DB records if available
    if db is not None:
        try:
            stmt = select(RegisteredFace).order_by(RegisteredFace.created_at.desc())
            result = await db.execute(stmt)
            for rf in result.scalars().all():
                faces_map[rf.id] = {
                    "id": rf.id,
                    "name": rf.name,
                    "identity_number": rf.identity_number,
                    "notes": rf.notes,
                    "photo_url": rf.photo_url,
                    "embedding": json.loads(rf.embedding_json) if rf.embedding_json else [],
                    "created_at": rf.created_at.strftime("%Y-%m-%d %H:%M") if hasattr(rf.created_at, "strftime") else str(rf.created_at),
                    "source": "database"
                }
        except Exception as e:
            logger.warning(f"Could not load registered faces from DB (using local fallback): {e}")

    items = list(faces_map.values())
    items.sort(key=lambda x: str(x.get("created_at", "")), reverse=True)
    return items


async def register_face(
    name: str,
    image_bytes: bytes,
    identity_number: Optional[str] = None,
    notes: Optional[str] = None,
    db: Optional[AsyncSession] = None
) -> Dict[str, Any]:
    """
    Detects face in image, verifies anti-spoofing liveness, extracts 512-dim ArcFace embedding using UniFace,
    saves photo to static directory, persists to local JSON registry, and syncs to DB.
    """
    img = decode_image_bytes(image_bytes)
    analyzer = get_face_analyzer()
    spoofer = get_spoofer()

    # Resize if huge
    max_dim = 640
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        proc_img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        proc_img = img

    faces = analyzer.analyze(proc_img)

    if not faces:
        raise ValueError("Tidak ada wajah terdeteksi pada foto! Pastikan wajah terlihat jelas dan menghadap kamera.")

    # Select the largest face by bounding box area
    best_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    # Anti-spoofing check during registration
    spoof_res = spoofer.predict(proc_img, best_face.bbox)
    if not spoof_res.is_real:
        raise ValueError("Pendaftaran ditolak! Wajah terdeteksi sebagai foto/layar HP (Spoof Attack). Harap gunakan orang asli secara langsung.")

    if best_face.embedding is None:
        raise ValueError("Gagal mengekstrak fitur biometrik (embedding) dari wajah.")

    embedding_list = best_face.embedding.tolist() if hasattr(best_face.embedding, "tolist") else list(best_face.embedding)

    face_id = str(uuid.uuid4())
    photo_filename = f"{face_id}.jpg"
    photo_path = os.path.join(PHOTOS_DIR, photo_filename)

    # Save photo to static folder
    with open(photo_path, "wb") as f:
        f.write(image_bytes)

    photo_url = f"/static/faces/{photo_filename}"
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")

    record_dict = {
        "id": face_id,
        "name": name.strip(),
        "identity_number": identity_number.strip() if identity_number else None,
        "notes": notes.strip() if notes else None,
        "photo_url": photo_url,
        "embedding": embedding_list,
        "created_at": now_str,
        "source": "local_storage"
    }

    # 1. Save to local JSON registry (guaranteed local fallback)
    local_faces = load_local_faces_registry()
    local_faces.append(record_dict)
    save_local_faces_registry(local_faces)

    # 2. Sync to PostgreSQL if DB available
    db_synced = False
    if db is not None:
        try:
            db_face = RegisteredFace(
                id=face_id,
                name=name.strip(),
                identity_number=identity_number.strip() if identity_number else None,
                notes=notes.strip() if notes else None,
                photo_url=photo_url,
                embedding_json=json.dumps(embedding_list)
            )
            db.add(db_face)
            await db.commit()
            db_synced = True
            logger.info(f"Registered face for '{name}' synced to database.")
        except Exception as e:
            logger.warning(f"Failed to sync registered face to DB (persisted locally): {e}")
            try:
                await db.rollback()
            except Exception:
                pass

    return {
        "success": True,
        "id": face_id,
        "name": name.strip(),
        "identity_number": identity_number,
        "photo_url": photo_url,
        "embedding_dim": len(embedding_list),
        "db_synced": db_synced
    }


async def delete_registered_face(face_id: str, db: Optional[AsyncSession] = None) -> bool:
    """Deletes face from database, local registry, and deletes photo file."""
    # 1. Delete from DB
    if db is not None:
        try:
            face = await db.get(RegisteredFace, face_id)
            if face:
                await db.delete(face)
                await db.commit()
                logger.info(f"Deleted face {face_id} from database.")
        except Exception as e:
            logger.warning(f"Could not delete face {face_id} from DB: {e}")

    # 2. Delete from local registry
    local_faces = load_local_faces_registry()
    kept = [f for f in local_faces if f.get("id") != face_id]
    save_local_faces_registry(kept)

    # 3. Remove photo file
    photo_path = os.path.join(PHOTOS_DIR, f"{face_id}.jpg")
    if os.path.exists(photo_path):
        try:
            os.remove(photo_path)
            logger.info(f"Removed photo file: {photo_path}")
        except Exception as e:
            logger.warning(f"Failed to remove photo file: {e}")

    return True


async def recognize_faces(
    image_bytes: bytes,
    similarity_threshold: float = 0.45,
    db: Optional[AsyncSession] = None
) -> Dict[str, Any]:
    """
    High-Performance Face Recognition with Anti-Spoofing Filter:
    1. Downscales image if needed for sub-50ms inference.
    2. Runs SCRFD Face Detection & ArcFace Biometric Extraction.
    3. Runs MiniFASNet Anti-Spoofing: Rejects screen/paper/photo attacks immediately!
    4. Matches verified live human faces with database embeddings.
    """
    start_time = time.time()
    orig_img = decode_image_bytes(image_bytes)
    orig_h, orig_w = orig_img.shape[:2]

    # SPEED OPTIMIZATION: Downscale to max 640px for blazing fast inference
    max_dim = 640
    if max(orig_h, orig_w) > max_dim:
        scale = max_dim / max(orig_h, orig_w)
        proc_w = int(orig_w * scale)
        proc_h = int(orig_h * scale)
        proc_img = cv2.resize(orig_img, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0
        proc_img = orig_img

    analyzer = get_face_analyzer()
    spoofer = get_spoofer()

    # Step 1: Detect Faces with SCRFD (Takes ~15ms)
    faces = analyzer.detector.detect(proc_img)

    registered_faces = await get_all_registered_faces(db)

    recognized_list: List[Dict[str, Any]] = []
    annotated = proc_img.copy()

    total_spoof_attacks = 0

    for idx, face in enumerate(faces):
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(proc_img.shape[1], x2), min(proc_img.shape[0], y2)

        conf = float(face.confidence)

        # Step 2: ANTI-SPOOFING FILTER FIRST (Liveness Verification takes ~7ms)
        # Blocks phone screens, printed photos, and tablets BEFORE any biometrics!
        spoof_res = spoofer.predict(proc_img, face.bbox)
        is_live = bool(spoof_res.is_real)
        liveness_score = round(float(spoof_res.confidence), 3)

        if not is_live:
            # ⚠️ SPOOF ATTACK: Stop right here! Zero biometric computation!
            total_spoof_attacks += 1
            status_str = "SPOOF_ATTACK"
            matched_name = "PALSU: LAYAR HP / FOTO"
            is_registered = False
            identity_num = None
            highest_sim = 0.0

            color = (0, 0, 255)  # Bright Red
            badge_text = "⚠️ SPOOF ATTACK: LAYAR / FOTO"
        else:
            # ✅ REAL LIVE HUMAN FACE -> Only now extract ArcFace 512-dim embedding & match DB
            face_emb = None
            if analyzer.recognizer is not None and getattr(face, "landmarks", None) is not None:
                try:
                    face_emb = analyzer.recognizer.get_normalized_embedding(proc_img, face.landmarks)
                except Exception as e:
                    logger.warning(f"Failed to extract ArcFace embedding: {e}")

            best_match = None
            highest_sim = 0.0

            if face_emb is not None and registered_faces:
                import uniface
                for person in registered_faces:
                    reg_emb = np.array(person.get("embedding", []))
                    if len(reg_emb) == len(face_emb):
                        try:
                            sim = float(uniface.compute_similarity(face_emb, reg_emb))
                            if sim > highest_sim:
                                highest_sim = sim
                                best_match = person
                        except Exception:
                            pass

            is_registered = (best_match is not None and highest_sim >= similarity_threshold)

            if is_registered:
                matched_name = best_match["name"]
                identity_num = best_match.get("identity_number")
                color = (0, 200, 80)  # Emerald Green
                badge_text = f"✓ {matched_name} ({int(highest_sim * 100)}%) [LIVE]"
                status_str = "REGISTERED"
            else:
                matched_name = "Wajah Tidak Dikenal"
                identity_num = None
                color = (0, 165, 255)  # Amber / Orange
                badge_text = f"? Tidak Dikenal ({int(highest_sim * 100)}%) [LIVE]" if highest_sim > 0.20 else "? Tidak Dikenal [LIVE]"
                status_str = "UNKNOWN"

        orig_bbox = [
            int(x1 / scale),
            int(y1 / scale),
            int(x2 / scale),
            int(y2 / scale)
        ]

        recognized_list.append({
            "face_index": idx + 1,
            "status": status_str,
            "is_real": is_live,
            "liveness_score": liveness_score,
            "is_registered": is_registered,
            "name": matched_name,
            "identity_number": identity_num,
            "similarity": round(highest_sim, 3),
            "confidence": round(conf, 3),
            "bbox": orig_bbox
        })

        # Draw Bounding Box
        thickness = 3 if is_live else 4
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        # Draw Header Badge
        (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2)
        cv2.rectangle(annotated, (x1, max(0, y1 - th - 12)), (x1 + tw + 10, y1), color, -1)
        cv2.putText(annotated, badge_text, (x1 + 5, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)

    elapsed_ms = (time.time() - start_time) * 1000

    return {
        "success": True,
        "total_faces": len(faces),
        "total_registered_matched": sum(1 for f in recognized_list if f["is_registered"]),
        "total_unknown": sum(1 for f in recognized_list if (not f["is_registered"] and f["is_real"])),
        "total_spoof_attacks": total_spoof_attacks,
        "faces": recognized_list,
        "annotated_image_base64": encode_image_to_base64(annotated),
        "execution_time_ms": round(elapsed_ms, 2)
    }
