from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.schemas.ml import ModelInfo, PredictRequest, PredictResponse
from app.models.user import User
from app.services.auth_service import get_current_user
from app.services.ml_service import ModelInputError, ModelNotFoundError, list_available_models, predict_with_model

router = APIRouter(prefix="/ml", tags=["Machine Learning"])


@router.get("/models", response_model=List[ModelInfo])
async def api_get_models(db: AsyncSession = Depends(get_db), user: User = Depends(get_current_user)):
    """Lists registered ML models, task types, metrics, and required features."""
    return await list_available_models(db)


@router.post("/predict/{model_name}", response_model=PredictResponse)
async def api_predict(
    model_name: str,
    payload: PredictRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user)
):
    """
    Executes inference against a registered model (Scikit-Learn/Joblib).
    Returns class prediction, human-readable label, and probability distribution.
    """
    try:
        result = await predict_with_model(db, model_name=model_name, feature_inputs=payload.features)
        return result
    except ModelNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ModelInputError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Inference error: {str(e)}")
