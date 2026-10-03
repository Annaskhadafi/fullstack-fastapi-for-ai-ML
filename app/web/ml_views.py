from fastapi import APIRouter, Request, Form, Depends
from fastapi.responses import HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.services.auth_service import get_current_user_optional
from app.services.ml_service import list_available_models, predict_with_model
from app.web.templates import render_template

router = APIRouter(prefix="/ml", tags=["Web ML"])


@router.get("", response_class=HTMLResponse)
async def ml_studio_page(
    request: Request,
    user=Depends(get_current_user_optional),
    db: AsyncSession = Depends(get_db)
):
    models = await list_available_models(db)
    active_model = models[0] if models else None

    return render_template(request, "ml/index.html", {
        "user": user,
        "models": models,
        "active_model": active_model
    })


@router.get("/form/{model_name}", response_class=HTMLResponse)
async def ml_form_htmx(
    request: Request,
    model_name: str,
    db: AsyncSession = Depends(get_db)
):
    models = await list_available_models(db)
    active_model = next((m for m in models if m.name == model_name), None)

    return render_template(request, "ml/partials/model_form.html", {
        "active_model": active_model
    })


@router.post("/predict-htmx", response_class=HTMLResponse)
async def ml_predict_htmx(
    request: Request,
    model_name: str = Form(...),
    db: AsyncSession = Depends(get_db)
):
    form_data = await request.form()
    feature_inputs = {}

    for k, v in form_data.items():
        if k != "model_name":
            feature_inputs[k] = v

    try:
        result = await predict_with_model(db, model_name=model_name, feature_inputs=feature_inputs)
        return render_template(request, "ml/partials/prediction_card.html", {
            "result": result
        })
    except Exception as e:
        return HTMLResponse(
            f"""<div class="alert alert-error"><span>Prediksi gagal: {str(e)}</span></div>""",
            status_code=500
        )
