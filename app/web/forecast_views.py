from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse

from app.services.auth_service import get_current_user_optional
from app.services.market_forecast_service import TIMEFRAMES, get_forecast, get_history
from app.web.templates import render_template

router = APIRouter(prefix="/grill-me", tags=["Market Forecast"])


@router.get("", response_class=HTMLResponse)
async def grill_me_page(request: Request, user=Depends(get_current_user_optional)):
    return render_template(request, "forecast/index.html", {
        "user": user, "timeframes": list(TIMEFRAMES),
    })


@router.get("/data")
def grill_me_data(symbol: str = "BTCUSD", timeframe: str = "M1"):
    try:
        return {"ok": True, "data": get_forecast(symbol.strip().upper(), timeframe)}
    except (RuntimeError, ValueError) as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=503)


@router.get("/history")
def grill_me_history(symbol: str = "", timeframe: str = ""):
    return get_history(symbol.strip().upper() or None, timeframe.upper() or None)
