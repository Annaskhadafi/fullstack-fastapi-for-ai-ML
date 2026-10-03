import os
from typing import Optional, Dict, Any
from fastapi import Request
from fastapi.templating import Jinja2Templates

current_dir = os.path.dirname(os.path.abspath(__file__))
templates_dir = os.path.join(os.path.dirname(current_dir), "templates")

templates = Jinja2Templates(directory=templates_dir)


def render_template(
    request: Request,
    name: str,
    context: Optional[Dict[str, Any]] = None,
    status_code: int = 200
):
    """
    Universal template renderer compatible with both new Starlette (0.38+)
    and older Starlette versions.
    """
    ctx = dict(context or {})
    ctx["request"] = request
    try:
        return templates.TemplateResponse(
            request=request,
            name=name,
            context=ctx,
            status_code=status_code
        )
    except TypeError:
        return templates.TemplateResponse(
            name,
            ctx,
            status_code=status_code
        )
