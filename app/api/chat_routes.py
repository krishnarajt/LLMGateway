"""
Chat routes — the main LLM gateway endpoints.

Authentication is via X-API-Key header (gateway API key), NOT via JWT.
This is by design: external applications call these endpoints with just an API key.

Endpoints:
  POST /chat  — non-streaming chat completion
"""

from fastapi import APIRouter, Depends, HTTPException, Header
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.common.schemas import ChatRequest, ChatResponse
from app.services.chat_service import execute_chat, ChatServiceError
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    x_api_key: str = Header(..., alias="X-API-Key", description="Your gateway API key"),
    db: Session = Depends(get_db),
):
    """
    Non-streaming chat completion.

    Send a system prompt, user prompt (and optionally a base64 image),
    along with a config specifying which model to use.

    Authentication: pass your gateway API key in the X-API-Key header.
    """
    try:
        result = execute_chat(
            db=db,
            raw_api_key=x_api_key,
            system_prompt=request.system_prompt,
            user_prompt=request.user_prompt,
            image_base64=request.image_base64,
            image_media_type=request.image_media_type or "image/png",
            config=request.config,
        )
        return ChatResponse(**result)

    except ChatServiceError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"LLM call failed: {str(e)}")
