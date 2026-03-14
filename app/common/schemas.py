"""
Pydantic schemas for request/response validation across all API routes.
"""

from __future__ import annotations
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


class LoginRequest(BaseModel):
    username: str
    password: str


class SignupRequest(BaseModel):
    username: str
    password: str


class AuthResponse(BaseModel):
    accessToken: str
    refreshToken: str
    message: str
    # Extra metadata the UI needs on login
    role: Optional[str] = None
    is_default_admin: Optional[bool] = None
    must_change_password: Optional[bool] = None


class RefreshRequest(BaseModel):
    refreshToken: str


class RefreshResponse(BaseModel):
    accessToken: str


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


# ---------------------------------------------------------------------------
# Admin – User management
# ---------------------------------------------------------------------------


class CreateUserRequest(BaseModel):
    username: str
    password: str
    role: str = "user"  # "admin" or "user"
    display_name: Optional[str] = ""


class UserOut(BaseModel):
    id: int
    username: str
    role: str
    display_name: Optional[str]
    is_default_admin: bool
    must_change_password: bool
    created_at: Optional[datetime]

    class Config:
        from_attributes = True


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


class ProviderCreate(BaseModel):
    name: str
    display_name: str
    base_url: Optional[str] = None
    provider_type: str  # openai | gemini | ollama


class ProviderUpdate(BaseModel):
    display_name: Optional[str] = None
    base_url: Optional[str] = None
    is_active: Optional[bool] = None


class ProviderOut(BaseModel):
    id: int
    name: str
    display_name: str
    base_url: Optional[str]
    provider_type: str
    is_active: bool
    created_at: Optional[datetime]

    class Config:
        from_attributes = True


# ---------------------------------------------------------------------------
# Provider API Keys
# ---------------------------------------------------------------------------


class ProviderApiKeyCreate(BaseModel):
    provider_id: int
    label: str = "default"
    api_key: str  # plaintext — will be encrypted before storage


class ProviderApiKeyOut(BaseModel):
    id: int
    provider_id: int
    label: str
    is_active: bool
    created_at: Optional[datetime]
    # Never expose the actual key — only metadata

    class Config:
        from_attributes = True


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class LLMModelCreate(BaseModel):
    provider_id: int
    model_id: str
    display_name: str
    max_context_tokens: Optional[int] = None


class LLMModelUpdate(BaseModel):
    display_name: Optional[str] = None
    max_context_tokens: Optional[int] = None
    is_active: Optional[bool] = None


class LLMModelOut(BaseModel):
    id: int
    provider_id: int
    model_id: str
    display_name: str
    max_context_tokens: Optional[int]
    is_active: bool
    provider_name: Optional[str] = None  # populated manually

    class Config:
        from_attributes = True


# ---------------------------------------------------------------------------
# Gateway API Keys (user-facing)
# ---------------------------------------------------------------------------


class GatewayApiKeyCreate(BaseModel):
    label: str = "default"


class GatewayApiKeyOut(BaseModel):
    id: int
    key_prefix: str
    label: str
    is_active: bool
    created_at: Optional[datetime]
    last_used_at: Optional[datetime]
    permissions: List[ApiKeyPermissionOut] = []

    class Config:
        from_attributes = True


class GatewayApiKeyCreated(BaseModel):
    """Returned only once when the key is first created — contains the raw key."""

    id: int
    key: str  # the raw key — shown only once
    key_prefix: str
    label: str


class ApiKeyPermissionOut(BaseModel):
    id: int
    model_id: int
    model_display_name: Optional[str] = None
    provider_name: Optional[str] = None
    max_input_tokens: Optional[int]
    max_output_tokens: Optional[int]
    is_active: bool

    class Config:
        from_attributes = True


# Fix forward reference
GatewayApiKeyOut.model_rebuild()


# ---------------------------------------------------------------------------
# Permission Requests
# ---------------------------------------------------------------------------


class PermissionRequestCreate(BaseModel):
    api_key_id: int
    model_id: int
    request_message: Optional[str] = None


class PermissionRequestReview(BaseModel):
    status: str  # "approved" or "rejected"
    admin_message: Optional[str] = None
    # Optional overrides set by admin when approving
    max_input_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None


class PermissionRequestOut(BaseModel):
    id: int
    user_id: int
    username: Optional[str] = None
    api_key_id: int
    api_key_label: Optional[str] = None
    model_id: int
    model_display_name: Optional[str] = None
    provider_name: Optional[str] = None
    status: str
    request_message: Optional[str]
    admin_message: Optional[str]
    reviewed_by: Optional[int]
    created_at: Optional[datetime]

    class Config:
        from_attributes = True


# ---------------------------------------------------------------------------
# Environment Variables (admin-only)
# ---------------------------------------------------------------------------


class EnvVarCreate(BaseModel):
    key: str
    value: str
    description: Optional[str] = None
    is_secret: bool = True


class EnvVarUpdate(BaseModel):
    value: Optional[str] = None
    description: Optional[str] = None
    is_secret: Optional[bool] = None


class EnvVarOut(BaseModel):
    id: int
    key: str
    # value is only returned for non-secret vars; secret vars show "***"
    value: Optional[str] = None
    description: Optional[str]
    is_secret: bool
    created_at: Optional[datetime]
    updated_at: Optional[datetime]

    class Config:
        from_attributes = True


# ---------------------------------------------------------------------------
# Chat (the main LLM gateway endpoint)
# ---------------------------------------------------------------------------


class ChatConfig(BaseModel):
    """
    Flexible configuration dict for an LLM call.
    The caller specifies what model to use and optional generation params.
    """

    # Which model to call — use the model_id string (e.g. "gpt-4o", "gemini-1.5-pro")
    model: str
    # Optional overrides
    temperature: Optional[float] = None
    max_output_tokens: Optional[int] = None
    top_p: Optional[float] = None
    # Any additional provider-specific params the caller wants to pass through
    extra: Optional[dict] = None


class ChatRequest(BaseModel):
    """
    Input to the /chat endpoint.
    - system_prompt: system-level instruction
    - user_prompt: the user's message
    - image_base64: optional base64-encoded image (for vision models)
    - config: model + generation parameters
    """

    system_prompt: Optional[str] = None
    user_prompt: str
    image_base64: Optional[str] = None
    image_media_type: Optional[str] = "image/png"  # mime type if image is provided
    config: ChatConfig


class ChatResponse(BaseModel):
    """Output from the /chat endpoint."""

    content: str
    model: str
    provider: str
    usage: Optional[dict] = None  # token usage info if available


# ---------------------------------------------------------------------------
# Admin – direct permission grant (admin adds permission to a key directly)
# ---------------------------------------------------------------------------


class AdminGrantPermission(BaseModel):
    api_key_id: int
    model_id: int
    max_input_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None
