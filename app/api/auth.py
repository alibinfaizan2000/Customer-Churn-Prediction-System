"""
auth.py — API Key authentication for the Churn Prediction API.

WHY API KEY AUTH (not JWT for this project):
- JWT is for user-facing apps with login/logout sessions
- API keys are for service-to-service communication — exactly what an ML API is
- Simpler to implement, rotate, and audit
- Industry standard for ML APIs (OpenAI, Anthropic, HuggingFace all use API keys)

HOW IT WORKS:
1. Keys are stored in a dict (in production: database or secrets manager)
2. Every protected request must include header: X-API-Key: <key>
3. FastAPI's Depends() system injects the check automatically
4. /health stays public — load balancers need it without auth

KEY ROLES:
- admin  → can call all endpoints including /model/info and /monitoring/stats
- service → can call /predict and /batch_predict (for internal services)
- readonly → can only call /health and /model/info (for dashboards)

INTERVIEW TIP:
"In production I'd store hashed keys in PostgreSQL, rotate them via a
/admin/rotate-key endpoint, and log every key usage for audit trails.
For secrets, I'd use AWS Secrets Manager or HashiCorp Vault — never
hardcode keys in source code."
"""

import os
import logging
import hashlib
import secrets
from datetime import datetime, timezone
from typing import Optional

from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

logger = logging.getLogger(__name__)

# ─── API Key header definition ────────────────────────────────────────────────
# FastAPI reads this header from every incoming request.
# auto_error=False means we handle the missing-key case ourselves
# (gives a cleaner error message than FastAPI's default).
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


# ─── Key store ────────────────────────────────────────────────────────────────
# WHY NOT HARDCODE IN SOURCE: Keys in source code get committed to git,
# leaked in logs, and exposed in Docker image layers.
# We read from environment variables — the correct production pattern.
#
# In production this would be:
#   SELECT * FROM api_keys WHERE key_hash = sha256(incoming_key) AND active = true
#
# Key format: role:description → actual key value from env
_KEY_STORE: dict[str, dict] = {
    # Admin key — full access
    os.getenv("ADMIN_API_KEY", "churn-admin-key-change-in-production"): {
        "role":        "admin",
        "description": "Admin key — full access",
        "created_at":  "2024-01-01",
    },
    # Service key — prediction endpoints only
    os.getenv("SERVICE_API_KEY", "churn-service-key-change-in-production"): {
        "role":        "service",
        "description": "Service key — prediction endpoints",
        "created_at":  "2024-01-01",
    },
    # Readonly key — health and model info only
    os.getenv("READONLY_API_KEY", "churn-readonly-key-change-in-production"): {
        "role":        "readonly",
        "description": "Readonly key — monitoring dashboards",
        "created_at":  "2024-01-01",
    },
}

# Role permission matrix — what each role can access
# WHY EXPLICIT MATRIX: Adding a new role or endpoint is one line change,
# not scattered if/else logic across the codebase.
ROLE_PERMISSIONS: dict[str, set[str]] = {
    "admin":    {"predict", "batch_predict", "model_info", "monitoring", "health"},
    "service":  {"predict", "batch_predict", "health"},
    "readonly": {"health", "model_info"},
}


# ─── Auth models ──────────────────────────────────────────────────────────────

class APIKeyInfo:
    """Holds metadata about a validated API key."""
    def __init__(self, key: str, role: str, description: str):
        self.key         = key
        self.role        = role
        self.description = description
        self.validated_at = datetime.now(timezone.utc).isoformat()

    def can(self, permission: str) -> bool:
        """Check if this key's role has a specific permission."""
        return permission in ROLE_PERMISSIONS.get(self.role, set())

    def __repr__(self):
        return f"APIKeyInfo(role={self.role})"


# ─── Core validation ──────────────────────────────────────────────────────────

def _validate_key(raw_key: str) -> Optional[APIKeyInfo]:
    """
    Validate an API key and return its metadata.

    WHY CONSTANT-TIME COMPARISON (secrets.compare_digest):
    Regular string comparison (==) short-circuits on the first differing
    character. An attacker can time many requests to guess keys character
    by character (timing attack). secrets.compare_digest always takes the
    same time regardless of where strings differ — immune to timing attacks.
    """
    for stored_key, metadata in _KEY_STORE.items():
        if secrets.compare_digest(raw_key.strip(), stored_key.strip()):
            return APIKeyInfo(
                key=stored_key,
                role=metadata["role"],
                description=metadata["description"]
            )
    return None


# ─── FastAPI dependency functions ─────────────────────────────────────────────
# These are injected via Depends() in endpoint signatures.
# FastAPI calls them automatically before the endpoint handler runs.

async def get_api_key(api_key: str = Security(API_KEY_HEADER)) -> APIKeyInfo:
    """
    Base dependency — validates any API key.
    Raises 401 if missing, 403 if invalid.

    WHY 401 vs 403:
    - 401 Unauthorized: no credentials provided
    - 403 Forbidden: credentials provided but insufficient permissions
    This distinction matters for client error handling.
    """
    if not api_key:
        logger.warning("Request rejected: no API key provided")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required. Include header: X-API-Key: <your-key>",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    key_info = _validate_key(api_key)
    if key_info is None:
        # Log the attempt (not the key itself — never log secrets)
        logger.warning(f"Request rejected: invalid API key (prefix: {api_key[:8]}...)")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )

    logger.debug(f"Authenticated: role={key_info.role}")
    return key_info


async def require_predict_permission(
    key_info: APIKeyInfo = Security(get_api_key)
) -> APIKeyInfo:
    """
    Dependency for prediction endpoints.
    Requires 'predict' permission (admin or service role).
    """
    if not key_info.can("predict"):
        logger.warning(f"Permission denied: role={key_info.role} tried predict")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Role '{key_info.role}' does not have predict permission.",
        )
    return key_info


async def require_admin_permission(
    key_info: APIKeyInfo = Security(get_api_key)
) -> APIKeyInfo:
    """
    Dependency for admin-only endpoints.
    Requires 'monitoring' permission (admin role only).
    """
    if not key_info.can("monitoring"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Role '{key_info.role}' does not have admin permission.",
        )
    return key_info


# ─── Key generation utility ───────────────────────────────────────────────────

def generate_api_key(prefix: str = "churn") -> str:
    """
    Generate a cryptographically secure API key.

    Format: churn-<32 random hex chars>
    Example: churn-a3f9b2c1d4e5f6a7b8c9d0e1f2a3b4c5

    WHY secrets.token_hex: Uses os.urandom() internally — cryptographically
    secure, not predictable like random.random(). NEVER use random for keys.

    Usage:
        python -c "from app.api.auth import generate_api_key; print(generate_api_key())"
    """
    return f"{prefix}-{secrets.token_hex(16)}"


if __name__ == "__main__":
    print("Generated API keys (save these somewhere safe):")
    print(f"  ADMIN_API_KEY   = {generate_api_key('churn-admin')}")
    print(f"  SERVICE_API_KEY = {generate_api_key('churn-svc')}")
    print(f"  READONLY_API_KEY= {generate_api_key('churn-ro')}")
    print("\nSet these as environment variables or in your .env file.")
