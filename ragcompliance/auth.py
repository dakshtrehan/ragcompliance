"""
Optional OIDC SSO for the dashboard.

Gated entirely by env vars — with no RAGCOMPLIANCE_OIDC_ISSUER the module
is a no-op and the dashboard stays open for local dev. Works with any
standards-compliant OIDC provider via discovery (Google Workspace, Okta,
Auth0, Microsoft Entra, Authentik, etc.).

    RAGCOMPLIANCE_OIDC_ISSUER=https://accounts.google.com
    RAGCOMPLIANCE_OIDC_CLIENT_ID=...
    RAGCOMPLIANCE_OIDC_CLIENT_SECRET=...
    RAGCOMPLIANCE_OIDC_REDIRECT_URI=https://dash.example.com/auth/callback
    RAGCOMPLIANCE_OIDC_ALLOWED_DOMAINS=acme.com,acme.co.uk   # optional allowlist
    RAGCOMPLIANCE_SESSION_SECRET=<32+ char random string>

When enabled, everything except the public allowlist
(/health, /login, /auth/callback, /logout, /stripe/webhook) requires a
signed-in user. Browsers get redirected to /login; API callers get 401.
"""
from __future__ import annotations

import logging
import os
import secrets
from dataclasses import dataclass, field
from typing import Any

# fastapi + starlette are installed whenever the dashboard is, so importing
# them at module level here is safe. authlib is the only thing we keep lazy
# because it lives in the optional `sso` extra.
from fastapi import FastAPI, HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import JSONResponse, RedirectResponse

logger = logging.getLogger(__name__)

SESSION_USER_KEY = "rc_user"

# Paths that bypass auth. /stripe/webhook stays open because it verifies
# its own Stripe signature; /health is a liveness probe.
PUBLIC_PATHS: set[str] = {
    "/health",
    "/login",
    "/auth/callback",
    "/logout",
    "/stripe/webhook",
}


@dataclass
class SSOConfig:
    issuer: str = ""
    client_id: str = ""
    client_secret: str = ""
    redirect_uri: str = ""
    allowed_domains: list[str] = field(default_factory=list)
    session_secret: str = ""

    @classmethod
    def from_env(cls) -> "SSOConfig":
        domains_raw = os.getenv("RAGCOMPLIANCE_OIDC_ALLOWED_DOMAINS", "")
        return cls(
            issuer=os.getenv("RAGCOMPLIANCE_OIDC_ISSUER", ""),
            client_id=os.getenv("RAGCOMPLIANCE_OIDC_CLIENT_ID", ""),
            client_secret=os.getenv("RAGCOMPLIANCE_OIDC_CLIENT_SECRET", ""),
            redirect_uri=os.getenv("RAGCOMPLIANCE_OIDC_REDIRECT_URI", ""),
            allowed_domains=[
                d.strip().lower()
                for d in domains_raw.split(",")
                if d.strip()
            ],
            session_secret=os.getenv("RAGCOMPLIANCE_SESSION_SECRET", ""),
        )

    @property
    def enabled(self) -> bool:
        """SSO is on when the minimal OIDC quartet is set."""
        return bool(
            self.issuer
            and self.client_id
            and self.client_secret
            and self.redirect_uri
        )

    def domain_allowed(self, email: str) -> bool:
        if not self.allowed_domains:
            return True
        return email.split("@")[-1].lower() in self.allowed_domains


# ------------------------------------------------------------------ #
# Path matching                                                         #
# ------------------------------------------------------------------ #


def is_public_path(path: str) -> bool:
    if path in PUBLIC_PATHS:
        return True
    # Allow prefix match for subpaths (e.g. /auth/callback/... later).
    for p in PUBLIC_PATHS:
        if path.startswith(p + "/"):
            return True
    return False


# ------------------------------------------------------------------ #
# FastAPI wiring                                                        #
# ------------------------------------------------------------------ #


def init_sso(app: Any) -> SSOConfig:
    """Attach SSO middleware + auth routes to a FastAPI app.

    When SSO is not configured this returns a disabled SSOConfig and the
    app is left untouched so local dev stays frictionless.
    """
    cfg = SSOConfig.from_env()
    if not cfg.enabled:
        logger.info("ragcompliance SSO: disabled (OIDC env vars not set)")
        return cfg

    # In prod the operator must set RAGCOMPLIANCE_SESSION_SECRET so sessions
    # survive restart and rotate on demand. If missing, fall back to an
    # ephemeral secret and shout about it.
    if not cfg.session_secret:
        cfg.session_secret = secrets.token_urlsafe(32)
        logger.warning(
            "ragcompliance SSO: RAGCOMPLIANCE_SESSION_SECRET not set — using "
            "an ephemeral secret. Sessions will invalidate on every restart."
        )

    # authlib lives in the optional `sso` extra so load it lazily — that way
    # installing ragcompliance without `[sso]` is still fine as long as SSO
    # env vars are unset.
    from authlib.integrations.starlette_client import OAuth

    # OIDC via discovery. One call, one config.
    oauth = OAuth()
    oauth.register(
        name="oidc",
        client_id=cfg.client_id,
        client_secret=cfg.client_secret,
        server_metadata_url=(
            f"{cfg.issuer.rstrip('/')}/.well-known/openid-configuration"
        ),
        client_kwargs={"scope": "openid email profile"},
    )

    # ---- Middleware: gate all non-public paths --------------------- #

    class _RequireAuthMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            path = request.url.path
            if is_public_path(path):
                return await call_next(request)
            # Session middleware runs OUTSIDE of us, so request.session is
            # always populated by the time we get here.
            user = None
            try:
                user = request.session.get(SESSION_USER_KEY)
            except Exception:
                user = None
            if user:
                return await call_next(request)
            # Browsers → redirect to /login. API clients → 401.
            accept = request.headers.get("accept", "")
            if "text/html" in accept:
                return RedirectResponse(url="/login", status_code=302)
            return JSONResponse(
                {"detail": "authentication required"}, status_code=401
            )

    # Order matters: middleware added LAST runs FIRST on the request path.
    # We want SessionMiddleware to populate request.session before the
    # auth gate reads it, so add SessionMiddleware second.
    app.add_middleware(_RequireAuthMiddleware)
    app.add_middleware(
        SessionMiddleware,
        secret_key=cfg.session_secret,
        session_cookie="rc_session",
        same_site="lax",
        https_only=cfg.redirect_uri.startswith("https://"),
    )

    # ---- Routes ----------------------------------------------------- #

    @app.get("/login")
    async def login(request: Request):
        return await oauth.oidc.authorize_redirect(request, cfg.redirect_uri)

    @app.get("/auth/callback")
    async def auth_callback(request: Request):
        try:
            token = await oauth.oidc.authorize_access_token(request)
        except Exception as e:
            logger.warning(f"ragcompliance SSO: callback failed — {e}")
            raise HTTPException(status_code=400, detail="authentication failed")

        userinfo = token.get("userinfo") or {}
        email = (userinfo.get("email") or "").lower()
        if not email:
            raise HTTPException(
                status_code=400,
                detail="OIDC provider did not return an email claim",
            )
        if not cfg.domain_allowed(email):
            raise HTTPException(
                status_code=403,
                detail=f"{email.split('@')[-1]} is not on the allowed domains list",
            )

        request.session[SESSION_USER_KEY] = {
            "email": email,
            "name": userinfo.get("name") or email,
            "sub": userinfo.get("sub") or "",
        }
        return RedirectResponse(url="/", status_code=302)

    @app.get("/logout")
    async def logout(request: Request):
        request.session.pop(SESSION_USER_KEY, None)
        return RedirectResponse(url="/login", status_code=302)

    logger.info(f"ragcompliance SSO: enabled, issuer={cfg.issuer}")
    return cfg
