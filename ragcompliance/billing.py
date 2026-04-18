"""
RAGCompliance billing — Stripe Checkout + subscription lifecycle + usage metering.

All Stripe interactions go through the `BillingManager`. Subscription state and
per-workspace usage counters are persisted in Supabase in the
`workspace_subscriptions` table (see supabase_migration_billing.sql).

Environment variables consumed (see .env.example):
    STRIPE_SECRET_KEY
    STRIPE_WEBHOOK_SECRET
    STRIPE_PRICE_ID_TEAM
    STRIPE_PRICE_ID_ENTERPRISE
    APP_BASE_URL
    RAGCOMPLIANCE_SUPABASE_URL
    RAGCOMPLIANCE_SUPABASE_KEY
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------

def _plans() -> dict[str, dict[str, Any]]:
    """Built lazily so env var changes during tests are picked up."""
    return {
        "team": {
            "name": "Team",
            "price_usd": 49,
            "query_limit": 10_000,
            "stripe_price_id": os.getenv("STRIPE_PRICE_ID_TEAM", ""),
            "description": "10K audited RAG queries per month, email support, CSV/JSON export.",
            "features": [
                "10,000 audited queries / month",
                "SHA-256 chain signatures",
                "Per-workspace row-level security",
                "CSV and JSON export",
                "Email support",
            ],
        },
        "enterprise": {
            "name": "Enterprise",
            "price_usd": 199,
            "query_limit": None,  # unlimited
            "stripe_price_id": os.getenv("STRIPE_PRICE_ID_ENTERPRISE", ""),
            "description": "Unlimited queries, SSO, priority support, export, custom retention.",
            "features": [
                "Unlimited audited queries",
                "SSO (SAML / OIDC)",
                "Priority support + Slack channel",
                "CSV and JSON export",
                "Custom retention policy",
                "SOC 2 report (on request)",
            ],
        },
    }


PLANS: dict[str, dict[str, Any]] = _plans()


def refresh_plans() -> None:
    """Re-read price IDs from env (useful in tests)."""
    global PLANS
    PLANS = _plans()


# ---------------------------------------------------------------------------
# Subscription record
# ---------------------------------------------------------------------------

@dataclass
class WorkspaceSubscription:
    workspace_id: str
    stripe_customer_id: str | None = None
    stripe_subscription_id: str | None = None
    tier: str = "free"
    status: str = "inactive"
    current_period_end: datetime | None = None
    query_count_current_period: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "WorkspaceSubscription":
        def _parse(v: Any) -> datetime | None:
            if not v:
                return None
            if isinstance(v, datetime):
                return v
            s = str(v).replace("Z", "+00:00")
            # PostgREST emits variable microsecond precision (e.g. 5 digits).
            # Python 3.10's fromisoformat requires 0, 3, or 6 digits of us, so
            # normalize the fractional-second part to 6 digits before parsing.
            try:
                return datetime.fromisoformat(s)
            except ValueError:
                import re
                m = re.match(r"^(.*?\.)(\d+)(.*)$", s)
                if m:
                    prefix, frac, tail = m.groups()
                    frac = (frac + "000000")[:6]
                    s = f"{prefix}{frac}{tail}"
                    try:
                        return datetime.fromisoformat(s)
                    except ValueError:
                        return None
                return None

        return cls(
            workspace_id=row["workspace_id"],
            stripe_customer_id=row.get("stripe_customer_id"),
            stripe_subscription_id=row.get("stripe_subscription_id"),
            tier=row.get("tier", "free"),
            status=row.get("status", "inactive"),
            current_period_end=_parse(row.get("current_period_end")),
            query_count_current_period=int(row.get("query_count_current_period") or 0),
            created_at=_parse(row.get("created_at")) or datetime.now(timezone.utc),
            updated_at=_parse(row.get("updated_at")) or datetime.now(timezone.utc),
        )

    def to_row(self) -> dict[str, Any]:
        return {
            "workspace_id": self.workspace_id,
            "stripe_customer_id": self.stripe_customer_id,
            "stripe_subscription_id": self.stripe_subscription_id,
            "tier": self.tier,
            "status": self.status,
            "current_period_end": self.current_period_end.isoformat()
            if self.current_period_end
            else None,
            "query_count_current_period": self.query_count_current_period,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }


# ---------------------------------------------------------------------------
# Live-mode readiness
# ---------------------------------------------------------------------------


@dataclass
class BillingReadiness:
    """Pre-flight status of the Stripe + Supabase billing path."""

    mode: str                 # "live" | "test" | "unconfigured"
    ok: bool                  # True when `mode` is live or test and nothing's broken
    issues: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "ok": self.ok,
            "issues": self.issues,
            "summary": self.summary,
        }


def _detect_mode(secret_key: str) -> str:
    if not secret_key:
        return "unconfigured"
    if secret_key.startswith("sk_live_"):
        return "live"
    if secret_key.startswith("sk_test_"):
        return "test"
    # Restricted keys (rk_live_ / rk_test_) and custom keys are still valid.
    if secret_key.startswith(("rk_live_", "whsec_live_")):
        return "live"
    if secret_key.startswith(("rk_test_", "whsec_test_")):
        return "test"
    return "unconfigured"


# ---------------------------------------------------------------------------
# BillingManager
# ---------------------------------------------------------------------------

class BillingManager:
    """
    Stripe wrapper + Supabase-backed subscription state.

    Safe to instantiate without real credentials for tests — every method will
    return a sensible no-op and log a warning if Stripe or Supabase isn't
    configured.
    """

    TABLE_NAME = "workspace_subscriptions"

    def __init__(
        self,
        stripe_secret_key: str = "",
        stripe_webhook_secret: str = "",
        supabase_url: str = "",
        supabase_key: str = "",
        app_base_url: str = "http://localhost:8000",
    ):
        self.stripe_secret_key = stripe_secret_key
        self.stripe_webhook_secret = stripe_webhook_secret
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        self.app_base_url = app_base_url.rstrip("/")

        self._stripe = None
        self._supabase = None

        if stripe_secret_key:
            try:
                import stripe  # type: ignore
                stripe.api_key = stripe_secret_key
                self._stripe = stripe
            except ImportError:
                logger.warning("stripe package not installed — run `pip install stripe`")

        if supabase_url and supabase_key:
            try:
                from supabase import create_client  # type: ignore
                self._supabase = create_client(supabase_url, supabase_key)
            except ImportError:
                logger.warning(
                    "supabase package not installed — run `pip install ragcompliance[supabase]`"
                )

    # ------------------------------------------------------------------ #
    # Constructors                                                         #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_env(cls) -> "BillingManager":
        return cls(
            stripe_secret_key=os.getenv("STRIPE_SECRET_KEY", ""),
            stripe_webhook_secret=os.getenv("STRIPE_WEBHOOK_SECRET", ""),
            supabase_url=os.getenv("RAGCOMPLIANCE_SUPABASE_URL", ""),
            supabase_key=os.getenv("RAGCOMPLIANCE_SUPABASE_KEY", ""),
            app_base_url=os.getenv("APP_BASE_URL", "http://localhost:8000"),
        )

    # ------------------------------------------------------------------ #
    # Readiness                                                            #
    # ------------------------------------------------------------------ #

    def readiness(self) -> BillingReadiness:
        """Pre-flight the billing path. Returns mode + list of issues.

        Intended to be called at startup and exposed through a /health/billing
        endpoint so operators catch live-mode misconfig before the first
        customer hits checkout. Never throws — always returns a summary.
        """
        mode = _detect_mode(self.stripe_secret_key)
        issues: list[str] = []

        if mode == "unconfigured":
            issues.append("STRIPE_SECRET_KEY is not set")
        if not self.stripe_webhook_secret:
            issues.append("STRIPE_WEBHOOK_SECRET is not set")
        if not self.supabase_url or not self.supabase_key:
            issues.append(
                "Supabase is not configured (RAGCOMPLIANCE_SUPABASE_URL / _KEY) — "
                "subscription state cannot be persisted"
            )
        # Price IDs use `price_...` (never `prod_...`) — guard against the
        # common mistake of pasting a product ID into STRIPE_PRICE_ID_*.
        refresh_plans()
        for tier, plan in PLANS.items():
            pid = plan.get("stripe_price_id", "")
            if not pid:
                issues.append(f"STRIPE_PRICE_ID_{tier.upper()} is not set")
                continue
            if not pid.startswith("price_"):
                issues.append(
                    f"STRIPE_PRICE_ID_{tier.upper()}={pid[:10]}… looks wrong "
                    "(expected prefix 'price_'; did you paste a product ID?)"
                )
        if not self.app_base_url or self.app_base_url in ("http://localhost:8000",):
            if mode == "live":
                issues.append(
                    "APP_BASE_URL is localhost but you're in live mode — Stripe "
                    "success/cancel redirects and webhook callbacks will not work"
                )

        # Sanitized summary — never include the actual secrets.
        summary: dict[str, Any] = {
            "secret_key_prefix": self.stripe_secret_key[:7] + "…"
            if self.stripe_secret_key
            else "",
            "webhook_secret_set": bool(self.stripe_webhook_secret),
            "supabase_configured": bool(self._supabase),
            "app_base_url": self.app_base_url,
            "price_ids": {
                tier: (PLANS[tier].get("stripe_price_id") or "")[:10] + "…"
                if PLANS[tier].get("stripe_price_id")
                else ""
                for tier in PLANS
            },
        }
        return BillingReadiness(
            mode=mode,
            ok=(mode in ("live", "test")) and not issues,
            issues=issues,
            summary=summary,
        )

    # ------------------------------------------------------------------ #
    # Checkout                                                             #
    # ------------------------------------------------------------------ #

    def create_checkout_session(
        self,
        workspace_id: str,
        tier: str,
        success_url: str | None = None,
        cancel_url: str | None = None,
    ) -> str:
        """Returns a Stripe Checkout URL for the given tier."""
        if self._stripe is None:
            raise RuntimeError("Stripe not configured — set STRIPE_SECRET_KEY")

        if tier not in PLANS:
            raise ValueError(f"Unknown tier {tier!r}. Choices: {list(PLANS)}")

        price_id = PLANS[tier]["stripe_price_id"]
        if not price_id:
            raise RuntimeError(
                f"STRIPE_PRICE_ID_{tier.upper()} is not set in env"
            )

        success_url = success_url or f"{self.app_base_url}/billing/success?session_id={{CHECKOUT_SESSION_ID}}"
        cancel_url = cancel_url or f"{self.app_base_url}/billing/cancel"

        session = self._stripe.checkout.Session.create(
            mode="subscription",
            line_items=[{"price": price_id, "quantity": 1}],
            success_url=success_url,
            cancel_url=cancel_url,
            client_reference_id=workspace_id,
            metadata={"workspace_id": workspace_id, "tier": tier},
            subscription_data={"metadata": {"workspace_id": workspace_id, "tier": tier}},
        )
        return session.url

    # ------------------------------------------------------------------ #
    # Webhook routing                                                      #
    # ------------------------------------------------------------------ #

    def handle_webhook(self, payload: bytes | str, signature: str) -> dict[str, Any]:
        """Verify signature and dispatch. Returns a summary dict."""
        if self._stripe is None:
            raise RuntimeError("Stripe not configured")
        if not self.stripe_webhook_secret:
            raise RuntimeError("STRIPE_WEBHOOK_SECRET not set")

        try:
            event = self._stripe.Webhook.construct_event(
                payload, signature, self.stripe_webhook_secret
            )
        except Exception as e:
            logger.error(f"Stripe webhook signature verification failed: {e}")
            raise

        event_type = event["type"]
        raw_object = event["data"]["object"]
        # Stripe SDK 12.x StripeObject.__getattr__ intercepts .get() calls and
        # raises AttributeError('get'). Convert to a plain nested dict up-front
        # so handlers can use normal dict semantics.
        if hasattr(raw_object, "to_dict_recursive"):
            data = raw_object.to_dict_recursive()
        elif hasattr(raw_object, "to_dict"):
            data = raw_object.to_dict()
        else:
            data = dict(raw_object)
        logger.info(f"Stripe webhook: {event_type}")

        if event_type == "checkout.session.completed":
            return self._on_checkout_completed(data)
        if event_type == "customer.subscription.updated":
            return self._on_subscription_updated(data)
        if event_type == "customer.subscription.deleted":
            return self._on_subscription_deleted(data)
        if event_type == "invoice.payment_failed":
            return self._on_payment_failed(data)

        return {"handled": False, "event_type": event_type}

    def _on_checkout_completed(self, session: dict[str, Any]) -> dict[str, Any]:
        workspace_id = (
            session.get("client_reference_id")
            or (session.get("metadata") or {}).get("workspace_id")
        )
        tier = (session.get("metadata") or {}).get("tier", "team")
        if not workspace_id:
            logger.warning("checkout.session.completed without workspace_id")
            return {"handled": False, "reason": "no workspace_id"}

        sub = self._load_or_new(workspace_id)
        sub.stripe_customer_id = session.get("customer")
        sub.stripe_subscription_id = session.get("subscription")
        sub.tier = tier
        sub.status = "active"
        sub.query_count_current_period = 0
        self._upsert(sub)
        return {"handled": True, "workspace_id": workspace_id, "tier": tier, "status": "active"}

    def _on_subscription_updated(self, subscription: dict[str, Any]) -> dict[str, Any]:
        workspace_id = (subscription.get("metadata") or {}).get("workspace_id")
        if not workspace_id:
            return {"handled": False, "reason": "no workspace_id"}

        sub = self._load_or_new(workspace_id)
        sub.stripe_subscription_id = subscription.get("id")
        sub.status = subscription.get("status", sub.status)
        # Period rollover detection: if the incoming current_period_end is
        # strictly later than the stored one, we crossed into a new billing
        # cycle and the query counter must reset to 0. This is the primary
        # mechanism; check_query_quota() also has a self-heal fallback for
        # the case where this webhook never arrives.
        period_rolled_over = False
        cpe = subscription.get("current_period_end")
        if cpe:
            new_cpe = datetime.fromtimestamp(cpe, tz=timezone.utc)
            if sub.current_period_end is None or new_cpe > sub.current_period_end:
                period_rolled_over = sub.current_period_end is not None
            sub.current_period_end = new_cpe
        if period_rolled_over:
            logger.info(
                f"Billing period rolled over for {workspace_id!r}; "
                f"resetting query_count_current_period from "
                f"{sub.query_count_current_period} to 0."
            )
            sub.query_count_current_period = 0
        # If price changed, reflect tier
        items = ((subscription.get("items") or {}).get("data") or [])
        if items:
            price_id = (items[0].get("price") or {}).get("id")
            for tier_name, plan in PLANS.items():
                if plan["stripe_price_id"] == price_id:
                    sub.tier = tier_name
                    break
        self._upsert(sub)
        return {
            "handled": True,
            "workspace_id": workspace_id,
            "tier": sub.tier,
            "status": sub.status,
            "period_rolled_over": period_rolled_over,
        }

    def _on_subscription_deleted(self, subscription: dict[str, Any]) -> dict[str, Any]:
        workspace_id = (subscription.get("metadata") or {}).get("workspace_id")
        if not workspace_id:
            return {"handled": False, "reason": "no workspace_id"}

        sub = self._load_or_new(workspace_id)
        sub.status = "canceled"
        sub.tier = "free"
        self._upsert(sub)
        return {"handled": True, "workspace_id": workspace_id, "status": "canceled"}

    def _on_payment_failed(self, invoice: dict[str, Any]) -> dict[str, Any]:
        customer_id = invoice.get("customer")
        logger.warning(f"invoice.payment_failed for customer {customer_id}")
        if self._supabase and customer_id:
            try:
                rows = (
                    self._supabase.table(self.TABLE_NAME)
                    .select("*")
                    .eq("stripe_customer_id", customer_id)
                    .execute()
                    .data
                    or []
                )
                for row in rows:
                    sub = WorkspaceSubscription.from_row(row)
                    sub.status = "past_due"
                    self._upsert(sub)
            except Exception as e:
                logger.error(f"Failed to flag past_due: {e}")
        return {"handled": True, "event": "payment_failed", "customer_id": customer_id}

    # ------------------------------------------------------------------ #
    # Subscription queries                                                 #
    # ------------------------------------------------------------------ #

    def get_workspace_subscription(self, workspace_id: str) -> dict[str, Any] | None:
        if self._supabase is None:
            return None
        try:
            res = (
                self._supabase.table(self.TABLE_NAME)
                .select("*")
                .eq("workspace_id", workspace_id)
                .limit(1)
                .execute()
            )
            if not res.data:
                return None
            return res.data[0]
        except Exception as e:
            logger.error(f"get_workspace_subscription failed: {e}")
            return None

    # ------------------------------------------------------------------ #
    # Quota + usage                                                        #
    # ------------------------------------------------------------------ #

    def check_query_quota(self, workspace_id: str) -> bool:
        """
        Returns True if the workspace is within its tier limit (or has no sub),
        False only when an active paid tier is over its limit.

        Also auto-resets the usage counter when the stored current_period_end
        is in the past — a self-healing fallback in case the Stripe
        customer.subscription.updated webhook is delayed or dropped. Without
        this, a period rollover that missed its webhook would leave the
        workspace permanently locked out.
        """
        row = self.get_workspace_subscription(workspace_id)
        if not row:
            # No sub record — default allow (free tier, pre-provisioning).
            return True
        sub = WorkspaceSubscription.from_row(row)
        if sub.status not in ("active", "trialing"):
            return True  # enforcement left to the app layer for inactive subs

        # Self-heal stale period: if current_period_end is in the past, the
        # billing cycle has rolled over without a webhook arriving. Reset the
        # counter optimistically so the user isn't locked out. Stripe will
        # eventually fire customer.subscription.updated with the new period_end
        # and we'll pick that up in _on_subscription_updated.
        now = datetime.now(timezone.utc)
        if sub.current_period_end is not None and sub.current_period_end < now:
            logger.info(
                f"Self-healing stale billing period for {workspace_id!r}: "
                f"current_period_end={sub.current_period_end.isoformat()} is "
                f"in the past, resetting query_count_current_period to 0."
            )
            try:
                if self._supabase is not None:
                    self._supabase.rpc(
                        "reset_workspace_usage",
                        {"p_workspace_id": workspace_id},
                    ).execute()
            except Exception as e:
                logger.warning(f"reset_workspace_usage rpc failed: {e}")
            return True

        limit = (PLANS.get(sub.tier) or {}).get("query_limit")
        if limit is None:
            return True  # unlimited
        return sub.query_count_current_period < limit

    def increment_usage(self, workspace_id: str) -> int | None:
        """
        Atomically bumps query_count_current_period via a Postgres RPC.
        Returns the new count, or None if Supabase isn't configured.
        """
        if self._supabase is None:
            return None
        try:
            res = self._supabase.rpc(
                "increment_workspace_usage",
                {"p_workspace_id": workspace_id},
            ).execute()
            if isinstance(res.data, list) and res.data:
                return int(res.data[0].get("query_count_current_period", 0))
            if isinstance(res.data, dict):
                return int(res.data.get("query_count_current_period", 0))
            return None
        except Exception as e:
            logger.error(f"increment_usage failed: {e}")
            return None

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _load_or_new(self, workspace_id: str) -> WorkspaceSubscription:
        row = self.get_workspace_subscription(workspace_id)
        if row:
            return WorkspaceSubscription.from_row(row)
        return WorkspaceSubscription(workspace_id=workspace_id)

    def _upsert(self, sub: WorkspaceSubscription) -> None:
        if self._supabase is None:
            logger.info(f"[dev] would upsert subscription: {sub.to_row()}")
            return
        try:
            self._supabase.table(self.TABLE_NAME).upsert(
                sub.to_row(), on_conflict="workspace_id"
            ).execute()
        except Exception as e:
            logger.error(f"Subscription upsert failed: {e}")
