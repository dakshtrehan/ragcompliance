import json
import logging
from typing import Any

from .config import RAGComplianceConfig
from .models import AuditRecord

logger = logging.getLogger(__name__)


class AuditStorage:
    def __init__(self, config: RAGComplianceConfig):
        self.config = config
        self._client = None

        if config.supabase_url and config.supabase_key:
            try:
                from supabase import create_client
                self._client = create_client(config.supabase_url, config.supabase_key)
                logger.info("RAGCompliance: Supabase client initialized.")
            except ImportError:
                logger.warning(
                    "RAGCompliance: supabase package not installed. "
                    "Run: pip install supabase. Falling back to dev mode."
                )
        else:
            if not config.dev_mode:
                logger.warning(
                    "RAGCompliance: No Supabase credentials found. "
                    "Set RAGCOMPLIANCE_DEV_MODE=true to suppress this warning."
                )

    def save(self, record: AuditRecord) -> bool:
        if not self.config.enabled:
            return True

        if self._client is None:
            if self.config.dev_mode:
                print(f"[RAGCompliance DEV] Audit record:\n{json.dumps(record.to_dict(), indent=2, default=str)}")
            return True

        try:
            self._client.table(self.config.table_name).insert(record.to_dict()).execute()
            return True
        except Exception as e:
            logger.error(f"RAGCompliance: Failed to save audit record: {e}")
            return False

    def query(
        self,
        workspace_id: str | None = None,
        session_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        if self._client is None:
            logger.warning("RAGCompliance: No Supabase client — cannot query.")
            return []

        try:
            q = self._client.table(self.config.table_name).select("*")
            if workspace_id:
                q = q.eq("workspace_id", workspace_id)
            if session_id:
                q = q.eq("session_id", session_id)
            result = q.order("timestamp", desc=True).limit(limit).execute()
            return result.data or []
        except Exception as e:
            logger.error(f"RAGCompliance: Query failed: {e}")
            return []
