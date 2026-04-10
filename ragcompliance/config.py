import os
from dataclasses import dataclass


@dataclass
class RAGComplianceConfig:
    supabase_url: str = ""
    supabase_key: str = ""
    workspace_id: str = "default"
    table_name: str = "rag_audit_logs"
    enabled: bool = True
    # If True, logs to stdout when Supabase is not configured (dev mode)
    dev_mode: bool = False

    @classmethod
    def from_env(cls) -> "RAGComplianceConfig":
        return cls(
            supabase_url=os.getenv("RAGCOMPLIANCE_SUPABASE_URL", ""),
            supabase_key=os.getenv("RAGCOMPLIANCE_SUPABASE_KEY", ""),
            workspace_id=os.getenv("RAGCOMPLIANCE_WORKSPACE_ID", "default"),
            table_name=os.getenv("RAGCOMPLIANCE_TABLE_NAME", "rag_audit_logs"),
            enabled=os.getenv("RAGCOMPLIANCE_ENABLED", "true").lower() == "true",
            dev_mode=os.getenv("RAGCOMPLIANCE_DEV_MODE", "false").lower() == "true",
        )
