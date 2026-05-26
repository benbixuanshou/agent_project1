from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    # Server
    app_env: str = "dev"  # dev / staging / prod
    host: str = "0.0.0.0"
    port: int = 9900

    # Security
    api_keys: str = ""  # comma-separated, empty = no auth (dev)
    webhook_secret: str = ""  # HMAC secret for Alertmanager webhook
    rate_limit_default_per_minute: int = 120
    rate_limit_chat_per_minute: int = 30

    # IM notification
    dingtalk_webhook_url: str = ""
    dingtalk_secret: str = ""
    feishu_webhook_url: str = ""
    feishu_secret: str = ""
    wecom_webhook_url: str = ""
    notify_enabled: bool = True

    # DashScope API (embedding only, LLM uses DeepSeek)
    dashscope_api_key: str = Field(alias="DASHSCOPE_API_KEY")

    # DashScope Embedding
    dashscope_embedding_model: str = "text-embedding-v4"

    # DeepSeek LLM (OpenAI-compatible API)
    deepseek_api_key: str = Field(alias="DEEPSEEK_API_KEY")
    deepseek_model: str = "deepseek-chat"
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_temperature: float = 0.7
    deepseek_max_tokens: int = 2000
    deepseek_timeout: int = 30          # seconds, per-request timeout
    deepseek_max_retries: int = 2       # retry on transient failures (429, 5xx)
    deepseek_retry_backoff: float = 1.5  # exponential backoff multiplier

    # Milvus
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_username: str = ""
    milvus_password: str = ""
    milvus_database: str = "default"
    milvus_collection: str = "biz"
    milvus_vector_dim: int = 1024
    milvus_timeout: int = 10_000

    # RAG
    rag_top_k: int = 3

    # Document chunking
    chunk_max_size: int = 800
    chunk_overlap: int = 100

    # Intent recognition
    intent_enabled: bool = True
    intent_confidence_threshold: float = 0.05

    # Prometheus
    prometheus_base_url: str = "http://localhost:9090"
    prometheus_timeout: int = 10
    prometheus_mock_enabled: bool = False

    # CLS (Cloud Log Service)
    cls_mock_enabled: bool = False

    # K8s Events
    k8s_mock_enabled: bool = True
    k8s_api_url: str = ""          # e.g. https://k8s-api.example.com
    k8s_api_token: str = ""        # Bearer token (or use in-cluster SA)
    k8s_verify_ssl: bool = True

    # External log systems
    elasticsearch_url: str = ""
    loki_url: str = ""

    # GitLab CI
    gitlab_api_url: str = ""
    gitlab_api_token: str = ""

    # ITSM
    jira_url: str = ""
    jira_api_token: str = ""
    jira_project_key: str = ""

    # Multi-cluster
    cluster_id: str = "default"    # unique per deployment

    # Patrol agent
    patrol_interval_minutes: int = 15  # 0 = disabled

    # Knowledge sync
    knowledge_cluster_tag: str = "default"  # cluster label for cross-cluster knowledge discovery

    # Change tracking
    change_tracking_enabled: bool = True
    change_tracking_mock: bool = True

    # File upload
    upload_path: str = "./uploads"
    upload_allowed_extensions: str = "txt,md"

    # Session
    session_backend: str = "sqlite"  # "memory" or "redis" or "sqlite" or "mysql"
    sqlite_path: str = "data/sessions.db"
    redis_url: str = "redis://localhost:6379/0"
    session_max_pairs: int = 6
    session_ttl_seconds: int = 7 * 24 * 3600  # 7 days

    # MySQL (for session persistence)
    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_user: str = "superbiz"
    mysql_password: str = "superbiz123"
    mysql_database: str = "superbiz"

    # ── Harness Engineering ──────────────────────────────
    # Guardrail engine
    guardrail_rules_path: str = ""  # empty = use default (app/harness/guardrails/rules.yaml)
    guardrail_enabled: bool = True

    # Circuit breaker
    circuit_breaker_failure_threshold: int = 3
    circuit_breaker_cooldown_seconds: int = 30

    # ReAct execution constraints
    react_max_iterations: int = 15
    react_max_tool_retries: int = 3
    react_timeout_seconds: int = 120

    # Incident learner
    incident_learner_data_dir: str = "data"
    incident_cluster_threshold: int = 3  # incidents before auto-generating candidate rule

    # Session housekeeping
    session_cleanup_interval_seconds: int = 3600  # 1 hour

    # Context management
    context_budget_limit_tokens: int = 16000
    context_compression_trigger_util: float = 0.70


settings = Settings()
