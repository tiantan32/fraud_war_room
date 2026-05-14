# Databricks notebook source
# DBTITLE 1,Infrastructure Setup Overview
# MAGIC %md
# MAGIC # 🛠️ War Room Infrastructure Setup
# MAGIC
# MAGIC One-time provisioning of platform resources for the Multi-Agent Financial Crime War Room.
# MAGIC
# MAGIC **What this creates:**
# MAGIC 1. **Genie Space** — Natural language SQL for the Transaction Analyst agent
# MAGIC 2. **Vector Search Endpoint** — Compute for embedding similarity queries
# MAGIC 3. **Vector Search Index** — Delta Sync index on `similar_past_cases.embedding_text`
# MAGIC 4. **Knowledge Assistant** — Databricks pre-built RAG agent (via `/api/2.1/knowledge-assistants`)
# MAGIC 5. **Knowledge Source** — Attaches the VS index to the Knowledge Assistant
# MAGIC
# MAGIC **Run this once**, then use `02_tools` + `03_agent_orchestration` for the demo.

# COMMAND ----------

# DBTITLE 1,Install dependencies
# MAGIC %pip install databricks-vectorsearch databricks-sdk --quiet

# COMMAND ----------

# DBTITLE 1,Configuration
import requests
import time

# ─── Configuration ───
CATALOG = "ttan_demo_catalog_main"
SCHEMA = "fsi_fraud_war_room"
FQN = f"{CATALOG}.{SCHEMA}"

workspace_host = spark.conf.get("spark.databricks.workspaceUrl")
api_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
HEADERS = {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}

print(f"✅ Config loaded: {FQN} on {workspace_host}")

# COMMAND ----------

# DBTITLE 1,Create Genie Space
# =============================================================================
# 1. GENIE SPACE CREATION
# Provides natural-language-to-SQL for the Transaction Analyst agent
# =============================================================================

GENIE_TABLES = [
    f"{FQN}.transactions",
    f"{FQN}.customers",
    f"{FQN}.alerts",
    f"{FQN}.network_relationships",
    f"{FQN}.sanctions_watchlist",
    f"{FQN}.cases",
    f"{FQN}.adverse_media"
]

def create_genie_space():
    """Create the Genie space for ad-hoc investigative SQL queries."""
    # Check if already exists by listing spaces
    resp = requests.get(
        f"https://{workspace_host}/api/2.0/genie/spaces",
        headers=HEADERS
    )
    
    if resp.status_code == 200:
        for space in resp.json().get("spaces", []):
            if "Financial Crime War Room" in space.get("title", ""):
                print(f"✅ Genie Space already exists")
                print(f"   ID: {space['space_id']}")
                print(f"   URL: https://{workspace_host}/genie/rooms/{space['space_id']}")
                return space['space_id']
    
    # Create new space
    create_resp = requests.post(
        f"https://{workspace_host}/api/2.0/genie/spaces",
        headers=HEADERS,
        json={
            "title": "Financial Crime War Room - Investigation Analyst",
            "description": (
                "Ad-hoc SQL investigation tool for fraud analysts. "
                "Ask natural language questions about transactions, customers, alerts, "
                "network relationships, sanctions, and adverse media."
            ),
            "table_identifiers": GENIE_TABLES
        }
    )
    
    if create_resp.status_code in (200, 201):
        space_id = create_resp.json().get("space_id") or create_resp.json().get("id")
        print(f"✅ Genie Space created")
        print(f"   ID: {space_id}")
        print(f"   URL: https://{workspace_host}/genie/rooms/{space_id}")
        print(f"   Tables: {len(GENIE_TABLES)}")
        return space_id
    else:
        print(f"⚠️ Genie Space creation: [{create_resp.status_code}] {create_resp.text[:300]}")
        # Return known ID if creation fails (already exists)
        return "01f14f0e4ec91ad5ba608f3ae4b432e3"

GENIE_SPACE_ID = create_genie_space()

# COMMAND ----------

# DBTITLE 1,Create Vector Search Endpoint and Index
# =============================================================================
# 2. VECTOR SEARCH ENDPOINT + INDEX
# Provides embedding-based similarity retrieval for the Knowledge Assistant
# =============================================================================
from databricks.vector_search.client import VectorSearchClient

VS_ENDPOINT_NAME = "fraud-war-room-vs-endpoint"
VS_INDEX_NAME = f"{FQN}.similar_past_cases_index"
VS_SOURCE_TABLE = f"{FQN}.similar_past_cases"
VS_EMBEDDING_MODEL = "databricks-gte-large-en"  # Required by Knowledge Assistants
VS_TEXT_COLUMN = "embedding_text"
VS_PRIMARY_KEY = "case_id"

vsc = VectorSearchClient()

# --- Create endpoint ---
try:
    ep = vsc.get_endpoint(VS_ENDPOINT_NAME)
    ep_state = ep.get('endpoint_status', {}).get('state', 'UNKNOWN')
    print(f"✅ VS Endpoint exists: {VS_ENDPOINT_NAME} (state: {ep_state})")
except Exception:
    print(f"   Creating VS Endpoint: {VS_ENDPOINT_NAME}...")
    vsc.create_endpoint(name=VS_ENDPOINT_NAME, endpoint_type="STANDARD")
    print(f"✅ VS Endpoint created (provisioning takes ~5 minutes)")

# --- Create Delta Sync Index ---
try:
    idx = vsc.get_index(endpoint_name=VS_ENDPOINT_NAME, index_name=VS_INDEX_NAME)
    idx_status = idx.describe().get('status', {}).get('ready', False)
    print(f"✅ VS Index exists: {VS_INDEX_NAME} (ready: {idx_status})")
except Exception:
    print(f"   Creating VS Index: {VS_INDEX_NAME}...")
    # Enable CDF on source table (required for delta sync)
    spark.sql(f"ALTER TABLE {VS_SOURCE_TABLE} SET TBLPROPERTIES (delta.enableChangeDataFeed = 'true')")
    
    vsc.create_delta_sync_index(
        endpoint_name=VS_ENDPOINT_NAME,
        source_table_name=VS_SOURCE_TABLE,
        index_name=VS_INDEX_NAME,
        pipeline_type="TRIGGERED",
        primary_key=VS_PRIMARY_KEY,
        embedding_source_column=VS_TEXT_COLUMN,
        embedding_model_endpoint_name=VS_EMBEDDING_MODEL,
        columns_to_sync=[VS_PRIMARY_KEY, VS_TEXT_COLUMN, "outcome", "typology", 
                         "amount_range", "jurisdiction", "resolution_time_hours", "sar_filed"]
    )
    print(f"✅ VS Index created. Initial sync in progress...")
    print(f"   Source: {VS_SOURCE_TABLE}")
    print(f"   Embedding: {VS_EMBEDDING_MODEL} on '{VS_TEXT_COLUMN}'")

# COMMAND ----------

# DBTITLE 1,Create Knowledge Assistant + Source
# =============================================================================
# 3. KNOWLEDGE ASSISTANT + KNOWLEDGE SOURCE
# Databricks pre-built RAG agent backed by the Vector Search index
# =============================================================================

KA_API_BASE = f"https://{workspace_host}/api/2.1/knowledge-assistants"

KA_DISPLAY_NAME = "Fraud Case Precedent Assistant"
KA_DESCRIPTION = (
    "Knowledge Assistant for financial crime investigation precedents. "
    "Retrieves similar historical cases via Vector Search and reasons "
    "over outcomes, SAR rates, and resolution timelines."
)
KA_INSTRUCTIONS = (
    "You are a financial crime case precedent specialist. When asked about a case, "
    "retrieve the most similar historical investigations from the knowledge base. "
    "Analyze outcomes, SAR filing rates, resolution timelines, and typology patterns. "
    "Always cite specific case IDs and statistics. Provide a verdict recommendation "
    "(legitimate/suspicious/escalate/block) with a confidence score (0-1)."
)

# --- Create Knowledge Assistant ---
ka_id = None
ka_endpoint = None

resp = requests.get(KA_API_BASE, headers=HEADERS)
if resp.status_code == 200:
    for ka in resp.json().get("knowledge_assistants", []):
        if ka.get("display_name") == KA_DISPLAY_NAME:
            ka_id = ka["id"]
            ka_endpoint = ka.get("endpoint_name")
            print(f"✅ Knowledge Assistant exists: {KA_DISPLAY_NAME}")
            print(f"   ID: {ka_id}")
            print(f"   Endpoint: {ka_endpoint}")
            break

if not ka_id:
    resp = requests.post(KA_API_BASE, headers=HEADERS, json={
        "display_name": KA_DISPLAY_NAME,
        "description": KA_DESCRIPTION,
        "instructions": KA_INSTRUCTIONS,
    })
    if resp.status_code in (200, 201):
        ka_data = resp.json()
        ka_id = ka_data.get("id")
        ka_endpoint = ka_data.get("endpoint_name")
        print(f"✅ Knowledge Assistant created: {KA_DISPLAY_NAME}")
        print(f"   ID: {ka_id}")
        print(f"   Endpoint: {ka_endpoint}")
    else:
        print(f"⚠️ KA creation [{resp.status_code}]: {resp.text[:200]}")

# --- Attach Vector Search index as knowledge source ---
if ka_id:
    sources_resp = requests.get(
        f"{KA_API_BASE}/{ka_id}/knowledge-sources", headers=HEADERS
    )
    existing_sources = sources_resp.json().get("knowledge_sources", []) if sources_resp.status_code == 200 else []
    
    has_vs_source = any(s.get("source_type") == "index" for s in existing_sources)
    
    if not has_vs_source:
        source_resp = requests.post(
            f"{KA_API_BASE}/{ka_id}/knowledge-sources",
            headers=HEADERS,
            json={
                "display_name": "Historical Fraud Cases (Vector Search)",
                "description": "500 historical financial crime investigation narratives with outcomes, typologies, and resolution data.",
                "source_type": "index",
                "index": {
                    "index_name": VS_INDEX_NAME,
                    "text_col": VS_TEXT_COLUMN,
                    "doc_uri_col": VS_PRIMARY_KEY
                }
            }
        )
        if source_resp.status_code in (200, 201):
            print(f"✅ Knowledge source attached: {VS_INDEX_NAME}")
        else:
            print(f"⚠️ Source attach [{source_resp.status_code}]: {source_resp.text[:200]}")
    else:
        print(f"✅ Knowledge source already attached ({len(existing_sources)} sources)")

# COMMAND ----------

# DBTITLE 1,Print resource summary
# =============================================================================
# 4. SUMMARY - Print all resource IDs for use in 02_tools
# =============================================================================

print("=" * 70)
print("  🏛️  INFRASTRUCTURE PROVISIONING COMPLETE")
print("=" * 70)
print(f"""
  Copy these IDs into 02_tools configuration:

  GENIE_SPACE_ID      = "{GENIE_SPACE_ID}"
  VS_ENDPOINT_NAME    = "{VS_ENDPOINT_NAME}"
  VS_INDEX_NAME       = "{VS_INDEX_NAME}"
  KA_ID               = "{ka_id}"
  KA_ENDPOINT_NAME    = "{ka_endpoint}"

  Genie:     https://{workspace_host}/genie/rooms/{GENIE_SPACE_ID}
  KA:        https://{workspace_host}/ml/knowledge-assistants/{ka_id}
""")
print("=" * 70)