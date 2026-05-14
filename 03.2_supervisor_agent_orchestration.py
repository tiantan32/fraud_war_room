# Databricks notebook source
# DBTITLE 1,Supervisor Agent Architecture
# MAGIC %md
# MAGIC # 🚨 Financial Crime War Room — Supervisor Agent API
# MAGIC
# MAGIC Same multi-agent fraud investigation, powered by the **Databricks Supervisor Agent API**.
# MAGIC
# MAGIC Instead of managing the agent loop (ThreadPoolExecutor + custom orchestration in `03.1`),
# MAGIC we define **hosted tools** (UC Functions, Genie Space, Knowledge Assistant) and let
# MAGIC Databricks manage tool selection, execution, and response synthesis.
# MAGIC
# MAGIC ## Key Differences vs. `03.1_custom_agent_orchestration`
# MAGIC
# MAGIC | Aspect | Custom (03.1) | Supervisor Agent (03.2) |
# MAGIC |--------|---------------|------------------------|
# MAGIC | Agent loop | Manual (ThreadPoolExecutor) | Databricks-managed |
# MAGIC | Tools | UC functions via `call_tool()` | UC functions as hosted `uc_function` |
# MAGIC | Genie | REST API calls | Hosted `genie_space` tool |
# MAGIC | KA | REST API calls | Hosted `knowledge_assistant` tool |
# MAGIC | Model | `databricks-claude-sonnet-4-5` | Same (per-request) |
# MAGIC | Tracing | MLflow `@mlflow.trace` | Built-in OpenTelemetry → UC |
# MAGIC | Code | ~150 lines orchestration | ~30 lines (single API call) |
# MAGIC
# MAGIC ## Architecture
# MAGIC
# MAGIC ```
# MAGIC ┌───────────────────────────────────────────────────────────┐
# MAGIC │  DatabricksOpenAI.responses.create(                       │
# MAGIC │    model, instructions, tools                             │
# MAGIC │  )                                                        │
# MAGIC └──────┬───────┬───────┬───────┬───────┬───────┬───────┘
# MAGIC        │       │       │       │       │       │
# MAGIC     Genie    UC      UC      UC      UC     KA
# MAGIC     Space   Func    Func    Func    Func   Agent
# MAGIC    (NL→SQL) (alert) (txn)  (net) (screen) (RAG)
# MAGIC ```
# MAGIC
# MAGIC All UC functions are created in `02_tools` — run that first.
# MAGIC
# MAGIC **Prerequisites:** `01_setup_infrastructure` (Genie + KA) and `02_tools` (UC functions)

# COMMAND ----------

# DBTITLE 1,Install dependencies
# MAGIC %pip install databricks-openai databricks-sdk mlflow --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Configuration
import time
import json
from datetime import datetime
from databricks.sdk import WorkspaceClient

# ─── Configuration ───
CATALOG = "ttan_demo_catalog_main"
SCHEMA = "fsi_fraud_war_room"
FQN = f"{CATALOG}.{SCHEMA}"

# Foundation Model (same as 03 custom orchestration)
SUPERVISOR_MODEL = "databricks-claude-sonnet-4-5"

# Platform resource IDs (from 01_setup_infrastructure)
GENIE_SPACE_ID = "01f14f0e4ec91ad5ba608f3ae4b432e3"
KA_ID = "8c765842-f021-4946-85b4-df326cc01b03"

# Supervisor Agent naming
SUPERVISOR_NAME = "Financial Crime War Room Supervisor"
SUPERVISOR_DESCRIPTION = (
    "Multi-agent supervisor for BSA/AML fraud alert investigations. "
    "Coordinates transaction analysis, customer profiling, network graph analysis, "
    "sanctions screening, and case precedent retrieval to render a verdict."
)

w = WorkspaceClient()
workspace_host = spark.conf.get("spark.databricks.workspaceUrl")

print(f"✅ Config: {FQN}")
print(f"   Model: {SUPERVISOR_MODEL}")
print(f"   Genie Space: {GENIE_SPACE_ID}")
print(f"   Knowledge Assistant: {KA_ID}")
print(f"   SDK: WorkspaceClient ready")
print(f"\n   ℹ️  UC Functions created in 02_tools (run that notebook first)")

# COMMAND ----------

# DBTITLE 1,Define Supervisor Tools
# =============================================================================
# STEP 2: DEFINE THE SUPERVISOR AGENT INVESTIGATION
# Single API call → Databricks manages the full agent loop
# =============================================================================
from databricks_openai import DatabricksOpenAI

client = DatabricksOpenAI(use_ai_gateway=True)

# ─── Tool definitions (hosted, server-side execution) ───
WAR_ROOM_TOOLS = [
    {
        "type": "genie_space",
        "genie_space": {
            "id": GENIE_SPACE_ID,
            "description": (
                "Ad-hoc SQL investigation tool for fraud analysis. Use this for dynamic, "
                "exploratory questions about transaction patterns, customer behavior, "
                "and relationships that the pre-built UC functions don't cover. "
                "Can answer natural language questions like 'Show me the distribution "
                "of transaction amounts between $8K-$10K by day of week for customer X'."
            )
        }
    },
    {
        "type": "knowledge_assistant",
        "knowledge_assistant": {
            "knowledge_assistant_id": KA_ID,
            "description": (
                "Historical fraud case precedent retrieval. Contains 500 past investigation "
                "narratives with outcomes, SAR filing rates, and resolution timelines. "
                "Use this to find similar historical cases by typology (structuring, layering, "
                "mule_network, sanctions_evasion) and reason over precedent to inform verdict."
            )
        }
    },
    {
        "type": "uc_function",
        "uc_function": {
            "name": f"{FQN}.get_alert_details",
            "description": "Look up alert details: customer, type, severity, triggering transaction. Always call this first."
        }
    },
    {
        "type": "uc_function",
        "uc_function": {
            "name": f"{FQN}.get_transaction_analysis",
            "description": (
                "Aggregate transaction statistics: totals, near-$10K structuring count, "
                "velocity, wire %, international %, unique countries/devices. "
                "Key for detecting structuring and velocity anomalies."
            )
        }
    },
    {
        "type": "uc_function",
        "uc_function": {
            "name": f"{FQN}.get_customer_risk_profile",
            "description": (
                "Full customer KYC profile: PEP status, risk score, occupation, income, "
                "account age, alert count, open cases. Essential for due diligence assessment."
            )
        }
    },
    {
        "type": "uc_function",
        "uc_function": {
            "name": f"{FQN}.get_network_analysis",
            "description": (
                "Network graph: 1st-hop connections, relationship types, cluster IDs, "
                "shared attributes. Dense clusters (>8 nodes, high risk scores) suggest "
                "mule networks. Normal customers have 3-5 connections."
            )
        }
    },
    {
        "type": "uc_function",
        "uc_function": {
            "name": f"{FQN}.get_high_risk_transactions",
            "description": "Individual high-risk transactions (risk_score >= 60). Use after statistics reveal suspicious patterns."
        }
    },
    {
        "type": "uc_function",
        "uc_function": {
            "name": f"{FQN}.screen_sanctions_and_media",
            "description": (
                "Sanctions watchlist (OFAC/EU/UN/UK) + adverse media screening. "
                "A direct sanctions match requires immediate escalation or block. "
                "Media hits alone are suspicious but not conclusive."
            )
        }
    },
]

print(f"✅ Supervisor tools defined: {len(WAR_ROOM_TOOLS)} hosted tools")
for t in WAR_ROOM_TOOLS:
    print(f"   • {t['type']}: {list(t[t['type']].values())[0] if t['type'] != 'uc_function' else t['uc_function']['name'].split('.')[-1]}")

# COMMAND ----------

# DBTITLE 1,Supervisor Instructions
# =============================================================================
# STEP 3: SUPERVISOR INSTRUCTIONS (replaces 7 individual system prompts)
# =============================================================================

SUPERVISOR_INSTRUCTIONS = """
You are the Financial Crime War Room Supervisor — a senior BSA/AML compliance AI 
that coordinates a multi-tool investigation of fraud alerts.

When given an alert ID, you MUST follow this investigation protocol:

## Investigation Protocol

1. **Alert Lookup** — Call `get_alert_details` to identify the customer, alert type, and severity.

2. **Parallel Investigation** (gather ALL of these):
   a. **Transaction Analysis** — Call `get_transaction_analysis` for aggregate stats.
      - If near_10k_count > 3 → STRUCTURING indicator. Use Genie Space for deeper pattern analysis.
      - If daily_velocity > 5 → VELOCITY anomaly. Use Genie Space for spike ratio.
   b. **Customer Profile** — Call `get_customer_risk_profile` for KYC/PEP/risk data.
   c. **Network Graph** — Call `get_network_analysis` for mule network indicators.
   d. **Sanctions Screening** — Call `screen_sanctions_and_media` for watchlist hits.
   e. **High-Risk Transactions** — Call `get_high_risk_transactions` for specific flagged txns.
   f. **Precedent Retrieval** — Query Knowledge Assistant for similar historical cases.
   g. **Ad-Hoc Investigation** — If patterns warrant deeper analysis, use the Genie Space.

3. **Synthesis & Verdict** — After gathering evidence from all tools:
   - Synthesize findings into a SAR-style narrative
   - Provide your verdict: `LEGITIMATE`, `SUSPICIOUS`, `ESCALATE`, or `BLOCK`
   - State confidence (0-100%)
   - Cite specific evidence (transaction IDs, amounts, case precedents)

## Decision Rules
- Sanctions match (high confidence) → BLOCK
- 4+ risk indicators converge → SUSPICIOUS or ESCALATE
- PEP + elevated activity → ESCALATE (always)
- High-risk jurisdiction + structuring → SUSPICIOUS minimum
- No material findings across all tools → LEGITIMATE

## Output Format
Always end your response with a structured verdict block:
```
═══ VERDICT ═══
Decision: [LEGITIMATE/SUSPICIOUS/ESCALATE/BLOCK]
Confidence: [0-100]%
Key Evidence: [bullet list of top 3-5 findings]
Recommended Action: [specific next step]
```
"""

print("✅ Supervisor instructions defined (investigation protocol + decision rules)")

# COMMAND ----------

# DBTITLE 1,Supervisor Investigation Function
# =============================================================================
# STEP 4: INVESTIGATE FUNCTION (single Supervisor API call)
# =============================================================================

def investigate_alert_supervisor(alert_id: str, stream: bool = True, 
                                  background: bool = False) -> dict:
    """
    🚨 Investigate a fraud alert using the Databricks Supervisor API.
    
    One API call → Databricks manages the full multi-tool agent loop:
    - Selects which tools to call based on investigation protocol
    - Executes tools server-side (no local compute needed)
    - Feeds results back to the model
    - Repeats until final verdict is synthesized
    
    Args:
        alert_id: Alert to investigate (e.g., "ALT-000001")
        stream: Whether to stream the response (shows progress)
        background: Use background mode for long-running investigations
    """
    start_time = time.time()
    
    investigation_prompt = (
        f"Investigate alert {alert_id}. Follow the full investigation protocol: "
        f"lookup the alert, analyze transactions, review customer profile, "
        f"check network connections, screen sanctions/media, retrieve similar cases, "
        f"and provide your final verdict with confidence and evidence."
    )
    
    print()
    print("═" * 70)
    print(f"  🚨 SUPERVISOR AGENT — INVESTIGATING: {alert_id}")
    print(f"  🤖 Model: {SUPERVISOR_MODEL}")
    print(f"  🧰 Tools: {len(WAR_ROOM_TOOLS)} hosted (Genie + KA + {len(WAR_ROOM_TOOLS)-2} UC Functions)")
    print(f"  🔄 Mode: {'Background' if background else 'Streaming' if stream else 'Synchronous'}")
    print("═" * 70)
    print()
    
    if stream:
        # Streaming mode — shows tool calls as they happen
        print("  ⏱️  Agent loop running (Databricks-managed)...")
        print("  ─" * 35)
        print()
        
        response = client.responses.create(
            model=SUPERVISOR_MODEL,
            instructions=SUPERVISOR_INSTRUCTIONS,
            input=[{"type": "message", "role": "user", "content": investigation_prompt}],
            tools=WAR_ROOM_TOOLS,
            stream=True,
            background=background
        )
        
        full_text = ""
        tool_calls_seen = []
        
        for event in response:
            # Track tool usage for audit
            if hasattr(event, 'type'):
                if 'function_call' in str(event.type):
                    tool_name = getattr(event, 'name', '') or ''
                    if tool_name and tool_name not in tool_calls_seen:
                        tool_calls_seen.append(tool_name)
                        print(f"  🔧 Tool called: {tool_name}")
            
            # Accumulate output text
            if hasattr(event, 'output_text') and event.output_text:
                full_text = event.output_text
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        print()
        print("  ─" * 35)
        print(f"  ✅ Investigation complete: {elapsed_ms:,}ms | Tools used: {len(tool_calls_seen)}")
        print()
        print(full_text)
        
        return {
            "alert_id": alert_id,
            "model": SUPERVISOR_MODEL,
            "mode": "supervisor_api",
            "tools_used": tool_calls_seen,
            "total_time_ms": elapsed_ms,
            "response": full_text
        }
    
    else:
        # Synchronous mode
        response = client.responses.create(
            model=SUPERVISOR_MODEL,
            instructions=SUPERVISOR_INSTRUCTIONS,
            input=[{"type": "message", "role": "user", "content": investigation_prompt}],
            tools=WAR_ROOM_TOOLS,
            stream=False,
            background=background
        )
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        output_text = response.output_text if hasattr(response, 'output_text') else str(response)
        
        print(f"  ✅ Investigation complete: {elapsed_ms:,}ms")
        print()
        print(output_text)
        
        return {
            "alert_id": alert_id,
            "model": SUPERVISOR_MODEL,
            "mode": "supervisor_api",
            "total_time_ms": elapsed_ms,
            "response": output_text
        }


print("✅ investigate_alert_supervisor() defined")
print("   Single API call → Databricks manages the agent loop")

# COMMAND ----------

# DBTITLE 1,DEMO - Investigate structuring alert
# =============================================================================
# 🎬 DEMO - Investigate Structuring Ring Alert
# =============================================================================

alert_id = "ALT-000001"  # Structuring ring (hot demo scenario)

print("=" * 70)
print("  🏛️  FINANCIAL CRIME WAR ROOM — SUPERVISOR AGENT")
print("=" * 70)

results = investigate_alert_supervisor(alert_id, stream=True)

# COMMAND ----------

# DBTITLE 1,BATCH DEMO - 3 Hot Scenarios
# =============================================================================
# BATCH DEMO - All 3 Hot Scenarios
# =============================================================================

hot_alerts = [
    ("ALT-000001", "Structuring Ring"),
    ("ALT-000010", "Sanctions Proximity"),
    ("ALT-000025", "Velocity Anomaly"),
]

print("=" * 70)
print("  🎬 BATCH INVESTIGATION — 3 Hot Scenarios (Supervisor API)")
print("=" * 70)

batch_results = []
for alert_id, scenario in hot_alerts:
    print(f"\n{'='*70}")
    print(f"  SCENARIO: {scenario}")
    print(f"{'='*70}")
    result = investigate_alert_supervisor(alert_id, stream=False)
    batch_results.append(result)
    print()

# Summary
print("\n" + "=" * 70)
print("  📊 BATCH RESULTS SUMMARY")
print("=" * 70)
for (alert_id, scenario), result in zip(hot_alerts, batch_results):
    print(f"  {scenario:<25} | {result['total_time_ms']:>6,}ms | Tools: {len(result.get('tools_used', []))}")
print("=" * 70)

# COMMAND ----------

# DBTITLE 1,Comparison - Custom vs Supervisor API
# =============================================================================
# COMPARISON: Custom Orchestration vs Supervisor API
# =============================================================================

print("""
╔═════════════════════════════════════════════════════════════════╗
║  🤜 CUSTOM (03) vs SUPERVISOR API (04) 🤛              ║
╠═════════════════════════════════════════════════════════════════╣
║  Custom (03_agent_orchestration):                        ║
║    • Full control over agent prompts & parallelism        ║
║    • Custom consensus logic (4+ agree = auto-decide)      ║
║    • In-process execution (needs running cluster)          ║
║    • ~300 lines of orchestration code                      ║
║    • MLflow tracing via @mlflow.trace decorator            ║
║    • 7 agent definitions + ThreadPoolExecutor              ║
║                                                             ║
║  Supervisor API (04_supervisor_agent_orchestration):        ║
║    • One API call → Databricks runs the agent loop          ║
║    • Hosted tool execution (no cluster needed for tools)    ║
║    • Model-agnostic (swap models per request)               ║
║    • Built-in tracing (OpenTelemetry → Unity Catalog)       ║
║    • ~30 lines of investigation code                        ║
║    • AI Gateway rate limits, fallbacks, governance          ║
║    • Background mode for complex investigations             ║
╚═════════════════════════════════════════════════════════════════╝

When to use which:
• Demo wow-factor / full control → 03_agent_orchestration (custom)
• Production deployment / simplicity → 04_supervisor_agent (Supervisor API)
• Both produce the same investigation with the same underlying data.
""")