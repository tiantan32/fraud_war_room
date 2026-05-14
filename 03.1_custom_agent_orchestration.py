# Databricks notebook source
# DBTITLE 1,War Room Architecture Overview
# MAGIC %md
# MAGIC # 🚨 Financial Crime War Room — Multi-Agent Investigation
# MAGIC
# MAGIC A fraud alert fires. A **swarm of 7 specialist AI agents** investigate in parallel, build a case file, and either auto-resolve or escalate — with full audit trail in Unity Catalog.
# MAGIC
# MAGIC ## Architecture
# MAGIC
# MAGIC ```
# MAGIC ┌─────────────┐     ┌─────────────────────────────────────────────────────────────┐
# MAGIC │  Alert Fire │────▶│  ThreadPoolExecutor (5 agents in parallel)                │
# MAGIC └─────────────┘     │                                                             │
# MAGIC                     │  ┌───────────────┐ ┌─────────────┐ ┌───────────────┐    │
# MAGIC                     │  │  Transaction  │ │  Customer   │ │   Network     │    │
# MAGIC                     │  │  Analyst      │ │  History    │ │   Graph       │    │
# MAGIC                     │  │  [Genie MCP]  │ │  [SQL]      │ │   [SQL]       │    │
# MAGIC                     │  └───────┬───────┘ └──────┬──────┘ └───────┬───────┘    │
# MAGIC                     │         │               │               │               │
# MAGIC                     │  ┌──────┴───────────────┴───────────────┴────────────┐ │
# MAGIC                     │  │  Sanctions/News Agent    │  Knowledge Assistant    │ │
# MAGIC                     │  │  [SQL + Screening]       │  [Vector Search/RAG]   │ │
# MAGIC                     │  └─────────────────────────┴────────────────────────┘ │
# MAGIC                     └────────────────────────────┬────────────────────────────────┘
# MAGIC                                                 ▼
# MAGIC                     ┌─────────────────────────────────────────────────────────────┐
# MAGIC                     │      Narrative Writer Agent (synthesizes all 5)             │
# MAGIC                     └────────────────────────────┬────────────────────────────────┘
# MAGIC                                                 ▼
# MAGIC                     ┌─────────────────────────────────────────────────────────────┐
# MAGIC                     │      Decision Agent (Consensus + Verdict)                   │
# MAGIC                     └────────────────────────────┬────────────────────────────────┘
# MAGIC                                                 ▼
# MAGIC                     ┌─────────────────────────────────────────────────────────────┐
# MAGIC                     │  MLflow Trace + Unity Catalog Audit Trail                   │
# MAGIC                     └─────────────────────────────────────────────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC ## Agent → Tool/Platform Mapping
# MAGIC
# MAGIC | Agent | Tool / MCP | Data Source |
# MAGIC |-------|-----------|-------------|
# MAGIC | Transaction Analyst | **Genie MCP** (natural language SQL) | `transactions` via Genie Space |
# MAGIC | Customer History | Direct SQL | `customers`, `alerts`, `cases` |
# MAGIC | Network Graph | Direct SQL | `network_relationships` |
# MAGIC | Sanctions/News | Direct SQL + Screening | `sanctions_watchlist`, `adverse_media` |
# MAGIC | Knowledge Assistant | **Vector Search** / RAG | `similar_past_cases` (embeddings) |
# MAGIC | Narrative Writer | LLM synthesis | All agent outputs |
# MAGIC | Decision Agent | LLM + rule engine | Consensus analysis |
# MAGIC
# MAGIC **Platform:** Genie (ad-hoc SQL) · Vector Search (similar cases) · Lakebase (case state) · AI Gateway (governance) · MLflow Tracing (audit)
# MAGIC
# MAGIC **Data:** `ttan_demo_catalog_main.fsi_fraud_war_room` — 130K rows across 10 tables

# COMMAND ----------

# DBTITLE 1,Install dependencies
# MAGIC %pip install mlflow databricks-sdk openai tiktoken databricks-vectorsearch --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Load Tools
import time, json, re, requests
from dataclasses import dataclass, field
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

import mlflow
from openai import OpenAI
from pyspark.sql import functions as F

# ─── Configuration ───
CATALOG = "ttan_demo_catalog_main"
SCHEMA = "fsi_fraud_war_room"
FQN = f"{CATALOG}.{SCHEMA}"

MODEL_ENDPOINT = "databricks-claude-sonnet-4-5"
GENIE_SPACE_ID = "01f14f0e4ec91ad5ba608f3ae4b432e3"
KA_ENDPOINT = "ka-8c765842-endpoint"

workspace_host = spark.conf.get("spark.databricks.workspaceUrl")
api_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# ─── LLM Client ───
llm_client = OpenAI(
    base_url=f"https://{workspace_host}/serving-endpoints",
    api_key=api_token
)

def call_llm(system_prompt: str, user_prompt: str) -> tuple:
    """Call Foundation Model API. Returns (text, tokens)."""
    try:
        r = llm_client.chat.completions.create(
            model=MODEL_ENDPOINT,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_prompt}],
            max_tokens=1024, temperature=0.1
        )
        return r.choices[0].message.content, r.usage.total_tokens if r.usage else 0
    except Exception as e:
        return f"[LLM ERROR: {e}]", 0

def parse_verdict(text: str) -> tuple:
    """Extract (verdict, confidence) from LLM response."""
    t = text.lower()
    if "block" in t and any(x in t for x in ["verdict: block", "verdict:**block"]):
        verdict = "block"
    elif "escalate" in t:
        verdict = "escalate"
    elif "legitimate" in t:
        verdict = "legitimate"
    else:
        verdict = "suspicious"
    conf_match = re.search(r'confidence[:\s]*([0-9]*\.?[0-9]+)', t)
    confidence = float(conf_match.group(1)) if conf_match else 0.75
    if confidence > 1: confidence /= 100.0
    return verdict, max(0.0, min(1.0, confidence))

# ─── UC Tool Caller ───
def call_tool(func_name: str, **params) -> List[dict]:
    """Call a UC function and return results as list of dicts."""
    args = ", ".join(
        f"'{v}'" if isinstance(v, str) else str(v) for v in params.values()
    )
    rows = spark.sql(f"SELECT * FROM {FQN}.{func_name}({args})").collect()
    return [row.asDict() for row in rows]

# ─── Genie Tool ───
def call_genie(question: str, timeout: int = 30) -> dict:
    """Query Genie Space for ad-hoc NL-to-SQL."""
    headers = {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}
    base = f"https://{workspace_host}/api/2.0/genie/spaces/{GENIE_SPACE_ID}"
    try:
        r = requests.post(f"{base}/start-conversation", headers=headers, json={"content": question})
        r.raise_for_status()
        data = r.json()
        conv_id = data.get("conversation_id") or data.get("conversation", {}).get("id")
        msg_id = data.get("message_id") or data.get("message", {}).get("id")
        if not conv_id:
            return {"status": "error", "sql": None, "results": []}
        start = time.time()
        while time.time() - start < timeout:
            poll = requests.get(f"{base}/conversations/{conv_id}/messages/{msg_id}", headers=headers)
            if poll.status_code == 200:
                msg = poll.json()
                if msg.get("status") in ("COMPLETED", "completed"):
                    sql_text, results = None, []
                    for att in msg.get("attachments", []):
                        if "query" in att.get("type", "").lower():
                            sql_text = att.get("query", {}).get("query", "")
                        elif "result" in att.get("type", "").lower():
                            results = att.get("data", {}).get("rows", [])
                    return {"status": "ok", "sql": sql_text, "results": results[:20]}
                elif msg.get("status") in ("FAILED", "CANCELLED"):
                    return {"status": "error", "sql": None, "results": []}
            time.sleep(1)
        return {"status": "timeout", "sql": None, "results": []}
    except Exception as e:
        return {"status": "error", "sql": None, "results": []}

# ─── AgentResult ───
@dataclass
class AgentResult:
    agent_name: str
    verdict: str
    confidence: float
    reasoning: str
    evidence_refs: List[str] = field(default_factory=list)
    duration_ms: int = 0
    tokens_used: int = 0
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)

# MLflow experiment
mlflow.set_experiment("/Users/tian.tan@databricks.com/fraud_war_room/war_room_traces")

print(f"✅ Config loaded: {FQN} | Model: {MODEL_ENDPOINT}")
print(f"   call_tool() → UC functions | call_genie() → Genie Space | call_llm() → LLM")

# COMMAND ----------

# DBTITLE 1,Agent 1 - Transaction Analyst
# =============================================================================
# AGENT 1: TRANSACTION ANALYST (Genie-enhanced)
# UC Tools: get_transaction_analysis, get_high_risk_transactions
# Platform: Genie Space (ad-hoc NL-to-SQL)
# =============================================================================

def run_transaction_analyst(customer_id: str, alert_id: str) -> AgentResult:
    start_time = time.time()
    tool_calls = []
    
    # UC Tool: get_transaction_analysis
    stats = call_tool("get_transaction_analysis", p_customer_id=customer_id)
    s = stats[0] if stats else {}
    tool_calls.append({"tool": "get_transaction_analysis", "customer_id": customer_id})
    
    # UC Tool: get_high_risk_transactions
    risky = call_tool("get_high_risk_transactions", p_customer_id=customer_id, p_min_risk_score=60)
    tool_calls.append({"tool": "get_high_risk_transactions", "count": len(risky)})
    
    velocity = s.get('daily_velocity', 0)
    
    # Genie: triggered conditionally
    genie_insights = "[Genie not triggered]"
    if s.get('near_10k_count', 0) > 3:
        genie_result = call_genie(f"For customer {customer_id}, show distribution of amounts between $8000-$10000 by day of week and channel")
        tool_calls.append({"tool": "genie_space", "status": genie_result["status"]})
        if genie_result["status"] == "ok":
            genie_insights = f"SQL: {genie_result['sql']}\nResults: {json.dumps(genie_result['results'][:5], default=str)}"
    elif velocity > 5:
        genie_result = call_genie(f"For customer {customer_id}, compare daily transaction count this week vs 90-day average")
        tool_calls.append({"tool": "genie_space", "status": genie_result["status"]})
        if genie_result["status"] == "ok":
            genie_insights = f"SQL: {genie_result['sql']}\nResults: {json.dumps(genie_result['results'][:5], default=str)}"
    
    # LLM reasoning
    system_prompt = "You are a transaction pattern analyst. Identify structuring (near-$10K), velocity anomalies, geographic risk. Provide: Verdict: [legitimate/suspicious/escalate/block] Confidence: [0-1] Reasoning: [brief]"
    user_prompt = f"""CUSTOMER: {customer_id} | ALERT: {alert_id}

=== UC Tool: get_transaction_analysis ===
{json.dumps(s, indent=2, default=str)}

=== UC Tool: get_high_risk_transactions ({len(risky)} hits) ===
{json.dumps(risky[:5], indent=2, default=str)}

=== Genie Space ===
{genie_insights}

Verdict/Confidence/Reasoning:"""
    
    response, tokens = call_llm(system_prompt, user_prompt)
    tool_calls.append({"tool": "llm", "tokens": tokens})
    verdict, confidence = parse_verdict(response)
    
    return AgentResult(
        agent_name="transaction_analyst", verdict=verdict, confidence=confidence,
        reasoning=response, evidence_refs=[f"near_10k:{s.get('near_10k_count',0)}", f"velocity:{velocity}"],
        duration_ms=int((time.time()-start_time)*1000), tokens_used=tokens, tool_calls=tool_calls
    )

print("✅ Agent 1: Transaction Analyst [UC: get_transaction_analysis, get_high_risk_transactions + Genie]")

# COMMAND ----------

# DBTITLE 1,Agent 2 - Customer History
# =============================================================================
# AGENT 2: CUSTOMER HISTORY AGENT
# UC Tools: get_customer_risk_profile
# =============================================================================

def run_customer_history_agent(customer_id: str) -> AgentResult:
    start_time = time.time()
    tool_calls = []
    
    # UC Tool: get_customer_risk_profile
    profile = call_tool("get_customer_risk_profile", p_customer_id=customer_id)
    p = profile[0] if profile else {}
    tool_calls.append({"tool": "get_customer_risk_profile", "customer_id": customer_id})
    
    system_prompt = "You are a customer due diligence specialist. Review KYC, PEP status, alert patterns. Provide: Verdict: [legitimate/suspicious/escalate/block] Confidence: [0-1] Reasoning: [brief]"
    user_prompt = f"""=== UC Tool: get_customer_risk_profile ===
{json.dumps(p, indent=2, default=str)}

Verdict/Confidence/Reasoning:"""
    
    response, tokens = call_llm(system_prompt, user_prompt)
    tool_calls.append({"tool": "llm", "tokens": tokens})
    verdict, confidence = parse_verdict(response)
    
    return AgentResult(
        agent_name="customer_history_agent", verdict=verdict, confidence=confidence,
        reasoning=response, evidence_refs=[customer_id],
        duration_ms=int((time.time()-start_time)*1000), tokens_used=tokens, tool_calls=tool_calls
    )

print("✅ Agent 2: Customer History [UC: get_customer_risk_profile]")

# COMMAND ----------

# DBTITLE 1,Agent 3 - Network Graph
# =============================================================================
# AGENT 3: NETWORK GRAPH AGENT
# UC Tools: get_network_analysis
# =============================================================================

def run_network_graph_agent(customer_id: str) -> AgentResult:
    start_time = time.time()
    tool_calls = []
    
    # UC Tool: get_network_analysis
    connections = call_tool("get_network_analysis", p_customer_id=customer_id)
    tool_calls.append({"tool": "get_network_analysis", "count": len(connections)})
    
    # Analyze topology
    degree = len(connections)
    rel_types = {}
    high_risk = 0
    for c in connections:
        rt = c.get('relationship_type', 'unknown')
        rel_types[rt] = rel_types.get(rt, 0) + 1
        if c.get('risk_score', 0) > 70:
            high_risk += 1
    
    system_prompt = "You are a network intelligence analyst. Dense clusters (>8 connections, high risk) suggest mule networks. Normal = 3-5 connections. Provide: Verdict: [legitimate/suspicious/escalate/block] Confidence: [0-1] Reasoning: [brief]"
    user_prompt = f"""=== UC Tool: get_network_analysis ===
Connections: {degree} | High-risk (>70): {high_risk}
Relationship types: {json.dumps(rel_types)}
Top connections: {json.dumps(connections[:8], indent=2, default=str)}

Verdict/Confidence/Reasoning:"""
    
    response, tokens = call_llm(system_prompt, user_prompt)
    tool_calls.append({"tool": "llm", "tokens": tokens})
    verdict, confidence = parse_verdict(response)
    
    return AgentResult(
        agent_name="network_graph_agent", verdict=verdict, confidence=confidence,
        reasoning=response, evidence_refs=[c.get('connected_to','') for c in connections[:5]],
        duration_ms=int((time.time()-start_time)*1000), tokens_used=tokens, tool_calls=tool_calls
    )

print("✅ Agent 3: Network Graph [UC: get_network_analysis]")

# COMMAND ----------

# DBTITLE 1,Agent 4 - Sanctions and News
# =============================================================================
# AGENT 4: SANCTIONS / NEWS AGENT
# UC Tools: screen_sanctions_and_media
# =============================================================================

def run_sanctions_agent(customer_id: str) -> AgentResult:
    start_time = time.time()
    tool_calls = []
    
    # UC Tool: screen_sanctions_and_media
    hits = call_tool("screen_sanctions_and_media", p_customer_id=customer_id)
    tool_calls.append({"tool": "screen_sanctions_and_media", "hits": len(hits)})
    
    sanctions_hits = [h for h in hits if h.get('source_type') == 'sanctions']
    media_hits = [h for h in hits if h.get('source_type') == 'adverse_media']
    
    system_prompt = "You are a sanctions screening specialist. Direct sanctions match = escalate/block. Adverse media alone = suspicious. Provide: Verdict: [legitimate/suspicious/escalate/block] Confidence: [0-1] Reasoning: [brief]"
    user_prompt = f"""=== UC Tool: screen_sanctions_and_media ===
Sanctions hits: {len(sanctions_hits)}
{json.dumps(sanctions_hits[:5], indent=2, default=str)}

Adverse media hits: {len(media_hits)}
{json.dumps(media_hits[:5], indent=2, default=str)}

Verdict/Confidence/Reasoning:"""
    
    response, tokens = call_llm(system_prompt, user_prompt)
    tool_calls.append({"tool": "llm", "tokens": tokens})
    verdict, confidence = parse_verdict(response)
    
    evidence = [f"sanctions:{h.get('entity_name','')}" for h in sanctions_hits[:3]]
    evidence += [f"media:{h.get('risk_category','')}" for h in media_hits[:2]]
    
    return AgentResult(
        agent_name="sanctions_agent", verdict=verdict, confidence=confidence,
        reasoning=response, evidence_refs=evidence,
        duration_ms=int((time.time()-start_time)*1000), tokens_used=tokens, tool_calls=tool_calls
    )

print("✅ Agent 4: Sanctions/News [UC: screen_sanctions_and_media]")

# COMMAND ----------

# DBTITLE 1,Agent 5 - Knowledge Assistant (Precedent Retrieval)
# =============================================================================
# AGENT 5: KNOWLEDGE ASSISTANT (Precedent Retrieval)
# UC Tools: get_similar_cases
# =============================================================================

def run_knowledge_assistant_agent(customer_id: str, alert_type: str) -> AgentResult:
    start_time = time.time()
    tool_calls = []
    
    typology_map = {
        "structuring": "structuring", "velocity": "layering",
        "sanctions_hit": "sanctions_evasion", "unusual_geography": "layering",
        "amount_anomaly": "structuring", "network_risk": "mule_network"
    }
    target_typology = typology_map.get(alert_type, "structuring")
    
    # UC Tool: get_similar_cases
    cases = call_tool("get_similar_cases", p_typology=target_typology, p_limit=5)
    tool_calls.append({"tool": "get_similar_cases", "typology": target_typology, "count": len(cases)})
    
    # Compute stats
    sar_count = sum(1 for c in cases if c.get('sar_filed'))
    outcomes = {}
    for c in cases:
        o = c.get('outcome', 'unknown')
        outcomes[o] = outcomes.get(o, 0) + 1
    
    system_prompt = "You are a case precedent specialist. Analyze historical cases and provide verdict based on what happened in similar past investigations. Cite case IDs. Provide: Verdict: [legitimate/suspicious/escalate/block] Confidence: [0-1] Reasoning: [brief]"
    user_prompt = f"""=== UC Tool: get_similar_cases(typology='{target_typology}') ===
Cases retrieved: {len(cases)}
SAR filed rate: {sar_count}/{len(cases)}
Outcome distribution: {json.dumps(outcomes)}

Case details:
{json.dumps([{k:v for k,v in c.items() if k != 'embedding_text'} for c in cases], indent=2, default=str)}

Based on precedent, verdict/confidence/reasoning:"""
    
    response, tokens = call_llm(system_prompt, user_prompt)
    tool_calls.append({"tool": "llm", "tokens": tokens})
    verdict, confidence = parse_verdict(response)
    
    return AgentResult(
        agent_name="knowledge_assistant", verdict=verdict, confidence=confidence,
        reasoning=response, evidence_refs=[c.get('case_id','') for c in cases[:3]],
        duration_ms=int((time.time()-start_time)*1000), tokens_used=tokens, tool_calls=tool_calls
    )

print("✅ Agent 5: Knowledge Assistant [UC: get_similar_cases]")

# COMMAND ----------

# DBTITLE 1,Agent 5 - Narrative Writer
# =============================================================================
# AGENT 6: NARRATIVE WRITER
# No UC tools — LLM synthesis only
# =============================================================================

def run_narrative_writer(customer_id: str, other_results: List[AgentResult]) -> AgentResult:
    start_time = time.time()
    
    findings = "\n\n".join([
        f"--- {r.agent_name.upper()} ---\nVerdict: {r.verdict} ({r.confidence:.0%})\n"
        f"Tools: {[t.get('tool','') for t in r.tool_calls]}\n"
        f"Analysis: {r.reasoning[:500]}"
        for r in other_results
    ])
    
    system_prompt = "You are a BSA/AML narrative writer. Synthesize findings into a SAR-style narrative. End with: Verdict: [legitimate/suspicious/escalate/block] Confidence: [0-1]"
    user_prompt = f"""CASE: {customer_id}\n\nFINDINGS:\n{findings}\n\nSynthesize into SAR narrative + verdict:"""
    
    response, tokens = call_llm(system_prompt, user_prompt)
    verdict, confidence = parse_verdict(response)
    
    return AgentResult(
        agent_name="narrative_writer", verdict=verdict, confidence=confidence,
        reasoning=response, evidence_refs=[r.agent_name for r in other_results],
        duration_ms=int((time.time()-start_time)*1000), tokens_used=tokens,
        tool_calls=[{"tool": "llm", "tokens": tokens}]
    )

print("✅ Agent 6: Narrative Writer [LLM only]")

# COMMAND ----------

# DBTITLE 1,Agent 6 - Decision Agent
# =============================================================================
# AGENT 7: DECISION AGENT
# No UC tools — LLM + hard rules engine
# =============================================================================

def run_decision_agent(all_results: List[AgentResult]) -> AgentResult:
    start_time = time.time()
    
    verdicts = [r.verdict for r in all_results]
    verdict_counts = Counter(verdicts)
    most_common_verdict, most_common_count = verdict_counts.most_common(1)[0]
    consensus_pct = most_common_count / len(verdicts)
    avg_confidence = sum(r.confidence for r in all_results) / len(all_results)
    
    agent_summary = "\n".join([f"  {r.agent_name:<25} → {r.verdict:<12} ({r.confidence:.0%})" for r in all_results])
    
    system_prompt = "You are the final decision authority. Apply rules: 4+ agree + >80% conf = auto-resolve, split = escalate, any block >90% = block. Verdict: [legitimate/suspicious/escalate/block] Confidence: [0-1] Reasoning:"
    user_prompt = f"""VERDICTS:\n{agent_summary}\n\nConsensus: {most_common_verdict} ({most_common_count}/{len(verdicts)}, {consensus_pct:.0%})\nAvg confidence: {avg_confidence:.0%}\n\nFinal decision:"""
    
    response, tokens = call_llm(system_prompt, user_prompt)
    verdict, confidence = parse_verdict(response)
    
    # Hard rules override
    block_agents = [r for r in all_results if r.verdict == "block" and r.confidence > 0.9]
    if block_agents:
        verdict, confidence = "block", max(confidence, 0.95)
    elif most_common_count >= 4 and avg_confidence > 0.8:
        verdict = most_common_verdict
    elif consensus_pct < 0.5:
        verdict = "escalate"
    
    return AgentResult(
        agent_name="decision_agent", verdict=verdict, confidence=confidence,
        reasoning=response, evidence_refs=[f"consensus:{most_common_count}/{len(verdicts)}"],
        duration_ms=int((time.time()-start_time)*1000), tokens_used=tokens,
        tool_calls=[{"tool": "llm", "tokens": tokens}]
    )

print("✅ Agent 7: Decision Agent [LLM + rules engine]")

# COMMAND ----------

# DBTITLE 1,War Room Orchestrator
# =============================================================================
# WAR ROOM ORCHESTRATOR
# 7 agents | 5 parallel | UC Tools + Genie + LLM
# =============================================================================
import uuid
from datetime import datetime

@mlflow.trace
def investigate_alert(alert_id: str) -> dict:
    """Investigate a fraud alert using 7 specialist agents calling UC tools."""
    total_start = time.time()
    
    # UC Tool: get_alert_details
    alert_info = call_tool("get_alert_details", p_alert_id=alert_id)
    if not alert_info:
        print(f"❌ Alert {alert_id} not found!")
        return {}
    
    alert = alert_info[0]
    customer_id = alert['customer_id']
    alert_type = alert['alert_type']
    severity = alert['severity']
    
    print()
    print("═" * 70)
    print(f"  🚨 ALERT: {alert_id} | Customer: {customer_id}")
    print(f"  Type: {alert_type} | Severity: {severity}")
    print(f"  Model: {MODEL_ENDPOINT}")
    print("═" * 70)
    print("  Launching 7 agents (5 parallel + narrative + decision)...")
    print()
    
    # Parallel agents
    parallel_results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(run_transaction_analyst, customer_id, alert_id): "transaction_analyst",
            executor.submit(run_customer_history_agent, customer_id): "customer_history",
            executor.submit(run_network_graph_agent, customer_id): "network_graph",
            executor.submit(run_sanctions_agent, customer_id): "sanctions",
            executor.submit(run_knowledge_assistant_agent, customer_id, alert_type): "knowledge_assistant",
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                result = future.result(timeout=60)
                parallel_results.append(result)
                icon = {"legitimate": "🟢", "suspicious": "🟡", "escalate": "🟠", "block": "🔴"}.get(result.verdict, "⚪")
                print(f"  {icon} {result.agent_name:<25} | {result.duration_ms:>5}ms | {result.verdict:<12} | {result.confidence:.0%}")
            except Exception as e:
                print(f"  ❌ {name:<25} | FAILED: {e}")
                parallel_results.append(AgentResult(agent_name=name, verdict="escalate", confidence=0.5, reasoning=str(e)))
    
    # Narrative + Decision
    print()
    narrative = run_narrative_writer(customer_id, parallel_results)
    print(f"  ✍️  {narrative.agent_name:<25} | {narrative.duration_ms:>5}ms | {narrative.verdict:<12} | {narrative.confidence:.0%}")
    
    all_results = parallel_results + [narrative]
    decision = run_decision_agent(all_results)
    all_results.append(decision)
    
    total_ms = int((time.time() - total_start) * 1000)
    total_tokens = sum(r.tokens_used for r in all_results)
    agreement = sum(1 for r in all_results if r.verdict == decision.verdict)
    
    ve = {"legitimate": "✅", "suspicious": "⚠️", "escalate": "🚨", "block": "🚫"}.get(decision.verdict, "❓")
    print()
    print(f"  ╔{'═'*62}╗")
    print(f"  ║  {ve} VERDICT: {decision.verdict.upper():<14}| Confidence: {decision.confidence:.0%}          ║")
    print(f"  ║  Consensus: {agreement}/{len(all_results)} agents     | Time: {total_ms:,}ms | Tokens: {total_tokens:,}  ║")
    print(f"  ╚{'═'*62}╝")
    print()
    
    return {
        "alert_id": alert_id, "customer_id": customer_id,
        "final_verdict": decision.verdict, "confidence": decision.confidence,
        "consensus": f"{agreement}/{len(all_results)}",
        "total_time_ms": total_ms, "total_tokens": total_tokens,
        "agent_results": all_results
    }

print("✅ Orchestrator: investigate_alert() [7 agents | UC tools + Genie + LLM]")

# COMMAND ----------

# DBTITLE 1,Audit Trail Logger
# =============================================================================
# AUDIT TRAIL LOGGER
# Writes results to Unity Catalog for traceability
# =============================================================================
from pyspark.sql.types import *

def log_to_unity_catalog(alert_id: str, customer_id: str, results: List[AgentResult]):
    case_id = f"CASE-LIVE-{alert_id}"
    trace_id = str(uuid.uuid4())
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    actions_data = [(
        f"ACT-{uuid.uuid4().hex[:8]}", case_id, r.agent_name, "analysis", now_str,
        f"Alert {alert_id} for {customer_id}", r.reasoning[:500],
        r.confidence, r.duration_ms, trace_id, MODEL_ENDPOINT, r.tokens_used
    ) for r in results]
    
    schema = StructType([
        StructField("action_id", StringType()), StructField("case_id", StringType()),
        StructField("agent_name", StringType()), StructField("action_type", StringType()),
        StructField("timestamp", StringType()), StructField("input_summary", StringType()),
        StructField("output_summary", StringType()), StructField("confidence_score", DoubleType()),
        StructField("duration_ms", IntegerType()), StructField("trace_id", StringType()),
        StructField("model_used", StringType()), StructField("tokens_consumed", IntegerType())
    ])
    
    df = spark.createDataFrame(actions_data, schema=schema)
    df = df.withColumn("timestamp", F.to_timestamp("timestamp"))
    df.write.mode("append").saveAsTable(f"{FQN}.case_actions")
    print(f"  💾 Logged to {FQN}.case_actions (trace: {trace_id[:8]}...)")

print("✅ Audit Logger defined")

# COMMAND ----------

# DBTITLE 1,DEMO - Investigate structuring alert
# =============================================================================
# 🎬 RUN THE DEMO - Structuring Ring Investigation
# =============================================================================

alert_id = "ALT-000001"  # Structuring ring alert (hot demo scenario)

print("=" * 70)
print("  🏛️  FINANCIAL CRIME WAR ROOM — LIVE INVESTIGATION")
print("=" * 70)

results = investigate_alert(alert_id)

# COMMAND ----------

# DBTITLE 1,View Case Narrative
# =============================================================================
# EXPLORE THE NARRATIVE
# View the full SAR-style narrative generated by the Narrative Writer
# =============================================================================

if results and 'agent_results' in results:
    narrative_agent = next(
        (r for r in results['agent_results'] if r.agent_name == 'narrative_writer'), 
        None
    )
    if narrative_agent:
        print("=" * 70)
        print("  📝 CASE NARRATIVE (SAR-style)")
        print("=" * 70)
        print()
        print(narrative_agent.reasoning)
        print()
        print("-" * 70)
        print(f"  Generated in {narrative_agent.duration_ms}ms | "
              f"{narrative_agent.tokens_used} tokens | "
              f"Verdict: {narrative_agent.verdict} ({narrative_agent.confidence:.0%})")
    else:
        print("No narrative found in results.")
else:
    print("Run the investigation cell above first.")

# COMMAND ----------

# DBTITLE 1,Explore MLflow Traces
# =============================================================================
# VIEW MLFLOW TRACES
# Click any trace to see the full agent reasoning chain
# =============================================================================

import mlflow

# Get the experiment ID for our war room traces
experiment = mlflow.get_experiment_by_name("/Users/tian.tan@databricks.com/fraud_war_room/war_room_traces")

if experiment:
    traces = mlflow.search_traces(
        experiment_ids=[experiment.experiment_id],
        max_results=5
    )
    if traces is not None and len(traces) > 0:
        display(traces)
    else:
        print("No traces found yet. Run an investigation first.")
else:
    print("Experiment not found. Run an investigation first.")

# COMMAND ----------

# DBTITLE 1,BATCH DEMO - 3 Hot Scenarios
# =============================================================================
# 🎬 BATCH DEMO - Run 3 Hot Scenarios Side by Side
# Structuring Ring → Sanctions Hit → Velocity Anomaly
# =============================================================================

hot_alerts = [
    ("ALT-000001", "Structuring Ring"),
    ("ALT-000010", "Sanctions Proximity"),
    ("ALT-000025", "Velocity Anomaly"),
]

print("=" * 70)
print("  🎬 BATCH INVESTIGATION — 3 Hot Scenarios")
print("=" * 70)

batch_results = []
for alert_id, scenario_name in hot_alerts:
    print(f"\n{'='*70}")
    print(f"  SCENARIO: {scenario_name}")
    print(f"{'='*70}")
    result = investigate_alert(alert_id)
    batch_results.append((scenario_name, result))

# ─── Summary Comparison ───
print("\n")
print("=" * 70)
print("  📊 INVESTIGATION SUMMARY - ALL SCENARIOS")
print("=" * 70)
print(f"\n  {'Scenario':<25} {'Verdict':<14} {'Confidence':<12} {'Time':<10} {'Tokens'}")
print(f"  {'-'*25} {'-'*14} {'-'*12} {'-'*10} {'-'*8}")

for scenario_name, result in batch_results:
    if result:
        print(f"  {scenario_name:<25} {result['final_verdict']:<14} "
              f"{result['confidence']:.0%:<12} {result['total_time_ms']:,}ms{'':<4} "
              f"{result['total_tokens']:,}")

print(f"\n  Total alerts processed: {len(batch_results)}")
print(f"  Auto-resolved: {sum(1 for _, r in batch_results if r and r.get('final_verdict') == 'legitimate')}")
print(f"  Escalated: {sum(1 for _, r in batch_results if r and r.get('final_verdict') in ('escalate', 'suspicious', 'block'))}")
print("=" * 70)