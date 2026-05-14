# Databricks notebook source
# DBTITLE 1,Tools Overview
# MAGIC %md
# MAGIC # 🧰 War Room Agent Tools — Unity Catalog Functions
# MAGIC
# MAGIC Reusable tool definitions for the Multi-Agent Financial Crime War Room,
# MAGIC registered as **Unity Catalog functions** so they can be:
# MAGIC
# MAGIC 1. Called from any notebook via `SELECT * FROM func(params)`
# MAGIC 2. Referenced by the **Supervisor Agent API** as `uc_function` tools
# MAGIC 3. Shared across teams with UC permissions (GRANT EXECUTE)
# MAGIC 4. Discovered in Catalog Explorer with descriptions
# MAGIC
# MAGIC ## Tools Created
# MAGIC
# MAGIC | UC Function | Purpose | Used By |
# MAGIC |-------------|---------|--------|
# MAGIC | `get_alert_details(alert_id)` | Look up alert: customer, type, severity | All agents |
# MAGIC | `get_transaction_analysis(customer_id)` | Aggregate txn stats: structuring, velocity | Transaction Analyst |
# MAGIC | `get_high_risk_transactions(customer_id)` | Individual flagged transactions | Transaction Analyst |
# MAGIC | `get_customer_risk_profile(customer_id)` | KYC, PEP, risk score, alert/case counts | Customer History |
# MAGIC | `get_network_analysis(customer_id)` | 1st-hop graph: connections, clusters | Network Graph |
# MAGIC | `screen_sanctions_and_media(customer_id)` | Sanctions + adverse media hits | Sanctions Agent |
# MAGIC | `get_similar_cases(typology)` | Historical case precedents by typology | Knowledge Assistant |
# MAGIC
# MAGIC ## Platform Tools (not UC functions — referenced by ID)
# MAGIC
# MAGIC | Tool | Type | ID |
# MAGIC |------|------|----|
# MAGIC | Genie Space | `genie_space` | `01f14f0e4ec91ad5ba608f3ae4b432e3` |
# MAGIC | Knowledge Assistant | `knowledge_assistant` | `8c765842-f021-4946-85b4-df326cc01b03` |
# MAGIC
# MAGIC **Location:** `ttan_demo_catalog_main.fsi_fraud_war_room.*`

# COMMAND ----------

# DBTITLE 1,Configuration and imports
# ─── Configuration ───
CATALOG = "ttan_demo_catalog_main"
SCHEMA = "fsi_fraud_war_room"
FQN = f"{CATALOG}.{SCHEMA}"

# Foundation Model API
MODEL_ENDPOINT = "databricks-claude-sonnet-4-5"

# Platform resource IDs (from 01_setup_infrastructure)
GENIE_SPACE_ID = "01f14f0e4ec91ad5ba608f3ae4b432e3"
KA_ID = "8c765842-f021-4946-85b4-df326cc01b03"

print(f"✅ Target schema: {FQN}")
print(f"   Model: {MODEL_ENDPOINT}")
print(f"   Genie Space: {GENIE_SPACE_ID}")
print(f"   Knowledge Assistant: {KA_ID}")

# COMMAND ----------

# DBTITLE 1,alert lookup tool
# MAGIC %sql
# MAGIC -- =============================================================================
# MAGIC -- TOOL: get_alert_details
# MAGIC -- Starting point for any investigation. Returns alert context.
# MAGIC -- =============================================================================
# MAGIC
# MAGIC CREATE OR REPLACE FUNCTION ttan_demo_catalog_main.fsi_fraud_war_room.get_alert_details(
# MAGIC     p_alert_id STRING COMMENT 'The alert ID to look up (e.g. ALT-000001)'
# MAGIC )
# MAGIC RETURNS TABLE (
# MAGIC     alert_id STRING,
# MAGIC     customer_id STRING,
# MAGIC     transaction_id STRING,
# MAGIC     alert_type STRING,
# MAGIC     severity STRING,
# MAGIC     status STRING,
# MAGIC     assigned_agent STRING,
# MAGIC     alert_timestamp TIMESTAMP
# MAGIC )
# MAGIC COMMENT 'Look up alert details including the customer involved, alert type (structuring, velocity, sanctions_hit, etc.), severity, and triggering transaction. This is the starting point for any fraud investigation.'
# MAGIC RETURN
# MAGIC     SELECT alert_id, customer_id, transaction_id, alert_type, 
# MAGIC            severity, status, assigned_agent, alert_timestamp
# MAGIC     FROM ttan_demo_catalog_main.fsi_fraud_war_room.alerts
# MAGIC     WHERE alert_id = p_alert_id

# COMMAND ----------

# DBTITLE 1,transaction analysis tool
# MAGIC %sql
# MAGIC -- =============================================================================
# MAGIC -- TOOL: get_transaction_analysis
# MAGIC -- Aggregate transaction statistics for fraud pattern detection
# MAGIC -- =============================================================================
# MAGIC
# MAGIC CREATE OR REPLACE FUNCTION ttan_demo_catalog_main.fsi_fraud_war_room.get_transaction_analysis(
# MAGIC     p_customer_id STRING COMMENT 'The customer ID to analyze transactions for'
# MAGIC )
# MAGIC RETURNS TABLE (
# MAGIC     txn_count BIGINT,
# MAGIC     total_amount DOUBLE,
# MAGIC     avg_amount DOUBLE,
# MAGIC     max_amount DOUBLE,
# MAGIC     near_10k_count BIGINT,
# MAGIC     over_10k_count BIGINT,
# MAGIC     wire_count BIGINT,
# MAGIC     intl_count BIGINT,
# MAGIC     flagged_count BIGINT,
# MAGIC     active_days BIGINT,
# MAGIC     unique_countries BIGINT,
# MAGIC     unique_devices BIGINT,
# MAGIC     daily_velocity DOUBLE
# MAGIC )
# MAGIC COMMENT 'Analyze transaction patterns for fraud indicators: structuring (near-$10K threshold), velocity anomalies, geographic concentration, wire transfer ratio. Returns aggregate statistics over all transactions for the customer. Key thresholds: near_10k_count > 3 = structuring, daily_velocity > 5 = velocity anomaly.'
# MAGIC RETURN
# MAGIC     SELECT 
# MAGIC         COUNT(*) as txn_count,
# MAGIC         ROUND(SUM(amount), 2) as total_amount,
# MAGIC         ROUND(AVG(amount), 2) as avg_amount,
# MAGIC         ROUND(MAX(amount), 2) as max_amount,
# MAGIC         COUNT(CASE WHEN amount BETWEEN 9000 AND 9999 THEN 1 END) as near_10k_count,
# MAGIC         COUNT(CASE WHEN amount >= 10000 THEN 1 END) as over_10k_count,
# MAGIC         COUNT(CASE WHEN transaction_type = 'wire' THEN 1 END) as wire_count,
# MAGIC         COUNT(CASE WHEN is_international = true THEN 1 END) as intl_count,
# MAGIC         COUNT(CASE WHEN fraud_label = 1 THEN 1 END) as flagged_count,
# MAGIC         COUNT(DISTINCT DATE(timestamp)) as active_days,
# MAGIC         COUNT(DISTINCT counterparty_country) as unique_countries,
# MAGIC         COUNT(DISTINCT device_id) as unique_devices,
# MAGIC         ROUND(COUNT(*) * 1.0 / GREATEST(COUNT(DISTINCT DATE(timestamp)), 1), 2) as daily_velocity
# MAGIC     FROM ttan_demo_catalog_main.fsi_fraud_war_room.transactions 
# MAGIC     WHERE customer_id = p_customer_id

# COMMAND ----------

# DBTITLE 1,high risk transaction tool
# MAGIC %sql
# MAGIC -- =============================================================================
# MAGIC -- TOOL: get_high_risk_transactions
# MAGIC -- Individual flagged transactions for evidence collection
# MAGIC -- =============================================================================
# MAGIC
# MAGIC CREATE OR REPLACE FUNCTION ttan_demo_catalog_main.fsi_fraud_war_room.get_high_risk_transactions(
# MAGIC     p_customer_id STRING COMMENT 'The customer ID',
# MAGIC     p_min_risk_score INT DEFAULT 60 COMMENT 'Minimum risk score threshold (0-100)'
# MAGIC )
# MAGIC RETURNS TABLE (
# MAGIC     transaction_id STRING,
# MAGIC     amount DOUBLE,
# MAGIC     transaction_type STRING,
# MAGIC     counterparty_country STRING,
# MAGIC     risk_score INT,
# MAGIC     timestamp TIMESTAMP,
# MAGIC     channel STRING,
# MAGIC     is_international BOOLEAN
# MAGIC )
# MAGIC COMMENT 'Retrieve individual high-risk transactions above the risk score threshold. Use after get_transaction_analysis reveals suspicious patterns. Shows specific transactions that triggered risk flags — useful for evidence citation in SAR narratives.'
# MAGIC RETURN
# MAGIC     SELECT transaction_id, amount, transaction_type, counterparty_country,
# MAGIC            risk_score, timestamp, channel, is_international
# MAGIC     FROM ttan_demo_catalog_main.fsi_fraud_war_room.transactions
# MAGIC     WHERE customer_id = p_customer_id AND risk_score >= p_min_risk_score
# MAGIC     ORDER BY timestamp DESC
# MAGIC     LIMIT 20

# COMMAND ----------

# DBTITLE 1,customer risk profile tool
# MAGIC %sql
# MAGIC -- =============================================================================
# MAGIC -- TOOL: get_customer_risk_profile
# MAGIC -- Full KYC profile with alert/case history counts
# MAGIC -- =============================================================================
# MAGIC
# MAGIC CREATE OR REPLACE FUNCTION ttan_demo_catalog_main.fsi_fraud_war_room.get_customer_risk_profile(
# MAGIC     p_customer_id STRING COMMENT 'The customer ID to retrieve profile for'
# MAGIC )
# MAGIC RETURNS TABLE (
# MAGIC     customer_id STRING,
# MAGIC     name STRING,
# MAGIC     segment STRING,
# MAGIC     country STRING,
# MAGIC     nationality STRING,
# MAGIC     occupation STRING,
# MAGIC     annual_income DOUBLE,
# MAGIC     source_of_funds STRING,
# MAGIC     account_open_date STRING,
# MAGIC     kyc_status STRING,
# MAGIC     pep_flag BOOLEAN,
# MAGIC     risk_score INT,
# MAGIC     alert_count BIGINT,
# MAGIC     case_count BIGINT,
# MAGIC     open_cases BIGINT
# MAGIC )
# MAGIC COMMENT 'Retrieve full customer KYC profile with PEP status, risk score, occupation, income, and alert/case history counts. Use to assess whether activity aligns with stated profile. PEP + elevated activity always requires escalation.'
# MAGIC RETURN
# MAGIC     SELECT 
# MAGIC         c.customer_id, c.name, c.segment, c.country, c.nationality,
# MAGIC         c.occupation, c.annual_income, c.source_of_funds,
# MAGIC         CAST(c.account_open_date AS STRING) as account_open_date,
# MAGIC         c.kyc_status, c.pep_flag, c.risk_score,
# MAGIC         (SELECT COUNT(*) FROM ttan_demo_catalog_main.fsi_fraud_war_room.alerts a WHERE a.customer_id = c.customer_id) as alert_count,
# MAGIC         (SELECT COUNT(*) FROM ttan_demo_catalog_main.fsi_fraud_war_room.cases cs WHERE cs.customer_id = c.customer_id) as case_count,
# MAGIC         (SELECT COUNT(*) FROM ttan_demo_catalog_main.fsi_fraud_war_room.cases cs WHERE cs.customer_id = c.customer_id AND cs.status IN ('open', 'investigating')) as open_cases
# MAGIC     FROM ttan_demo_catalog_main.fsi_fraud_war_room.customers c
# MAGIC     WHERE c.customer_id = p_customer_id

# COMMAND ----------

# DBTITLE 1,network analysis tool
# MAGIC %sql
# MAGIC -- =============================================================================
# MAGIC -- TOOL: get_network_analysis
# MAGIC -- 1st-hop network graph for mule network detection
# MAGIC -- =============================================================================
# MAGIC
# MAGIC CREATE OR REPLACE FUNCTION ttan_demo_catalog_main.fsi_fraud_war_room.get_network_analysis(
# MAGIC     p_customer_id STRING COMMENT 'The customer ID to analyze network connections for'
# MAGIC )
# MAGIC RETURNS TABLE (
# MAGIC     connected_to STRING,
# MAGIC     relationship_type STRING,
# MAGIC     risk_score INT,
# MAGIC     cluster_id STRING,
# MAGIC     shared_attributes STRING
# MAGIC )
# MAGIC COMMENT 'Analyze 1st-hop network graph connections for mule network indicators. A normal customer has 3-5 connections. Dense clusters (>8 connections with high risk scores) may indicate organized crime or mule networks. Returns all direct connections with relationship type, cluster membership, and shared attributes.'
# MAGIC RETURN
# MAGIC     SELECT entity_id_2 as connected_to, relationship_type, 
# MAGIC            risk_score, cluster_id, shared_attributes
# MAGIC     FROM ttan_demo_catalog_main.fsi_fraud_war_room.network_relationships
# MAGIC     WHERE entity_id_1 = p_customer_id
# MAGIC     UNION ALL
# MAGIC     SELECT entity_id_1 as connected_to, relationship_type,
# MAGIC            risk_score, cluster_id, shared_attributes
# MAGIC     FROM ttan_demo_catalog_main.fsi_fraud_war_room.network_relationships
# MAGIC     WHERE entity_id_2 = p_customer_id

# COMMAND ----------

# DBTITLE 1,sanction and adverse media tool
# MAGIC %sql
# MAGIC -- =============================================================================
# MAGIC -- TOOL: screen_sanctions_and_media
# MAGIC -- Unified sanctions watchlist + adverse media screening
# MAGIC -- =============================================================================
# MAGIC
# MAGIC CREATE OR REPLACE FUNCTION ttan_demo_catalog_main.fsi_fraud_war_room.screen_sanctions_and_media(
# MAGIC     p_customer_id STRING COMMENT 'The customer ID to screen against sanctions and adverse media'
# MAGIC )
# MAGIC RETURNS TABLE (
# MAGIC     source_type STRING,
# MAGIC     entity_name STRING,
# MAGIC     entity_type STRING,
# MAGIC     list_source STRING,
# MAGIC     country STRING,
# MAGIC     risk_category STRING,
# MAGIC     severity STRING,
# MAGIC     summary STRING
# MAGIC )
# MAGIC COMMENT 'Screen customer against sanctions watchlists (OFAC, EU, UN, UK) and adverse media sources. Returns all hits from both sanctions lists and news. A sanctions match requires immediate escalation or block. Adverse media alone is suspicious but not conclusive. Also checks jurisdiction exposure.'
# MAGIC RETURN
# MAGIC     WITH customer_info AS (
# MAGIC         SELECT name, country, nationality 
# MAGIC         FROM ttan_demo_catalog_main.fsi_fraud_war_room.customers 
# MAGIC         WHERE customer_id = p_customer_id
# MAGIC     )
# MAGIC     SELECT 'sanctions' as source_type, s.entity_name, s.entity_type, 
# MAGIC            s.list_source, s.country, s.risk_category, 
# MAGIC            CAST(NULL AS STRING) as severity, CAST(NULL AS STRING) as summary
# MAGIC     FROM ttan_demo_catalog_main.fsi_fraud_war_room.sanctions_watchlist s, customer_info ci
# MAGIC     WHERE LOWER(s.country) = LOWER(ci.country) 
# MAGIC        OR LOWER(s.country) = LOWER(ci.nationality)
# MAGIC     UNION ALL
# MAGIC     SELECT 'adverse_media' as source_type, am.entity_name, CAST(NULL AS STRING) as entity_type,
# MAGIC            am.source as list_source, CAST(NULL AS STRING) as country, am.risk_category,
# MAGIC            am.severity, am.summary
# MAGIC     FROM ttan_demo_catalog_main.fsi_fraud_war_room.adverse_media am, customer_info ci
# MAGIC     WHERE LOWER(am.entity_name) LIKE CONCAT('%', LOWER(SPLIT(ci.name, ' ')[0]), '%')

# COMMAND ----------

# DBTITLE 1,historical case precedent tool
# MAGIC %sql
# MAGIC -- =============================================================================
# MAGIC -- TOOL: get_similar_cases
# MAGIC -- Historical case precedent retrieval by typology
# MAGIC -- =============================================================================
# MAGIC
# MAGIC CREATE OR REPLACE FUNCTION ttan_demo_catalog_main.fsi_fraud_war_room.get_similar_cases(
# MAGIC     p_typology STRING COMMENT 'The fraud typology to search for (structuring, layering, mule_network, sanctions_evasion, identity_fraud)',
# MAGIC     p_limit INT DEFAULT 5 COMMENT 'Number of similar cases to return'
# MAGIC )
# MAGIC RETURNS TABLE (
# MAGIC     case_id STRING,
# MAGIC     typology STRING,
# MAGIC     outcome STRING,
# MAGIC     amount_range STRING,
# MAGIC     jurisdiction STRING,
# MAGIC     resolution_time_hours INT,
# MAGIC     sar_filed BOOLEAN,
# MAGIC     embedding_text STRING
# MAGIC )
# MAGIC COMMENT 'Retrieve historical case precedents by fraud typology. Returns past investigation outcomes, SAR filing rates, and resolution timelines. Use to reason over what happened in similar past cases and inform the current verdict. Complements the Knowledge Assistant for direct SQL access to case data.'
# MAGIC RETURN
# MAGIC     SELECT case_id, typology, outcome, amount_range, jurisdiction,
# MAGIC            resolution_time_hours, sar_filed, embedding_text
# MAGIC     FROM ttan_demo_catalog_main.fsi_fraud_war_room.similar_past_cases
# MAGIC     WHERE typology = p_typology
# MAGIC     ORDER BY RAND()
# MAGIC     LIMIT p_limit

# COMMAND ----------

# DBTITLE 1,Tool Registry Summary
# =============================================================================
# TOOL REGISTRY SUMMARY
# =============================================================================

uc_tools = [
    f"{FQN}.get_alert_details",
    f"{FQN}.get_transaction_analysis",
    f"{FQN}.get_high_risk_transactions",
    f"{FQN}.get_customer_risk_profile",
    f"{FQN}.get_network_analysis",
    f"{FQN}.screen_sanctions_and_media",
    f"{FQN}.get_similar_cases",
]

platform_tools = {
    "genie_space": GENIE_SPACE_ID,
    "knowledge_assistant": KA_ID,
}

print("\n" + "=" * 60)
print("  🧰  TOOL REGISTRY (Unity Catalog Functions)")
print("=" * 60)
for fn in uc_tools:
    print(f"   • {fn.split('.')[-1]:<35} [uc_function]")
print()
for tool_type, tool_id in platform_tools.items():
    print(f"   • {tool_type:<35} [{tool_id[:16]}...]")
print("=" * 60)
print(f"\n  Total: {len(uc_tools)} UC functions + {len(platform_tools)} platform tools")
print(f"  Schema: {FQN}")
print(f"\n  Usage in SQL:  SELECT * FROM {FQN}.get_alert_details('ALT-000001')")
print(f"  Usage in SDK:  tool_type='uc_function', name='{FQN}.get_alert_details'")