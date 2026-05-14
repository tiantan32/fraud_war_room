# Databricks notebook source
# DBTITLE 1,Header - War Room Overview
# MAGIC %md
# MAGIC # Multi-Agent Financial Crime War Room — Synthetic Data Generation
# MAGIC
# MAGIC This notebook generates all synthetic datasets required for the **Multi-Agent Financial Crime War Room** demo. The war room simulates a fraud alert triggering a swarm of 6 specialist AI agents that investigate in parallel, build a case file, and converge on a verdict — all with full audit trail in Unity Catalog.
# MAGIC
# MAGIC **Architecture:**
# MAGIC - **Agents:** Transaction Analyst, Customer-History Agent, Network-Graph Agent, News/Sanctions Agent, Narrative-Writer Agent, Decision Agent
# MAGIC - **Platform:** Genie (ad-hoc SQL), Vector Search (similar past cases), Lakebase (case state), AI Gateway (governance), MLflow Tracing (audit)
# MAGIC - **MCPs/Tools:** OFAC/Sanctions MCP, News MCP, Internal CRM MCP, Email MCP
# MAGIC
# MAGIC **Tables Generated:**
# MAGIC | Table | Description | ~Rows |
# MAGIC |-------|-------------|-------|
# MAGIC | `customers` | Customer master with KYC/PEP | 5,000 |
# MAGIC | `transactions` | 90 days of transaction history | 100,000 |
# MAGIC | `alerts` | Fraud/AML alerts | 2,000 |
# MAGIC | `network_relationships` | Entity graph edges | 15,000 |
# MAGIC | `sanctions_watchlist` | OFAC/EU/UN watchlist | 500 |
# MAGIC | `cases` | Case management state | 800 |
# MAGIC | `case_actions` | Agent audit trail | 5,000 |
# MAGIC | `similar_past_cases` | Historical cases for Vector Search | 500 |
# MAGIC | `adverse_media` | News/sanctions media hits | 1,000 |
# MAGIC | `agent_verdicts` | Decision convergence data | 200 |

# COMMAND ----------

# DBTITLE 1,Install dependencies
# MAGIC %pip install faker --quiet

# COMMAND ----------

# DBTITLE 1,Imports and configuration
import random
import uuid
import hashlib
from datetime import datetime, timedelta
from faker import Faker
from pyspark.sql import functions as F
from pyspark.sql.types import *

fake = Faker()
Faker.seed(42)
random.seed(42)

# Configuration
CATALOG = "ttan_demo_catalog_main"
SCHEMA = "fsi_fraud_war_room"
NUM_CUSTOMERS = 5000
NUM_TRANSACTIONS = 100000
NUM_ALERTS = 2000
NUM_NETWORK_EDGES = 15000
NUM_SANCTIONS = 500
NUM_CASES = 800
NUM_CASE_ACTIONS = 5000
NUM_PAST_CASES = 500
NUM_ADVERSE_MEDIA = 1000
NUM_VERDICTS = 200

# Time boundaries
NOW = datetime(2026, 5, 13, 14, 0, 0)
NINETY_DAYS_AGO = NOW - timedelta(days=90)

# High-risk countries for realistic patterns
HIGH_RISK_COUNTRIES = ["Iran", "North Korea", "Syria", "Myanmar", "Russia", "Venezuela", "Cuba", "Sudan", "Libya", "Somalia"]
NORMAL_COUNTRIES = ["United States", "United Kingdom", "Canada", "Germany", "France", "Japan", "Australia", "Singapore", "Switzerland", "Netherlands"]

print(f"Configuration loaded. Target: {CATALOG}.{SCHEMA}")
print(f"Time window: {NINETY_DAYS_AGO.strftime('%Y-%m-%d')} to {NOW.strftime('%Y-%m-%d')}")

# COMMAND ----------

# DBTITLE 1,Create catalog and schema
# MAGIC %sql
# MAGIC CREATE CATALOG IF NOT EXISTS ttan_demo_catalog_main;
# MAGIC CREATE SCHEMA IF NOT EXISTS ttan_demo_catalog_main.fsi_fraud_war_room
# MAGIC COMMENT 'Synthetic data for the Multi-Agent Financial Crime War Room demo. Contains transactions, customers, alerts, cases, and agent audit trails for fraud investigation simulation.';

# COMMAND ----------

# DBTITLE 1,Generate customers table
# =============================================================================
# TABLE 1: CUSTOMERS (~5,000 rows)
# =============================================================================

segments = ["retail", "commercial", "private_banking"]
segment_weights = [0.7, 0.2, 0.1]
kyc_statuses = ["verified", "pending_review", "expired", "enhanced_due_diligence"]
occupations = [
    "Software Engineer", "Doctor", "Lawyer", "Business Owner", "Retired",
    "Government Official", "Real Estate Agent", "Trader", "Accountant", "Consultant",
    "Import/Export", "Restaurant Owner", "Crypto Entrepreneur", "Diplomat", "Journalist",
    "Construction", "Professor", "Pharmacist", "Banker", "Artist"
]
source_of_funds_options = [
    "employment", "business_income", "inheritance", "investments", "real_estate",
    "pension", "gift", "lottery", "crypto_trading", "unknown"
]

customers_data = []
for i in range(NUM_CUSTOMERS):
    customer_id = f"CUST-{i+1:06d}"
    segment = random.choices(segments, weights=segment_weights, k=1)[0]
    
    # Higher risk scores for certain profiles
    base_risk = random.randint(1, 40)
    if segment == "private_banking":
        base_risk = random.randint(10, 60)
    
    # PEP flag (2% of customers)
    pep_flag = random.random() < 0.02
    if pep_flag:
        base_risk = min(100, base_risk + random.randint(20, 40))
    
    # Some customers from high-risk jurisdictions
    if random.random() < 0.05:
        country = random.choice(HIGH_RISK_COUNTRIES)
        nationality = country
        base_risk = min(100, base_risk + random.randint(15, 35))
    else:
        country = "United States"
        nationality = random.choice(["United States", "United Kingdom", "Canada", "India", "China", "Brazil", "Germany", "France", "Nigeria", "Mexico"])
    
    # Income varies by segment
    if segment == "private_banking":
        annual_income = random.randint(500000, 10000000)
    elif segment == "commercial":
        annual_income = random.randint(100000, 2000000)
    else:
        annual_income = random.randint(25000, 250000)
    
    dob = fake.date_of_birth(minimum_age=22, maximum_age=80)
    account_open = fake.date_between(start_date='-10y', end_date='-30d')
    
    customers_data.append((
        customer_id,
        fake.name(),
        fake.email(),
        fake.phone_number(),
        fake.street_address(),
        fake.city(),
        fake.state_abbr(),
        country,
        str(dob),
        str(account_open),
        min(100, max(1, base_risk)),
        segment,
        random.choice(kyc_statuses),
        pep_flag,
        nationality,
        random.choice(occupations),
        annual_income,
        random.choice(source_of_funds_options)
    ))

customers_schema = StructType([
    StructField("customer_id", StringType()),
    StructField("name", StringType()),
    StructField("email", StringType()),
    StructField("phone", StringType()),
    StructField("address", StringType()),
    StructField("city", StringType()),
    StructField("state", StringType()),
    StructField("country", StringType()),
    StructField("date_of_birth", StringType()),
    StructField("account_open_date", StringType()),
    StructField("risk_score", IntegerType()),
    StructField("segment", StringType()),
    StructField("kyc_status", StringType()),
    StructField("pep_flag", BooleanType()),
    StructField("nationality", StringType()),
    StructField("occupation", StringType()),
    StructField("annual_income", LongType()),
    StructField("source_of_funds", StringType())
])

df_customers = spark.createDataFrame(customers_data, schema=customers_schema)
df_customers = df_customers.withColumn("date_of_birth", F.to_date("date_of_birth")) \
    .withColumn("account_open_date", F.to_date("account_open_date"))

df_customers.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.customers")
spark.sql(f"COMMENT ON TABLE {CATALOG}.{SCHEMA}.customers IS 'Customer master table with KYC, PEP flags, and risk scoring for fraud war room demo'")
print(f"✅ customers: {df_customers.count():,} rows")

# COMMAND ----------

# DBTITLE 1,Generate transactions table
# =============================================================================
# TABLE 2: TRANSACTIONS (~100,000 rows, last 90 days)
# Includes realistic fraud patterns:
#   - Structuring (amounts just under $10K)
#   - Layering (rapid in/out)
#   - Velocity spikes
#   - Geographic anomalies
#   - Round-dollar amounts to high-risk jurisdictions
# =============================================================================

txn_types = ["wire", "ach", "card", "crypto", "check"]
txn_type_weights = [0.15, 0.30, 0.35, 0.10, 0.10]
directions = ["inbound", "outbound"]
channels = ["online", "branch", "atm", "mobile"]
channel_weights = [0.40, 0.15, 0.15, 0.30]
merchant_categories = [
    "retail", "restaurants", "travel", "utilities", "entertainment",
    "healthcare", "education", "financial_services", "real_estate",
    "crypto_exchange", "gambling", "money_service_business", "jewelry",
    "electronics", "automotive"
]
currencies = ["USD", "EUR", "GBP", "CHF", "JPY", "CAD", "AUD", "SGD"]
currency_weights = [0.65, 0.12, 0.08, 0.03, 0.04, 0.03, 0.03, 0.02]

# Collect customer IDs for FK consistency
customer_ids = [row.customer_id for row in df_customers.select("customer_id").collect()]

# === HOT DEMO SCENARIOS ===
# Scenario 1: Structuring ring - 5 customers making transactions just under $10K
structuring_customers = customer_ids[100:105]
# Scenario 2: High-value velocity anomaly - private banking customer
velocity_customer = customer_ids[4500]  # private_banking segment area
# Scenario 3: Sanctions-adjacent - customer transacting with entities in Iran
sanctions_customer = customer_ids[200]

transactions_data = []
fraud_txn_ids = []  # Track for alerts

for i in range(NUM_TRANSACTIONS):
    txn_id = f"TXN-{i+1:08d}"
    
    # Determine if this is a fraud transaction (~2%)
    is_fraud = random.random() < 0.02
    
    # Pick customer - bias fraud toward hot scenario customers
    if is_fraud and random.random() < 0.4:
        cust_id = random.choice(structuring_customers + [velocity_customer, sanctions_customer])
    else:
        cust_id = random.choice(customer_ids)
    
    # Timestamp within last 90 days
    ts = NINETY_DAYS_AGO + timedelta(
        seconds=random.randint(0, int((NOW - NINETY_DAYS_AGO).total_seconds()))
    )
    
    # Transaction characteristics based on fraud pattern
    if is_fraud:
        fraud_pattern = random.choice(["structuring", "layering", "geo_anomaly", "velocity", "round_dollar"])
        
        if fraud_pattern == "structuring":
            # Amounts just under $10K CTR threshold
            amount = round(random.uniform(9000, 9999), 2)
            txn_type = random.choice(["cash", "check", "wire"])
            cust_id = random.choice(structuring_customers) if random.random() < 0.6 else cust_id
        elif fraud_pattern == "layering":
            # Rapid movement, varied amounts
            amount = round(random.uniform(15000, 150000), 2)
            txn_type = "wire"
        elif fraud_pattern == "geo_anomaly":
            # Transactions to/from high-risk jurisdictions
            amount = round(random.uniform(5000, 500000), 2)
            txn_type = "wire"
            cust_id = sanctions_customer if random.random() < 0.3 else cust_id
        elif fraud_pattern == "velocity":
            # Unusually high number/amount in short period
            amount = round(random.uniform(1000, 50000), 2)
            txn_type = random.choice(["wire", "ach", "crypto"])
            cust_id = velocity_customer if random.random() < 0.4 else cust_id
        else:  # round_dollar
            # Suspiciously round amounts
            amount = float(random.choice([10000, 25000, 50000, 75000, 100000, 250000]))
            txn_type = "wire"
        
        fraud_txn_ids.append(txn_id)
        risk_score = random.randint(60, 100)
    else:
        # Normal transaction
        amount = round(random.expovariate(1/500) + 10, 2)  # Exponential distribution
        amount = min(amount, 50000)  # Cap normal transactions
        txn_type = random.choices(txn_types, weights=txn_type_weights, k=1)[0]
        risk_score = random.randint(1, 40)
    
    direction = random.choice(directions)
    channel = random.choices(channels, weights=channel_weights, k=1)[0]
    
    # Counterparty
    if is_fraud and fraud_pattern == "geo_anomaly":
        cp_country = random.choice(HIGH_RISK_COUNTRIES)
    else:
        cp_country = random.choices(
            NORMAL_COUNTRIES + HIGH_RISK_COUNTRIES,
            weights=[0.09]*10 + [0.01]*10, k=1
        )[0]
    
    is_international = cp_country != "United States"
    currency = random.choices(currencies, weights=currency_weights, k=1)[0]
    
    # Generate device/IP fingerprints
    device_id = hashlib.md5(f"{cust_id}-device-{random.randint(1,3)}".encode()).hexdigest()[:16]
    ip_address = fake.ipv4()
    
    transactions_data.append((
        txn_id, cust_id, ts.strftime("%Y-%m-%d %H:%M:%S"), amount, currency,
        txn_type, direction, fake.company(), fake.bban(), cp_country,
        channel, random.choice(merchant_categories), ip_address, device_id,
        is_international, risk_score, int(is_fraud)
    ))

txn_schema = StructType([
    StructField("transaction_id", StringType()),
    StructField("customer_id", StringType()),
    StructField("timestamp", StringType()),
    StructField("amount", DoubleType()),
    StructField("currency", StringType()),
    StructField("transaction_type", StringType()),
    StructField("direction", StringType()),
    StructField("counterparty_name", StringType()),
    StructField("counterparty_account", StringType()),
    StructField("counterparty_country", StringType()),
    StructField("channel", StringType()),
    StructField("merchant_category", StringType()),
    StructField("ip_address", StringType()),
    StructField("device_id", StringType()),
    StructField("is_international", BooleanType()),
    StructField("risk_score", IntegerType()),
    StructField("fraud_label", IntegerType())
])

df_transactions = spark.createDataFrame(transactions_data, schema=txn_schema)
df_transactions = df_transactions.withColumn("timestamp", F.to_timestamp("timestamp"))

df_transactions.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.transactions")
spark.sql(f"COMMENT ON TABLE {CATALOG}.{SCHEMA}.transactions IS 'Transaction history (90 days) with fraud labels. Includes structuring, layering, velocity, and geographic anomaly patterns.'")

fraud_count = df_transactions.filter(F.col("fraud_label") == 1).count()
total_count = df_transactions.count()
print(f"✅ transactions: {total_count:,} rows ({fraud_count:,} fraudulent = {fraud_count/total_count*100:.1f}%)")

# COMMAND ----------

# DBTITLE 1,Generate alerts table
# =============================================================================
# TABLE 3: ALERTS (~2,000 rows)
# Some alerts are open/new - these trigger the war room
# =============================================================================

alert_types = ["velocity", "structuring", "sanctions_hit", "unusual_geography", 
               "amount_anomaly", "network_risk", "behavioral"]
alert_type_weights = [0.20, 0.20, 0.10, 0.15, 0.15, 0.10, 0.10]
severities = ["critical", "high", "medium", "low"]
severity_weights = [0.05, 0.20, 0.45, 0.30]
statuses = ["new", "investigating", "escalated", "resolved_legitimate", "resolved_sar"]
status_weights = [0.10, 0.08, 0.07, 0.55, 0.20]

agent_names_assignment = [
    "Agent Smith", "Agent Johnson", "Agent Williams", "Agent Brown",
    "Agent Davis", "Agent Miller", "Agent Wilson", "Agent Moore",
    "Auto-Triage Bot", None
]

resolution_notes_templates = [
    "Customer confirmed legitimate business transaction with counterparty.",
    "Pattern consistent with known money laundering typology. SAR filed.",
    "Velocity spike due to payroll processing - confirmed with employer.",
    "Structuring pattern detected across 3 accounts. Escalated to BSA team.",
    "Geographic anomaly explained by customer travel records.",
    "Network analysis revealed connection to known fraud ring.",
    "Behavioral deviation within normal seasonal patterns.",
    "Sanctions screening produced false positive - name similarity only.",
    "Amount anomaly triggered by one-time real estate closing.",
    "Multiple alerts consolidated into single SAR filing."
]

alerts_data = []
for i in range(NUM_ALERTS):
    alert_id = f"ALT-{i+1:06d}"
    
    # Use fraud transactions for some alerts
    if i < len(fraud_txn_ids) and random.random() < 0.7:
        txn_id = fraud_txn_ids[i % len(fraud_txn_ids)]
        # Find the customer for this transaction
        cust_id = random.choice(structuring_customers + [velocity_customer, sanctions_customer])
    else:
        txn_id = f"TXN-{random.randint(1, NUM_TRANSACTIONS):08d}"
        cust_id = random.choice(customer_ids)
    
    alert_type = random.choices(alert_types, weights=alert_type_weights, k=1)[0]
    severity = random.choices(severities, weights=severity_weights, k=1)[0]
    status = random.choices(statuses, weights=status_weights, k=1)[0]
    
    # Hot demo alerts - first 50 are new/investigating with high severity
    if i < 50:
        status = random.choice(["new", "investigating"])
        severity = random.choice(["critical", "high"])
        cust_id = random.choice(structuring_customers + [velocity_customer, sanctions_customer])
        if cust_id == sanctions_customer:
            alert_type = "sanctions_hit"
        elif cust_id in structuring_customers:
            alert_type = "structuring"
        else:
            alert_type = "velocity"
    
    alert_ts = NINETY_DAYS_AGO + timedelta(
        seconds=random.randint(0, int((NOW - NINETY_DAYS_AGO).total_seconds()))
    )
    
    # Resolution timestamp only for resolved alerts
    if status.startswith("resolved"):
        resolution_ts = (alert_ts + timedelta(hours=random.randint(1, 72))).strftime("%Y-%m-%d %H:%M:%S")
        resolution_notes = random.choice(resolution_notes_templates)
    else:
        resolution_ts = None
        resolution_notes = None
    
    assigned = random.choice(agent_names_assignment)
    
    alerts_data.append((
        alert_id, cust_id, txn_id, alert_ts.strftime("%Y-%m-%d %H:%M:%S"),
        alert_type, severity, status, assigned, resolution_ts, resolution_notes
    ))

alerts_schema = StructType([
    StructField("alert_id", StringType()),
    StructField("customer_id", StringType()),
    StructField("transaction_id", StringType()),
    StructField("alert_timestamp", StringType()),
    StructField("alert_type", StringType()),
    StructField("severity", StringType()),
    StructField("status", StringType()),
    StructField("assigned_agent", StringType()),
    StructField("resolution_timestamp", StringType()),
    StructField("resolution_notes", StringType())
])

df_alerts = spark.createDataFrame(alerts_data, schema=alerts_schema)
df_alerts = df_alerts.withColumn("alert_timestamp", F.to_timestamp("alert_timestamp")) \
    .withColumn("resolution_timestamp", F.to_timestamp("resolution_timestamp"))

df_alerts.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.alerts")
spark.sql(f"COMMENT ON TABLE {CATALOG}.{SCHEMA}.alerts IS 'Fraud/AML alerts with severity and status tracking. New/investigating alerts trigger the multi-agent war room.'")

new_alerts = df_alerts.filter(F.col("status").isin("new", "investigating")).count()
print(f"✅ alerts: {df_alerts.count():,} rows ({new_alerts} open for war room)")

# COMMAND ----------

# DBTITLE 1,Generate network relationships table
# =============================================================================
# TABLE 4: NETWORK RELATIONSHIPS (~15,000 edges)
# Creates clusters of connected entities (mule networks of 5-15 nodes)
# =============================================================================

relationship_types = ["shared_address", "shared_device", "shared_ip", 
                      "funds_transfer", "shared_beneficiary", "same_employer"]
rel_type_weights = [0.15, 0.15, 0.20, 0.30, 0.10, 0.10]

network_data = []
rel_counter = 0

# Create 50 mule network clusters (5-15 nodes each)
for cluster_idx in range(50):
    cluster_size = random.randint(5, 15)
    cluster_members = random.sample(customer_ids, cluster_size)
    
    # Create dense connections within cluster
    for j in range(len(cluster_members)):
        # Each node connects to 2-5 others in the cluster
        num_connections = random.randint(2, min(5, len(cluster_members) - 1))
        targets = random.sample([m for m in cluster_members if m != cluster_members[j]], num_connections)
        
        for target in targets:
            rel_counter += 1
            rel_type = random.choices(relationship_types, weights=rel_type_weights, k=1)[0]
            first_seen = fake.date_between(start_date='-2y', end_date='-90d')
            last_seen = fake.date_between(start_date=first_seen, end_date='today')
            
            network_data.append((
                f"REL-{rel_counter:07d}",
                cluster_members[j],
                target,
                rel_type,
                round(random.uniform(0.3, 1.0), 3),  # Higher strength within clusters
                str(first_seen),
                str(last_seen),
                random.randint(1, 200)
            ))

# Add random background noise edges to reach ~15K
while rel_counter < NUM_NETWORK_EDGES:
    rel_counter += 1
    src = random.choice(customer_ids)
    tgt = random.choice([c for c in customer_ids if c != src])
    rel_type = random.choices(relationship_types, weights=rel_type_weights, k=1)[0]
    first_seen = fake.date_between(start_date='-3y', end_date='-30d')
    last_seen = fake.date_between(start_date=first_seen, end_date='today')
    
    network_data.append((
        f"REL-{rel_counter:07d}",
        src, tgt, rel_type,
        round(random.uniform(0.05, 0.5), 3),  # Lower strength for noise
        str(first_seen),
        str(last_seen),
        random.randint(1, 20)
    ))

network_schema = StructType([
    StructField("relationship_id", StringType()),
    StructField("source_customer_id", StringType()),
    StructField("target_customer_id", StringType()),
    StructField("relationship_type", StringType()),
    StructField("strength_score", DoubleType()),
    StructField("first_seen", StringType()),
    StructField("last_seen", StringType()),
    StructField("transaction_count", IntegerType())
])

df_network = spark.createDataFrame(network_data, schema=network_schema)
df_network = df_network.withColumn("first_seen", F.to_date("first_seen")) \
    .withColumn("last_seen", F.to_date("last_seen"))

df_network.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.network_relationships")
spark.sql(f"COMMENT ON TABLE {CATALOG}.{SCHEMA}.network_relationships IS 'Entity relationship graph with mule network clusters. Edges represent shared attributes or fund flows between customers.'")
print(f"✅ network_relationships: {df_network.count():,} rows (50 mule network clusters)")

# COMMAND ----------

# DBTITLE 1,Generate sanctions watchlist table
# =============================================================================
# TABLE 5: SANCTIONS WATCHLIST (~500 entries)
# Simulated OFAC/EU/UN/UK sanctions lists
# =============================================================================

programs = ["OFAC_SDN", "EU_sanctions", "UN_sanctions", "UK_sanctions"]
program_weights = [0.40, 0.25, 0.20, 0.15]
entity_types = ["individual", "organization", "vessel"]
entity_type_weights = [0.50, 0.35, 0.15]

# Realistic sanctions entity name patterns
first_names_sanctions = ["Mohammed", "Ali", "Hassan", "Ahmad", "Ibrahim", "Youssef", "Dmitri", "Sergei", "Vladimir", "Andrei", "Kim", "Park", "Chen", "Wei", "Omar"]
last_names_sanctions = ["Al-Rashid", "Petrov", "Kuznetsov", "Nazari", "Mohammadi", "Khorasani", "Volkov", "Popov", "Jong-un", "Sung-ho", "Xiaoming", "Al-Hussein", "Bazzi", "Soleimani", "Karimi"]
org_names = [
    "Global Trade Holdings Ltd", "Eastern Star Shipping Co", "Petrochemical Industries Group",
    "Pan-Asia Import Export", "Crescent Financial Services", "Northern Wind Trading",
    "Pacific Rim Logistics", "Silk Road Capital Partners", "Golden Bridge Enterprises",
    "Continental Resource Management", "Oceanic Freight Solutions", "Atlas Mining Corporation",
    "Meridian Defense Systems", "Horizon Energy Partners", "Sovereign Wealth Advisory"
]
vessel_names = [
    "MV Storm Petrel", "MT Dark Waters", "MV Jade Dragon", "MT Shadow Runner",
    "MV Northern Star", "MT Pacific Ghost", "MV Desert Fox", "MT Crimson Tide",
    "MV Iron Eagle", "MT Silent Wave"
]

sanctions_data = []
for i in range(NUM_SANCTIONS):
    entity_id = f"SANC-{i+1:05d}"
    entity_type = random.choices(entity_types, weights=entity_type_weights, k=1)[0]
    
    if entity_type == "individual":
        entity_name = f"{random.choice(first_names_sanctions)} {random.choice(last_names_sanctions)}"
        aliases = [f"{random.choice(first_names_sanctions)} {random.choice(last_names_sanctions)}" for _ in range(random.randint(1, 4))]
    elif entity_type == "organization":
        entity_name = random.choice(org_names) + f" {random.randint(1,99)}"
        aliases = [fake.company() for _ in range(random.randint(1, 3))]
    else:  # vessel
        entity_name = random.choice(vessel_names) + f" {random.choice(['I','II','III',''])}".strip()
        aliases = [f"IMO-{random.randint(1000000, 9999999)}"]
    
    country = random.choice(HIGH_RISK_COUNTRIES)
    program = random.choices(programs, weights=program_weights, k=1)[0]
    date_listed = fake.date_between(start_date='-15y', end_date='-30d')
    
    sanctions_data.append((
        entity_id, entity_name, entity_type, country,
        aliases,  # array column
        str(date_listed), program,
        round(random.uniform(0.6, 1.0), 3),
        f"https://sanctionssearch.ofac.treas.gov/Details.aspx?id={random.randint(10000, 99999)}",
        fake.sentence(nb_words=15)
    ))

sanctions_schema = StructType([
    StructField("entity_id", StringType()),
    StructField("entity_name", StringType()),
    StructField("entity_type", StringType()),
    StructField("country", StringType()),
    StructField("aliases", ArrayType(StringType())),
    StructField("date_listed", StringType()),
    StructField("program", StringType()),
    StructField("match_score", DoubleType()),
    StructField("source_url", StringType()),
    StructField("description", StringType())
])

df_sanctions = spark.createDataFrame(sanctions_data, schema=sanctions_schema)
df_sanctions = df_sanctions.withColumn("date_listed", F.to_date("date_listed"))

df_sanctions.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.sanctions_watchlist")
spark.sql(f"COMMENT ON TABLE {CATALOG}.{SCHEMA}.sanctions_watchlist IS 'Simulated global sanctions watchlist (OFAC SDN, EU, UN, UK). Used by the sanctions agent for entity screening.'")
print(f"✅ sanctions_watchlist: {df_sanctions.count():,} rows")

# COMMAND ----------

# DBTITLE 1,Generate cases table
# =============================================================================
# TABLE 6: CASES (~800 rows)
# 20+ cases in open/investigating status for live demo
# =============================================================================

case_types = ["fraud", "aml", "sanctions", "insider_threat"]
case_type_weights = [0.35, 0.35, 0.20, 0.10]
case_statuses = ["open", "investigating", "pending_review", "escalated", "closed_no_action", "closed_sar_filed"]
case_status_weights = [0.05, 0.05, 0.08, 0.07, 0.45, 0.30]
priorities = ["P1", "P2", "P3", "P4"]
priority_weights = [0.05, 0.15, 0.40, 0.40]
teams = ["BSA/AML Team", "Fraud Investigations", "Sanctions Compliance", "Special Investigations Unit", "Cyber Fraud Unit"]
analysts = ["Sarah Chen", "Michael Torres", "Priya Patel", "James O'Brien", "Fatima Al-Rashid", "David Kim", "Elena Volkov", "Marcus Johnson"]

verdicts_options = ["legitimate", "suspicious", "fraudulent", "sanctions_violation", "insufficient_evidence", None]
narrative_templates = [
    "Customer exhibited structuring behavior with {n} transactions averaging ${amt:,.0f} over {days} days. Pattern consistent with CTR avoidance.",
    "Network analysis revealed customer is connected to {n} entities in a suspected mule network. Total funds flow: ${amt:,.0f}.",
    "Sanctions screening flagged potential match with {entity} ({program}). Confidence: {conf:.0%}. Manual review required.",
    "Velocity anomaly: {n} transactions totaling ${amt:,.0f} within {hours} hours, exceeding normal pattern by {x}x.",
    "Geographic anomaly detected. Customer transacted with counterparties in {country} despite no known business relationship.",
    "Behavioral analysis shows deviation from established profile. Risk score increased from {old} to {new}.",
    "Combined typology: structuring + geographic risk. Multiple wires to {country} in amounts just below reporting threshold.",
    "Insider threat indicators: employee accessed {n} accounts outside normal duties. Potential collusion with external actor."
]

cases_data = []
alert_ids_list = [f"ALT-{i+1:06d}" for i in range(NUM_ALERTS)]

for i in range(NUM_CASES):
    case_id = f"CASE-{i+1:06d}"
    
    # First 25 cases are hot demo cases (open/investigating)
    if i < 25:
        status = random.choice(["open", "investigating"])
        priority = random.choice(["P1", "P2"])
        case_type = random.choice(["fraud", "aml", "sanctions"])
        cust_id = random.choice(structuring_customers + [velocity_customer, sanctions_customer])
        created_at = NOW - timedelta(hours=random.randint(1, 48))  # Very recent
    else:
        status = random.choices(case_statuses, weights=case_status_weights, k=1)[0]
        priority = random.choices(priorities, weights=priority_weights, k=1)[0]
        case_type = random.choices(case_types, weights=case_type_weights, k=1)[0]
        cust_id = random.choice(customer_ids)
        created_at = NINETY_DAYS_AGO + timedelta(
            seconds=random.randint(0, int((NOW - NINETY_DAYS_AGO).total_seconds()))
        )
    
    updated_at = created_at + timedelta(hours=random.randint(1, 168))
    if updated_at > NOW:
        updated_at = NOW
    
    alert_id = random.choice(alert_ids_list)
    exposure = round(random.uniform(5000, 5000000), 2) if priority in ["P1", "P2"] else round(random.uniform(1000, 500000), 2)
    
    # Verdict only for closed cases
    if status.startswith("closed"):
        verdict = random.choice(["legitimate", "suspicious", "fraudulent", "sanctions_violation", "insufficient_evidence"])
        verdict_confidence = round(random.uniform(0.7, 0.99), 3)
        narrative = random.choice(narrative_templates).format(
            n=random.randint(3, 25), amt=exposure, days=random.randint(5, 60),
            hours=random.randint(2, 48), x=random.randint(3, 15),
            entity=f"{random.choice(first_names_sanctions)} {random.choice(last_names_sanctions)}",
            program=random.choice(programs), conf=random.uniform(0.7, 0.95),
            country=random.choice(HIGH_RISK_COUNTRIES),
            old=random.randint(20, 40), new=random.randint(60, 95)
        )
    else:
        verdict = None
        verdict_confidence = None
        narrative = None
    
    cases_data.append((
        case_id, alert_id, cust_id, case_type, status, priority,
        created_at.strftime("%Y-%m-%d %H:%M:%S"),
        updated_at.strftime("%Y-%m-%d %H:%M:%S"),
        random.choice(teams), random.choice(analysts),
        exposure, verdict, verdict_confidence, narrative
    ))

cases_schema = StructType([
    StructField("case_id", StringType()),
    StructField("alert_id", StringType()),
    StructField("customer_id", StringType()),
    StructField("case_type", StringType()),
    StructField("status", StringType()),
    StructField("priority", StringType()),
    StructField("created_at", StringType()),
    StructField("updated_at", StringType()),
    StructField("assigned_team", StringType()),
    StructField("assigned_analyst", StringType()),
    StructField("total_exposure_amount", DoubleType()),
    StructField("verdict", StringType()),
    StructField("verdict_confidence", DoubleType()),
    StructField("narrative_summary", StringType())
])

df_cases = spark.createDataFrame(cases_data, schema=cases_schema)
df_cases = df_cases.withColumn("created_at", F.to_timestamp("created_at")) \
    .withColumn("updated_at", F.to_timestamp("updated_at"))

df_cases.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.cases")
spark.sql(f"COMMENT ON TABLE {CATALOG}.{SCHEMA}.cases IS 'Case management table (Lakebase state). Open/investigating cases drive the live war room demo.'")

open_cases = df_cases.filter(F.col("status").isin("open", "investigating")).count()
print(f"✅ cases: {df_cases.count():,} rows ({open_cases} open for live demo)")

# COMMAND ----------

# DBTITLE 1,Generate case_actions audit trail
# =============================================================================
# TABLE 7: CASE ACTIONS - Agent Audit Trail (~5,000 rows)
# MLflow tracing: each agent's investigation steps with timing & tokens
# =============================================================================

agent_names = [
    "transaction_analyst", "customer_history_agent", "network_graph_agent",
    "sanctions_agent", "narrative_writer", "decision_agent"
]
action_types = ["data_retrieval", "analysis", "enrichment", "decision", "escalation"]
action_type_weights = [0.30, 0.30, 0.20, 0.15, 0.05]

models_used = [
    "gpt-4o", "claude-sonnet-4-20250514", "databricks-meta-llama-3.1-70b",
    "databricks-dbrx", "text-embedding-3-large", "custom-fraud-classifier-v3",
    None  # Some actions don't use a model
]

# Templates for input/output summaries per agent type
agent_io_templates = {
    "transaction_analyst": {
        "inputs": [
            "Analyzing {n} transactions for customer {cust} in date range {start} to {end}",
            "Running velocity analysis on wire transfers > $5K for {cust}",
            "Checking structuring patterns: transactions between $8K-$10K for {cust}"
        ],
        "outputs": [
            "Found {n} suspicious transactions totaling ${amt:,.0f}. Structuring confidence: {conf:.0%}",
            "Velocity spike detected: {n}x normal rate. Peak activity: {peak}",
            "No anomalies detected. Transaction pattern within normal bounds."
        ]
    },
    "customer_history_agent": {
        "inputs": [
            "Retrieving full customer profile and history for {cust}",
            "Checking KYC status and PEP screening for {cust}",
            "Analyzing account behavior changes over last 90 days for {cust}"
        ],
        "outputs": [
            "Customer risk profile: {risk}/100. PEP: {pep}. KYC: {kyc}. Account age: {age} years.",
            "Behavioral shift detected: {n}% increase in international wires since {date}",
            "Customer profile stable. No significant changes in activity pattern."
        ]
    },
    "network_graph_agent": {
        "inputs": [
            "Expanding network graph 2 hops from {cust}",
            "Identifying clusters and shared attributes for {cust}",
            "Cross-referencing network with known fraud rings"
        ],
        "outputs": [
            "Network expansion: {n} connected entities, {clusters} clusters. Max cluster density: {density:.2f}",
            "ALERT: Customer connected to known mule network (cluster #{cluster_id}). Shared: {shared}",
            "No high-risk network connections identified. Isolated node."
        ]
    },
    "sanctions_agent": {
        "inputs": [
            "Screening {cust} and counterparties against OFAC/EU/UN/UK lists",
            "Running fuzzy name matching against sanctions database",
            "Checking counterparty countries against restricted jurisdictions"
        ],
        "outputs": [
            "MATCH FOUND: {match_name} ({program}) - similarity score: {score:.0%}. Manual review required.",
            "No direct sanctions hits. {n} counterparties in restricted jurisdictions flagged for review.",
            "Clear - no sanctions matches above threshold (0.85)"
        ]
    },
    "narrative_writer": {
        "inputs": [
            "Synthesizing findings from all agents for case {case_id}",
            "Generating SAR narrative draft for {case_id}",
            "Compiling evidence summary with citations for {case_id}"
        ],
        "outputs": [
            "Narrative generated: {words} words. Key findings: {findings}. Recommendation: {rec}",
            "SAR draft complete. Filing category: {category}. Estimated suspicious activity: ${amt:,.0f}",
            "Evidence package assembled: {n} exhibits, {refs} cross-references."
        ]
    },
    "decision_agent": {
        "inputs": [
            "Evaluating consensus across 5 specialist agents for case {case_id}",
            "Applying decision framework: risk threshold={threshold}, confidence minimum={min_conf}",
            "Checking against auto-resolution criteria"
        ],
        "outputs": [
            "VERDICT: {verdict} (confidence: {conf:.0%}). Consensus: {consensus}. Action: {action}",
            "Auto-resolved: Legitimate transaction. All agent confidence > 95%. No escalation needed.",
            "ESCALATE: Insufficient consensus ({conf:.0%}). Routing to human analyst {analyst}."
        ]
    }
}

case_ids_list = [f"CASE-{i+1:06d}" for i in range(NUM_CASES)]

case_actions_data = []
for i in range(NUM_CASE_ACTIONS):
    action_id = f"ACT-{i+1:07d}"
    case_id = random.choice(case_ids_list[:100])  # Focus on first 100 cases for density
    agent = random.choice(agent_names)
    action_type = random.choices(action_types, weights=action_type_weights, k=1)[0]
    
    # Timestamp - actions happen in sequence within minutes
    base_ts = NINETY_DAYS_AGO + timedelta(
        seconds=random.randint(0, int((NOW - NINETY_DAYS_AGO).total_seconds()))
    )
    
    # Generate contextual I/O summaries
    templates = agent_io_templates[agent]
    cust_id = random.choice(customer_ids[:200])  # Focus on active customers
    
    input_summary = random.choice(templates["inputs"]).format(
        n=random.randint(5, 50), cust=cust_id, case_id=case_id,
        start=(NOW - timedelta(days=90)).strftime("%Y-%m-%d"),
        end=NOW.strftime("%Y-%m-%d"),
        threshold=random.choice([0.7, 0.8, 0.85, 0.9]),
        min_conf=random.choice([0.8, 0.85, 0.9, 0.95])
    )
    
    output_summary = random.choice(templates["outputs"]).format(
        n=random.randint(3, 30), amt=random.uniform(10000, 2000000),
        conf=random.uniform(0.6, 0.99), peak="14:00-16:00 UTC",
        risk=random.randint(40, 95), pep=random.choice(["Yes", "No"]),
        kyc=random.choice(["verified", "expired", "enhanced_due_diligence"]),
        age=random.randint(1, 10), date="2026-03",
        clusters=random.randint(1, 5), density=random.uniform(0.3, 0.9),
        cluster_id=random.randint(1, 50), shared="device_id, IP",
        match_name=f"{random.choice(first_names_sanctions)} {random.choice(last_names_sanctions)}",
        program=random.choice(programs), score=random.uniform(0.85, 0.98),
        words=random.randint(200, 1500), findings=random.randint(3, 8),
        rec=random.choice(["Escalate", "Auto-close", "File SAR", "Request more info"]),
        category=random.choice(["Structuring", "ML/TF", "Fraud", "Sanctions"]),
        refs=random.randint(5, 25), verdict=random.choice(["legitimate", "suspicious", "escalate", "block"]),
        consensus=random.choice(["4/5 agree", "5/5 agree", "3/5 agree", "unanimous"]),
        action=random.choice(["auto_close", "escalate_l2", "file_sar", "block_account"]),
        analyst=random.choice(analysts)
    )
    
    # Realistic timing and token usage
    if agent == "narrative_writer":
        duration_ms = random.randint(3000, 15000)
        tokens = random.randint(2000, 8000)
    elif agent == "decision_agent":
        duration_ms = random.randint(1000, 5000)
        tokens = random.randint(500, 3000)
    elif action_type == "data_retrieval":
        duration_ms = random.randint(200, 2000)
        tokens = random.randint(100, 500)
    else:
        duration_ms = random.randint(500, 8000)
        tokens = random.randint(300, 4000)
    
    trace_id = str(uuid.uuid4())
    model = random.choice(models_used)
    confidence = round(random.uniform(0.5, 0.99), 3)
    
    case_actions_data.append((
        action_id, case_id, agent, action_type,
        base_ts.strftime("%Y-%m-%d %H:%M:%S"),
        input_summary, output_summary, confidence,
        duration_ms, trace_id, model, tokens
    ))

actions_schema = StructType([
    StructField("action_id", StringType()),
    StructField("case_id", StringType()),
    StructField("agent_name", StringType()),
    StructField("action_type", StringType()),
    StructField("timestamp", StringType()),
    StructField("input_summary", StringType()),
    StructField("output_summary", StringType()),
    StructField("confidence_score", DoubleType()),
    StructField("duration_ms", IntegerType()),
    StructField("trace_id", StringType()),
    StructField("model_used", StringType()),
    StructField("tokens_consumed", IntegerType())
])

df_actions = spark.createDataFrame(case_actions_data, schema=actions_schema)
df_actions = df_actions.withColumn("timestamp", F.to_timestamp("timestamp"))

df_actions.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.case_actions")
spark.sql(f"COMMENT ON TABLE {CATALOG}.{SCHEMA}.case_actions IS 'Agent audit trail (MLflow tracing). Records every action each specialist agent takes during investigation, with timing, model usage, and full I/O.'")
print(f"✅ case_actions: {df_actions.count():,} rows")

# COMMAND ----------

# DBTITLE 1,Generate similar past cases for Vector Search
# =============================================================================
# TABLE 8: SIMILAR PAST CASES (~500 rows, for Vector Search)
# Historical cases with narrative text for embedding-based retrieval
# =============================================================================

typologies = ["structuring", "layering", "mule_network", "sanctions_evasion", "identity_fraud", "insider_trading"]
amount_ranges = ["$1K-$10K", "$10K-$50K", "$50K-$100K", "$100K-$500K", "$500K-$1M", "$1M+"]
jurisdictions = ["US-Northeast", "US-Southeast", "US-West", "US-Midwest", "UK", "EU", "APAC", "Middle East", "LatAm"]
outcomes = ["sar_filed", "account_closed", "law_enforcement_referral", "no_action", "consent_order", "fine_imposed"]

# Rich narrative templates for vector search
narrative_templates_past = [
    "Investigation of {typology} scheme involving {n} accounts over {months} months. "
    "Customer {name}, a {occupation} from {city}, conducted {txn_count} transactions "
    "totaling ${total:,.0f}. Pattern: {pattern}. Network analysis revealed "
    "{network_desc}. Resolution: {outcome}.",
    
    "SAR filed for suspected {typology}. Subject {name} ({nationality}) used "
    "{n} accounts to move ${total:,.0f} through {method}. Transactions characterized by "
    "{char}. Counterparties located in {countries}. Investigation duration: {days} days. "
    "Verdict: {verdict}.",
    
    "Alert-triggered investigation: {alert_type} on account of {name}. "
    "Initial risk score: {risk}. Deep dive revealed {finding}. "
    "Connected to {n} other subjects via {connection_type}. "
    "Total exposure: ${total:,.0f}. Case disposition: {outcome}.",
    
    "Proactive detection of {typology} ring. {n} subjects identified through "
    "network graph analysis. Primary actor: {name} ({occupation}). "
    "Funds flowed through {hops} layers before reaching {destination}. "
    "Timeframe: {months} months. Regulatory action: {action}.",
    
    "Cross-border {typology} case. Subject {name} maintained accounts in {n} jurisdictions. "
    "Triggered by {trigger}. Investigation uncovered {finding}. "
    "Total suspicious activity: ${total:,.0f} across {txn_count} transactions. "
    "Involved counterparties in {countries}. Final determination: {verdict}."
]

past_cases_data = []
for i in range(NUM_PAST_CASES):
    case_id = f"HIST-{i+1:05d}"
    typology = random.choice(typologies)
    
    # Generate rich narrative text for embedding
    template = random.choice(narrative_templates_past)
    narrative = template.format(
        typology=typology,
        n=random.randint(2, 20),
        months=random.randint(1, 24),
        name=fake.name(),
        occupation=random.choice(occupations),
        city=fake.city(),
        txn_count=random.randint(10, 500),
        total=random.uniform(10000, 5000000),
        pattern=random.choice(["rapid in/out within 24hrs", "amounts just below CTR threshold",
                               "round-dollar wires to shell companies", "multiple cash deposits at different branches",
                               "crypto-to-fiat conversion chain"]),
        network_desc=random.choice(["3 shell companies and 2 offshore trusts",
                                     "mule network of 8 individuals",
                                     "single beneficial owner behind 5 entities",
                                     "family network spanning 3 countries"]),
        outcome=random.choice(outcomes),
        nationality=random.choice(["US", "UK", "Russian", "Iranian", "Chinese", "Nigerian", "Brazilian"]),
        method=random.choice(["wire transfers", "crypto exchanges", "trade-based laundering",
                              "real estate purchases", "cash-intensive businesses"]),
        char=random.choice(["velocity spikes", "round amounts", "layered transfers",
                            "geographic dispersion", "time-of-day anomalies"]),
        countries=random.choice(["Iran, UAE, Turkey", "Russia, Cyprus, Malta",
                                  "China, Hong Kong, Singapore", "Nigeria, UK, US",
                                  "Venezuela, Panama, Colombia"]),
        days=random.randint(5, 180),
        verdict=random.choice(["confirmed fraud", "suspicious but inconclusive", "legitimate activity"]),
        alert_type=random.choice(["velocity", "structuring", "geographic anomaly", "amount spike"]),
        risk=random.randint(50, 95),
        finding=random.choice(["undisclosed business relationships", "fictitious counterparties",
                               "forged documentation", "layered ownership structure",
                               "connection to sanctioned entity"]),
        connection_type=random.choice(["shared addresses", "fund transfers", "common beneficial owner",
                                        "shared devices/IPs", "familial relationships"]),
        hops=random.randint(2, 6),
        destination=random.choice(["offshore shell company", "crypto wallet", "real estate",
                                    "luxury goods dealers", "foreign bank account"]),
        action=random.choice(["consent order", "$2M fine", "account closure", "law enforcement referral"]),
        trigger=random.choice(["automated alert", "tip from law enforcement", "regulatory exam",
                               "correspondent bank notification", "internal audit"])
    )
    
    past_cases_data.append((
        case_id,
        narrative,
        random.choice(outcomes),
        typology,
        random.choice(amount_ranges),
        random.choice(jurisdictions),
        random.randint(4, 720),  # resolution hours
        random.random() < 0.6  # 60% had SAR filed
    ))

past_cases_schema = StructType([
    StructField("case_id", StringType()),
    StructField("embedding_text", StringType()),
    StructField("outcome", StringType()),
    StructField("typology", StringType()),
    StructField("amount_range", StringType()),
    StructField("jurisdiction", StringType()),
    StructField("resolution_time_hours", IntegerType()),
    StructField("sar_filed", BooleanType())
])

df_past_cases = spark.createDataFrame(past_cases_data, schema=past_cases_schema)

df_past_cases.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.similar_past_cases")
spark.sql(f"COMMENT ON TABLE {CATALOG}.{SCHEMA}.similar_past_cases IS 'Historical case narratives for Vector Search similarity matching. Used to find precedent cases with similar typologies.'")
print(f"✅ similar_past_cases: {df_past_cases.count():,} rows")

# COMMAND ----------

# DBTITLE 1,Generate adverse media table
# =============================================================================
# TABLE 9: ADVERSE MEDIA (~1,000 rows)
# News/sanctions media hits linked to customers
# =============================================================================

media_sources = ["reuters", "bloomberg", "dow_jones", "local_media"]
media_source_weights = [0.25, 0.25, 0.30, 0.20]
risk_categories_pool = ["fraud", "corruption", "sanctions", "terrorism_financing", "tax_evasion"]

headline_templates = [
    "{name} Linked to International Money Laundering Network",
    "Authorities Investigate {name} for Suspected {crime}",
    "{company} Under Scrutiny for Sanctions Violations in {country}",
    "Former {role} {name} Charged with {crime}",
    "{country} Nationals Arrested in ${amount}M Fraud Scheme",
    "FinCEN Advisory: {typology} Patterns Emerging in {sector}",
    "{name} Added to {list} Sanctions List",
    "DOJ Indicts {n} Individuals in {country} Bribery Case",
    "SEC Charges {company} with Insider Trading",
    "FATF Adds {country} to Grey List Over AML Deficiencies",
    "{bank} Fined ${amount}M for BSA/AML Compliance Failures",
    "Panama Papers Reveal {name}'s Hidden Offshore Empire",
    "Crypto Exchange {company} Accused of Facilitating {crime}",
    "{name} Designated as Specially Designated National by OFAC",
    "UN Report Links {company} to Arms Trafficking in {country}"
]

summary_templates = [
    "Investigation reveals {name} allegedly facilitated ${amount:,.0f} in illicit transactions through a network of shell companies in {countries}. Authorities seized assets and froze accounts. Case pending trial.",
    "{company} faces regulatory action after compliance audit uncovered systematic failures in sanctions screening processes. {n} transactions flagged retroactively. Fine of ${fine:,.0f} expected.",
    "Intelligence reports indicate {name} maintains connections to entities previously designated under {program}. Enhanced due diligence recommended for all associated accounts.",
    "Multi-jurisdictional investigation spanning {countries} has identified {name} as key facilitator in {typology} network. Estimated proceeds: ${amount:,.0f}. Interpol red notice issued.",
    "Regulatory filing reveals {company} failed to file {n} SARs over {years} year period. Consent order requires enhanced monitoring program and independent audit."
]

adverse_media_data = []
for i in range(NUM_ADVERSE_MEDIA):
    article_id = f"ART-{i+1:06d}"
    entity_name = fake.name() if random.random() < 0.6 else fake.company()
    source = random.choices(media_sources, weights=media_source_weights, k=1)[0]
    publish_date = fake.date_between(start_date='-2y', end_date='today')
    
    # Generate headline
    headline = random.choice(headline_templates).format(
        name=entity_name, company=fake.company(),
        crime=random.choice(["Money Laundering", "Fraud", "Bribery", "Sanctions Evasion", "Tax Fraud"]),
        country=random.choice(HIGH_RISK_COUNTRIES + NORMAL_COUNTRIES),
        role=random.choice(["CEO", "CFO", "Government Official", "Bank Executive", "Fund Manager"]),
        amount=random.randint(1, 500),
        typology=random.choice(["Trade-Based Laundering", "Crypto Mixing", "Structuring", "Shell Company"]),
        sector=random.choice(["Banking", "Real Estate", "Crypto", "Trade Finance"]),
        list=random.choice(["OFAC SDN", "EU", "UN", "UK"]),
        n=random.randint(3, 25),
        bank=random.choice(["Global Bank Corp", "First National Trust", "Pacific Finance Group", "Continental Savings"])
    )
    
    # Generate summary
    summary = random.choice(summary_templates).format(
        name=entity_name, company=fake.company(),
        amount=random.uniform(100000, 50000000),
        countries=random.choice(["UAE, Cyprus, Panama", "Russia, UK, Malta", "China, Hong Kong, Macau"]),
        n=random.randint(5, 100), fine=random.uniform(1000000, 100000000),
        program=random.choice(programs), typology=random.choice(typologies),
        years=random.randint(2, 8)
    )
    
    # Risk categories (1-3 per article)
    num_categories = random.randint(1, 3)
    risk_cats = random.sample(risk_categories_pool, num_categories)
    
    # Link to some customers (sparse - most articles don't match)
    matched_customers = []
    if random.random() < 0.15:  # 15% of articles match a customer
        matched_customers = random.sample(customer_ids[:500], random.randint(1, 3))
    
    sentiment_score = round(random.uniform(-1.0, -0.1), 3)  # All adverse = negative
    credibility_score = round(random.uniform(0.5, 1.0), 3)
    
    adverse_media_data.append((
        article_id, entity_name, source, str(publish_date),
        headline, summary, sentiment_score, risk_cats,
        matched_customers, credibility_score
    ))

media_schema = StructType([
    StructField("article_id", StringType()),
    StructField("entity_name", StringType()),
    StructField("source", StringType()),
    StructField("publish_date", StringType()),
    StructField("headline", StringType()),
    StructField("summary", StringType()),
    StructField("sentiment_score", DoubleType()),
    StructField("risk_categories", ArrayType(StringType())),
    StructField("matched_customer_ids", ArrayType(StringType())),
    StructField("credibility_score", DoubleType())
])

df_media = spark.createDataFrame(adverse_media_data, schema=media_schema)
df_media = df_media.withColumn("publish_date", F.to_date("publish_date"))

df_media.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.adverse_media")
spark.sql(f"COMMENT ON TABLE {CATALOG}.{SCHEMA}.adverse_media IS 'Adverse media/news articles linked to customers. Used by the news/sanctions agent for risk enrichment.'")
print(f"✅ adverse_media: {df_media.count():,} rows")

# COMMAND ----------

# DBTITLE 1,Generate agent verdicts table
# =============================================================================
# TABLE 10: AGENT VERDICTS (~200 rows)
# Decision convergence data for the split-screen UI
# Shows all 6 agents thinking in parallel and converging on a verdict
# =============================================================================

verdict_options = ["legitimate", "suspicious", "escalate", "block"]
verdict_weights_by_agent = {
    "transaction_analyst": [0.30, 0.35, 0.25, 0.10],
    "customer_history_agent": [0.40, 0.30, 0.20, 0.10],
    "network_graph_agent": [0.25, 0.35, 0.25, 0.15],
    "sanctions_agent": [0.35, 0.25, 0.20, 0.20],
    "narrative_writer": [0.30, 0.35, 0.25, 0.10],  # Narrative writer also votes
    "decision_agent": [0.25, 0.30, 0.30, 0.15]
}

reasoning_templates = {
    "transaction_analyst": [
        "Transaction pattern shows {n} structured deposits averaging ${amt:,.0f}. Confidence in structuring: {conf:.0%}.",
        "Velocity within normal bounds. No anomalous patterns detected in {n}-day window.",
        "Wire transfers to {country} exceed baseline by {x}x. Recommend enhanced monitoring."
    ],
    "customer_history_agent": [
        "Customer has {years}-year relationship, consistent income pattern. Recent activity aligns with stated occupation ({occ}).",
        "KYC expired {months} months ago. PEP flag active. Risk score trending upward: {old}→{new}.",
        "New-to-bank customer (<6 months) with rapid activity escalation. Source of funds unclear."
    ],
    "network_graph_agent": [
        "Connected to {n} entities in cluster #{cluster}. {shared} shared attributes. Mule network probability: {conf:.0%}.",
        "Isolated node with no high-risk connections. Low network risk.",
        "2nd-degree connection to sanctioned entity via {path}. Indirect exposure: ${amt:,.0f}."
    ],
    "sanctions_agent": [
        "Direct name match: {name} ({program}). Score: {score:.2f}. Recommending immediate block.",
        "No sanctions hits above 0.85 threshold. Closest match: {name} at {score:.2f} (below threshold).",
        "Counterparty {cp_name} in {country} flagged. Country-level risk but no entity match."
    ],
    "narrative_writer": [
        "Compiled evidence from {n} sources. Narrative strength: strong. Key risk indicators: {indicators}.",
        "Evidence insufficient for SAR. Recommend monitoring and re-evaluation in 30 days.",
        "Clear pattern of {typology}. Drafted SAR narrative ({words} words) with {refs} supporting exhibits."
    ],
    "decision_agent": [
        "Consensus analysis: {agree}/6 agents recommend {verdict}. Overall confidence: {conf:.0%}. Final: {action}.",
        "Split verdict ({split}). Escalating to human analyst per policy threshold.",
        "Unanimous agreement: {verdict}. Auto-{action} applied. Audit trail preserved."
    ]
}

# Use first 35 cases for verdict convergence (includes all hot demo cases)
verdict_case_ids = [f"CASE-{i+1:06d}" for i in range(35)]

verdicts_data = []
verdict_counter = 0

for case_id in verdict_case_ids:
    # Each case gets verdicts from all 6 agents
    case_verdicts = []
    base_ts = NOW - timedelta(minutes=random.randint(1, 60))  # Recent for demo
    
    for agent_idx, agent in enumerate(agent_names):
        verdict_counter += 1
        verdict_id = f"VRD-{verdict_counter:06d}"
        
        # Agent's individual verdict
        weights = verdict_weights_by_agent[agent]
        verdict = random.choices(verdict_options, weights=weights, k=1)[0]
        confidence = round(random.uniform(0.65, 0.99), 3)
        case_verdicts.append(verdict)
        
        # Reasoning
        reasoning = random.choice(reasoning_templates[agent]).format(
            n=random.randint(3, 20), amt=random.uniform(5000, 500000),
            conf=confidence, country=random.choice(HIGH_RISK_COUNTRIES),
            x=random.randint(2, 10), years=random.randint(1, 15),
            occ=random.choice(occupations), months=random.randint(1, 12),
            old=random.randint(20, 40), new=random.randint(60, 95),
            cluster=random.randint(1, 50), shared=random.randint(2, 5),
            path="shared_device → funds_transfer",
            name=f"{random.choice(first_names_sanctions)} {random.choice(last_names_sanctions)}",
            program=random.choice(programs), score=random.uniform(0.85, 0.98),
            cp_name=fake.company(),
            indicators="velocity+geography+structuring",
            typology=random.choice(typologies), words=random.randint(500, 2000),
            refs=random.randint(5, 15), agree=random.randint(4, 6),
            verdict=verdict, action=random.choice(["close", "escalate", "block", "monitor"]),
            split="3 escalate / 2 suspicious / 1 legitimate"
        )
        
        # Evidence references
        evidence_refs = [f"TXN-{random.randint(1, NUM_TRANSACTIONS):08d}" for _ in range(random.randint(2, 8))]
        
        # Timestamp: agents complete within seconds of each other (parallel execution)
        agent_ts = base_ts + timedelta(seconds=random.randint(agent_idx * 2, agent_idx * 5 + 10))
        
        verdicts_data.append((
            verdict_id, case_id, agent, verdict, confidence,
            reasoning, evidence_refs, agent_ts.strftime("%Y-%m-%d %H:%M:%S"),
            False, None  # consensus fields filled below
        ))
    
    # Determine consensus for this case
    from collections import Counter
    verdict_counts = Counter(case_verdicts)
    most_common = verdict_counts.most_common(1)[0]
    consensus_reached = most_common[1] >= 4  # 4/6 agree = consensus
    final_decision = most_common[0] if consensus_reached else "escalate"
    
    # Update the last 6 entries with consensus info
    for j in range(6):
        idx = len(verdicts_data) - 6 + j
        row = list(verdicts_data[idx])
        row[8] = consensus_reached
        row[9] = final_decision
        verdicts_data[idx] = tuple(row)

verdicts_schema = StructType([
    StructField("verdict_id", StringType()),
    StructField("case_id", StringType()),
    StructField("agent_name", StringType()),
    StructField("verdict", StringType()),
    StructField("confidence", DoubleType()),
    StructField("reasoning_summary", StringType()),
    StructField("evidence_refs", ArrayType(StringType())),
    StructField("timestamp", StringType()),
    StructField("consensus_reached", BooleanType()),
    StructField("final_decision", StringType())
])

df_verdicts = spark.createDataFrame(verdicts_data, schema=verdicts_schema)
df_verdicts = df_verdicts.withColumn("timestamp", F.to_timestamp("timestamp"))

df_verdicts.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.agent_verdicts")
spark.sql(f"COMMENT ON TABLE {CATALOG}.{SCHEMA}.agent_verdicts IS 'Agent verdict convergence data for the split-screen war room UI. Shows all 6 agents reasoning in parallel and reaching consensus.'")

consensus_count = df_verdicts.filter(F.col("consensus_reached") == True).select("case_id").distinct().count()
print(f"✅ agent_verdicts: {df_verdicts.count():,} rows ({consensus_count} cases reached consensus)")

# COMMAND ----------

# DBTITLE 1,Print final summary
# =============================================================================
# FINAL SUMMARY: All tables created
# =============================================================================

print("="*70)
print("  MULTI-AGENT FINANCIAL CRIME WAR ROOM - DATA GENERATION COMPLETE")
print("="*70)
print(f"\n  Catalog: {CATALOG}")
print(f"  Schema:  {SCHEMA}")
print(f"\n{'  Table':<40} {'Rows':>10}")
print(f"  {'-'*38} {'-'*10}")

tables = [
    "customers", "transactions", "alerts", "network_relationships",
    "sanctions_watchlist", "cases", "case_actions", "similar_past_cases",
    "adverse_media", "agent_verdicts"
]

total_rows = 0
for table in tables:
    count = spark.table(f"{CATALOG}.{SCHEMA}.{table}").count()
    total_rows += count
    print(f"  {table:<38} {count:>10,}")

print(f"  {'-'*38} {'-'*10}")
print(f"  {'TOTAL':<38} {total_rows:>10,}")
print(f"\n  🚨 Hot Demo Scenarios:")
print(f"     • Structuring ring: customers {structuring_customers}")
print(f"     • Velocity anomaly: customer {velocity_customer}")
print(f"     • Sanctions hit:    customer {sanctions_customer}")
print(f"\n  🎯 Ready for the war room! Open cases trigger the agent swarm.")
print("="*70)