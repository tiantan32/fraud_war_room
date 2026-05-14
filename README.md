# Fraud War Room

Databricks notebooks for a fraud detection "war room" demo: synthetic data generation, infrastructure setup, tools, and two agent orchestration patterns (custom + Supervisor Agent).

## Contents

| Notebook | Purpose |
|---|---|
| `00_generate_synthetic_data.py` | Generate synthetic fraud data |
| `01_setup_infrastructure.py` | Provision Unity Catalog / Delta tables / serving endpoints |
| `02_tools.py` | Define agent tools (UC functions, vector search, etc.) |
| `03.1_custom_agent_orchestration.py` | Custom agent orchestration pattern |
| `03.2_supervisor_agent_orchestration.py` | Supervisor Agent orchestration pattern |

## Format

Notebooks are exported as Databricks SOURCE (`.py`) files. To use them:

1. Import into a Databricks workspace: **Workspace** → **Import** → paste the `.py` file.
2. Or clone this repo as a Databricks Git folder via **Repos** → **Add Repo**.

## Source workspace

Exported from `fe-vm-ttan-vm.cloud.databricks.com` → `/Users/tian.tan@databricks.com/fraud_war_room`.
