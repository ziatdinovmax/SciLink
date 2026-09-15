# Planning agent — critical material recovery demo

An end-to-end planning example for the planning orchestrator. The
goal: recommend an experimental campaign for recovering critical
materials from produced water (Permian Basin), grounded in the
provided ICP-MS measurements and the DOE critical-material context.

Layout:

- `experimental_data/` — `prowater_icpms.{xlsx,json}` (ICP-MS readings
  and a short objective sidecar)
- `knowledge_folder/` — DOE 2023 critical-materials assessment (PDF),
  the criticality matrix figure, and a Public Water System database
  for cross-reference

## Run

**Streamlit UI** (recommended):

```bash
scilink ui
```

— in the planning panel, point the agent at the two folders above.

**CLI**:

```bash
scilink plan --autonomy autopilot \
             --data-dir examples/planning_agent_demo/experimental_data \
             --knowledge-dir examples/planning_agent_demo/knowledge_folder
```

## What to expect

The agent should ingest the ICP-MS readings, cross-reference the
critical-material list from the DOE document, and propose a prioritized
plan for follow-up experiments aimed at recovery of the highest-value
species in the produced water.
