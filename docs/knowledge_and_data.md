# Knowledge bases, RAG and data query

Two ways to ground the agents in what you already know:

- **RAG over your knowledge base.** Papers, project notes, instrument manuals
  and prior results are indexed and retrieved to ground hypothesis generation
  and experiment design.
- **Agentic knowledge query.** For *structured* data — tabular files and record
  databases — the agent writes and runs query code, with no upfront schema.
  `query_knowledge_data` is for ad-hoc exploration (*"what fields exist?"*,
  *"value range of X?"*); `screen_database` is for production filter-and-rank
  passes with a structured top-K output.

## Named knowledge bases

A knowledge base can be promoted to a **named, reusable artifact** stored under
`~/.scilink/knowledge_bases/` (follows `$SCILINK_HOME`). Build it once (this
embeds the documents, so it needs the embedding provider's API key); every
later session reuses the persisted index from any directory:

```bash
scilink kb create produced-water --from ./papers ./composition_data \
    --description "Produced-water composition and criticality references"
scilink kb list                          # what exists, built with which embedding model
scilink kb add produced-water --from ./new_paper.pdf   # embeds only the new documents
scilink kb import legacy --from ./kb_storage --embedding-model gemini-embedding-001
scilink kb rebuild produced-water --embedding-model text-embedding-3-small
```

Use a KB by name wherever a knowledge directory is accepted (an existing
directory of the same name wins):

```bash
scilink plan --knowledge-dir produced-water
scilink explore --knowledge-dir produced-water
```

**Grounding is always an explicit choice.** A mission-control session never
silently inherits a `kb_storage/` from the launch directory. Detached KBs — that
directory plus every named KB, with its description and sources — are offered
in chat: in autopilot the agent asks before the first planning delegation; in
autonomous mode it attaches the KB whose sources are clearly relevant to the
task, and leaves all detached otherwise. Standalone `scilink plan` keeps the
`--knowledge-dir` it was given.

Creating and growing a KB work from chat too — *"save the DOE report I uploaded
as a knowledge base called produced-water"* builds it in the store (with the
session's embedding model, or keyword-only without one); *"add this datasheet
to my produced-water knowledge base"* embeds just that document with the KB's
own embedding model, and the running session picks it up immediately. Both
happen only on your explicit request, since a named KB is shared across
sessions; otherwise sessions treat it as read-only. On a shared multi-user
server both are refused (the store is every user's); an operator uses
`scilink kb`.

Each KB's `manifest.json` records the embedding model that built it, so a
provider mismatch (a Gemini-built KB in an OpenAI-embedding session) warns
upfront with a rebuild hint. A KB is never unusable: retrieval degrades from
dense, to model-free keyword (BM25) search over the stored chunks, to no
retrieved context, each step logged, and never aborts generation. A KB made
with `kb create` keeps copies of its sources and can be re-embedded with
`kb rebuild`; an imported one cannot, so prefer `create` when you have the
original documents.

## Metadata from notes

```python
from scilink.agents.exp_agents import generate_metadata_json_from_text

# "HAADF-STEM of MoS2 monolayer, 50nm FOV, 300kV"
# → {"experiment_type": "Microscopy", "experiment": {"technique": "HAADF-STEM"}, ...}
metadata = generate_metadata_json_from_text("./experiment_notes.txt")
```

## Novelty assessment

Analysis findings can be checked against the scientific literature to find
what is genuinely new, through [FutureHouse](https://www.futurehouse.org/)
agents (`FUTUREHOUSE_API_KEY`):

```
👤 You: Assess novelty of these claims

🤖 Agent: ⚡ Searching literature via FutureHouse...

  📚 [Score 2/5] Mixed 2H/1T phase coexistence → Well-documented
  🤔 [Score 3/5] Sulfur vacancy density of 3.2 × 10¹³ cm⁻² → Similar measurements exist
  🌟 [Score 4/5] 1T phase localized within 5nm of grain boundaries → Limited prior reports

  Summary: 1 HIGH-NOVELTY finding identified
```

The discovery loop: **analysis** produces scientific claims → **novelty
assessment** scores each against the literature → **recommendations** put
validation experiments for the novel findings first.
