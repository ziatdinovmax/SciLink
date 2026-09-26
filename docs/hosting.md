# Hosting SciLink: the operator contract

This page is for whoever runs SciLink as a service: what the container
needs, what it exposes, and what it does not do. It is deliberately
cloud-neutral; the choices below were exercised on AWS (ECS Fargate, EFS,
ECR, CodeBuild) with two workspaces running analyses at the same time.

## The unit is a workspace

One SciLink process serves one **workspace**: a campaign's sessions,
uploads, knowledge bases, banked scripts, learned skills, usage ledger
and secrets. Isolation between workspaces is the container boundary; there
is no in-process multi-tenancy, and the `--users` mode is for colleagues
who trust each other inside one workspace. Everything a workspace learns
or writes stays on its volume.

## The image

`docker build --target web -t scilink-web .` builds the server image
(Python 3.12, CPU PyTorch; `--build-arg TORCH_INDEX=...` for a GPU host).
Its entrypoint creates the workspace layout on a fresh volume and starts
`scilink-web` on all interfaces; extra arguments are passed through. The
React bundle under `scilink/server/static/` must exist before the build
(`scripts/build_webui.sh`; release wheels ship it).

## Volumes

| mount | content | sharing |
|---|---|---|
| `/workspace` | `sessions/` (session root), `home/` (`SCILINK_HOME`: `knowledge_bases/`, `script_bank/`, learned skills, `instruments/`, `config.json`), `data/`, `usage.jsonl`, optional `workspace.json` | one per workspace, read-write |
| `/models` | downloaded model weights (`SCILINK_MODELS`) | shared, read-only is fine once populated |

A fresh volume may be empty; the entrypoint creates what is missing. The
write pattern is small files rewritten every turn plus per-frame writes in
a live loop; a network file system (EFS) handled two concurrent analyses
without errors, but sustained live loops on it are unmeasured.

## Environment

| variable | meaning |
|---|---|
| `SCILINK_WEB_TOKEN` | the workspace's access token (required when binding a non-loopback address, unless the header mode is used) |
| `SCILINK_AUTH_HEADER`, `SCILINK_TRUSTED_PROXIES` | identity from an authenticating proxy: the user is the header's value, honoured only from those addresses/CIDRs; per-user roots inside the workspace |
| `SCILINK_OPS_TOKEN` | sent as `X-Ops-Token` by the control plane for status, drain and usage |
| `SCILINK_TOKEN_BUDGET` | soft cap on tokens in the current period; new turns get a 429 once spent |
| `SCILINK_USAGE_FILE` | where the usage ledger lives (default `<session root>/usage.jsonl`) |
| `SCILINK_WORKSPACE` | path of the manifest (`{"id", "name", ...}`) named by `/api/v1/ops/health` |
| `SCILINK_HOME`, `SCILINK_MODELS` | the persistent store and the model cache |
| `SCILINK_FILE_ROOTS` | extra directories an agent may read and write; on a non-loopback bind every path is already fenced to the workspace |
| `SCILINK_MAX_WORKERS` | ceiling for every worker pool (`auto` = CPU count); size it to the task |
| `SCILINK_SANDBOX_MEM_MB`, `SCILINK_SANDBOX_FILE_MB`, `SCILINK_SANDBOX_PROCS` | rlimits for generated scripts, off by default |
| `SCILINK_SANDBOX_ENV` | extra environment variables generated scripts may see (they get an allowlist, never the model keys) |
| model credentials | `AWS_BEARER_TOKEN_BEDROCK` + `AWS_REGION_NAME`, or a task/pod role, or `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GEMINI_API_KEY`; a key can also be given per session at creation. The role mode needs Bedrock model access in the hosting account (for Anthropic models, the use-case form); without it, a key from an account that has access is passed per session and the container holds no model secret |
| `MP_API_KEY`, `FUTUREHOUSE_API_KEY` | optional: Materials Project, literature search |

## Network

One port, 8422, HTTP. Terminate TLS in front. Ingress: users through the
proxy, the control plane for the ops routes. Egress the server needs: the
model provider, the Materials Project and literature APIs if used, and a
package index only if generated scripts may install (they are not told to).
There is no east-west traffic between workspaces.

## The ops surface

| route | auth | purpose |
|---|---|---|
| `GET /api/v1/ops/health` | none | up, version, workspace id; for load-balancer and container health checks |
| `GET /api/v1/ops/status` | ops token or a signed-in user | `idle` / `busy` / `draining`, the busy reasons (turns, live runs, memory jobs), `idle_for_s` as seen by the poller |
| `POST /api/v1/ops/drain` `{"drain": true}` | ops token (a signed-in user on a single-user workspace) | refuse new turns, sessions and live runs with 503 so running work finishes; `false` reopens |
| `GET /api/v1/usage` | ops token or a signed-in user | calls and tokens in the current period, per model and per session, the budget |
| `POST /api/v1/usage/period` | ops token | start a new billing period after reading the old one |
| `GET /api/v1/workspace` | signed-in user | the manifest |

Stopping a workspace: drain, poll status until `idle`, then stop the task.
On SIGTERM the server marks any running turn as interrupted; the resumed
session shows one note saying so. A stop without a drain loses the running
turn's work since its last checkpoint, nothing else.

## Sizing and cost shape

One process per workspace; a turn runs on a thread, worker pools inside it
are bounded by `SCILINK_MAX_WORKERS`. Measured on a 2 vCPU / 8 GB task: one
full curve-fitting analysis of a demo spectrum took about 8 minutes and
15 model calls, and two workspaces ran such analyses concurrently without
interference. The image is about 7 GB, so a cold start is a few minutes of
image pull; keep active workspaces warm and scale dormant ones to zero.
Model calls dominate the cost; the usage ledger is the meter.

## What SciLink does not do

Provision workspaces, route users to them, decide when to stop a task,
back up volumes, or delete a workspace. Those belong to a control plane
built on the routes above. `scripts/check_deployment.py` exercises a
running workspace end to end and is the acceptance test for one.
