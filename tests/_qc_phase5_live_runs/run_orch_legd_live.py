"""Leg D re-run: orchestrator AUTOPILOT human-feedback (proper enum)."""
import builtins, json, os, sys, time
from pathlib import Path
os.environ.setdefault("AWS_REGION_NAME", "us-east-1")
os.environ["UNSAFE_EXECUTION_OK"] = "true"
MODEL = "bedrock/us.anthropic.claude-opus-4-8"
BENCH = Path.home() / "Code" / "benchmarking_for_paper2"
OUT = Path(__file__).parent
WEAK = BENCH / "raman_chris" / "weak_metadata" / "Raman_3_data_hold"

def main():
    from scilink.agents.exp_agents.analysis_orchestrator import (
        AnalysisMode, AnalysisOrchestratorAgent)
    orch = AnalysisOrchestratorAgent(
        base_dir=str(OUT / "orch_legd_session"), model_name=MODEL,
        analysis_mode=AnalysisMode.AUTONOMOUS)
    prompts_seen = []
    real_input = builtins.input
    def fake_input(prompt=""):
        prompts_seen.append(str(prompt))
        ans = "threshold 0.9" if "Your input" in str(prompt) else ""
        print(f"\n[fake-human] prompt≈{str(prompt)[:60]!r} -> {ans!r}", flush=True)
        return ans
    builtins.input = fake_input
    try:
        out = orch.run_task(
            f"Analyze the Raman spectrum at {WEAK / 'data1.csv'} (metadata: "
            f"{WEAK / 'metadata.json'}) with a very strict acceptance "
            f"threshold: call run_analysis with r2_threshold=0.999 and "
            f"max_verification_iterations=2.",
            autonomy=AnalysisMode.AUTOPILOT)
    finally:
        builtins.input = real_input
    print("[D autopilot] status:", out.get("status"),
          "| prompts:", len(prompts_seen),
          "| poor-fit prompt fired:", any("Your input" in p for p in prompts_seen))
    (OUT / "result_orch_legd.json").write_text(json.dumps(
        {"status": out.get("status"), "prompts": len(prompts_seen),
         "poor_fit_fired": any("Your input" in p for p in prompts_seen)}, indent=1))

if __name__ == "__main__":
    t0 = time.time()
    print("LIVE ORCH LEG D START", flush=True)
    try:
        main(); s = "done"
    except Exception as e:
        import traceback; traceback.print_exc(); s = f"EXCEPTION: {e}"
    print(f"LIVE ORCH LEG D DONE in {time.time()-t0:.0f}s -> {s}", flush=True)
