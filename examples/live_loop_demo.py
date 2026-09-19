"""Live measurement loop: a simulated experiment first, your instrument next.

Run it as is to watch SciLink follow a simulated in-situ Raman anneal:

    python examples/live_loop_demo.py            # needs an LLM credential for setup

One reference spectrum is analysed thoroughly (a few minutes, a handful of model
calls). The verified script is locked as the recipe and the outputs you named
are pinned to those names. Every later frame is then answered by the recipe
alone — about a second, no model call — and flagged when the recipe stops
fitting, at which point a new one is built in the background.

To move to a real instrument, replace ``get_simulator(...)`` with your own
``Instrument`` subclass (the sketch at the bottom). Nothing else changes; the
same class also works in the web UI's Live tab as ``package.module:ClassName``.
"""

from scilink.live import MeasurementLoop, run_experiment
from scilink.live.simulators import get_simulator

MODEL = "claude-opus-4-6"           # or e.g. "bedrock/us.anthropic.claude-opus-4-8"


def main():
    instrument = get_simulator("insitu_raman")      # <- swap this line for MySpectrometer()

    loop = MeasurementLoop(
        "live_demo/loop", model_name=MODEL,
        system_info=instrument.system_info,         # technique, sample, axes
        targets=instrument.targets,                 # what matters, in plain words
        outputs=instrument.outputs,                 # name -> definition; reported under these names
        schema=instrument.schema,                   # what the controller accepts
        auto_escalate=True)                         # rebuild the recipe when it stops fitting

    reference = instrument.acquire({}).save("live_demo/reference", 0, stem="reference")
    loop.setup(reference=reference)                 # the only slow, model-driven step

    def show(frame, record):
        values = {k: round(record["features"][k], 3) for k in instrument.outputs
                  if k in record["features"]}
        print(f"frame {record['step']:3d}  {record['latency_s']:.1f}s  {values}  {record['flags']}")

    # apply="never": recommendations are recorded, parameters never change.
    run_experiment(instrument, loop, n_frames=40, apply="never", on_frame=show)
    print(loop.status())


# ── your instrument ──────────────────────────────────────────────────────────
#
# from scilink.live import Frame, Instrument, InstrumentSchema
#
# class MySpectrometer(Instrument):
#     name = "my_raman"
#     system_info = {"technique": "Raman spectroscopy", "sample": "...",
#                    "x_axis": "Raman shift (cm^-1)", "y_axis": "intensity (counts)"}
#     schema = InstrumentSchema.from_dict({
#         "integration_s": {"low": 0.1, "high": 60, "units": "s",
#                           "description": "detector integration time"}})
#     defaults = {"integration_s": 1.0}
#     outputs = {"g_position": "position of the G band maximum (cm^-1)"}
#     targets = ["G band position"]
#
#     def acquire(self, params):
#         p = self.check(params)                                  # defaults + schema guard
#         x, y = vendor_api.measure(integration=p["integration_s"])
#         return Frame(x=x, y=y, params=p, x_label="raman_shift", y_label="intensity")


if __name__ == "__main__":
    main()
