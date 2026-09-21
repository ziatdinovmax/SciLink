"""Live measurement loops: analyse each frame as it arrives, in bounded time.

Not a mode and not an agent — a driver that composes the analysis agents'
typed methods. See :mod:`scilink.live.measurement_loop`.
"""

from .instruments import EndOfData, Frame, Instrument, ReplayInstrument, run_experiment
from .mcp_instrument import MCPInstrument
from .measurement_loop import LOOP_LOG_NAME, LoopNotReady, MeasurementLoop
from .modality import CurveModality, HyperspectralModality, ImageModality
from .recommend import (GPRecommender, InstrumentSchema, LLMRecommender,
                        ParameterSpec, Recommender, RuleTableRecommender)

__all__ = ["MeasurementLoop", "LoopNotReady", "LOOP_LOG_NAME",
           "Instrument", "Frame", "run_experiment", "ReplayInstrument", "EndOfData", "MCPInstrument",
           "CurveModality", "HyperspectralModality", "ImageModality",
           "InstrumentSchema", "ParameterSpec", "Recommender",
           "RuleTableRecommender", "GPRecommender", "LLMRecommender"]
