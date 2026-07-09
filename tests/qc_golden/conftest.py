import logging
import os

# Sandbox approval + memory staging gated off for deterministic offline runs.
os.environ["UNSAFE_EXECUTION_OK"] = "true"
os.environ["SCILINK_T2_AUTODISTILL"] = "0"
os.environ["SCILINK_FEEDBACK_AUTODISTILL"] = "0"

logging.basicConfig(level=logging.INFO)
