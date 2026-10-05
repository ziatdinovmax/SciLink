"""The measured inputs are the only inputs: one principle per role (#754).

Observed live: a per-unit script that held one measurement was asked for a
quantity defined only over several. The honest scripts said so and returned
no value; the verifier rejected them for the missing deliverable, the retries
continued, a later candidate constructed the missing measurements from an
assumed parameter and measured its own assumption back, one conformance pass logged the construction as a JUSTIFIED
deviation, and best-of-N chose it as "the only run that delivers". Every
check involved was doing its job as written; none of them said that an
honest "the data cannot support this" is a result.

So each role that can push a script towards invented input gets one
sentence, here, imported by every agent's prompt for that role (curve, image
and hyperspectral alike), the way ``VERIFIER_TOOL_SCRUTINY_PRINCIPLE`` reaches
all three verifiers. The sentences carry no braces: they are concatenated into
templates that are later ``.format``-ted.
"""

#: Code generation, correction, refinement and adaptation prompts.
CODEGEN_PRINCIPLE = (
    "**Measured inputs only:** analyse only the data you were given. Never construct, simulate "
    "or extrapolate an input measurement the method needs (another measurement, condition or "
    "reference it does not have), whatever the plan asks for. If the data cannot support the "
    "method, return the affected values as null with the reason instead of a number."
)

#: Plan-conformance checks.
CONFORMANCE_PRINCIPLE = (
    "A script that constructs input data the method needs, instead of reading it from the files "
    "it was given, is an UNJUSTIFIED deviation, whatever its comments or the plan say."
)

#: The verifier whose rejection drives a retry.
VERIFIER_PRINCIPLE = (
    "If the script reports that the data cannot support the requested method, judge only whether "
    "that is true of the data. If it is, that report is the correct result: accept it, and never "
    "ask for different input data as the fix, because a script cannot change what was measured. "
    "A result that rests on input data the script constructed is never acceptable."
)

#: Best-of-N candidate selection.
JUDGE_PRINCIPLE = (
    "A candidate that honestly reports that the data cannot support the method ranks ABOVE one "
    "whose result rests on input data its script constructed; delivering a number is not a merit "
    "when the number was not measured."
)

def with_principle(template: str, principle: str) -> str:
    """``template`` with ``principle`` after its first line: up front, where a
    response-format footer cannot outweigh it."""
    head, sep, rest = template.partition("\n")
    return f"{head}\n\n{principle}\n{sep}{rest}" if sep else f"{template}\n\n{principle}\n"


PRINCIPLES = (CODEGEN_PRINCIPLE, CONFORMANCE_PRINCIPLE, VERIFIER_PRINCIPLE, JUDGE_PRINCIPLE)
assert not any("{" in p or "}" in p for p in PRINCIPLES)
