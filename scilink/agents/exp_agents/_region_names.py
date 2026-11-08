"""A region is named by evidence, not by its contrast.

Observed live: asked to compare regions of a spectrum image, the planner
told the code to threshold an intensity map and call the bright side the
material. In the data at hand the material was the dark side, so every region
average carried the wrong name, and the run's own reading of its maps then
contradicted its region table. Which side of a threshold a named thing falls
on depends on the technique, the detector and the energy window; it is a
fact about the data, not a default.

One sentence, given to the role that plans the analysis and to the role that
writes its code (every attempt and retry is built on the same base prompt).
It carries no braces: it is concatenated into templates that are later
``.format``-ted.
"""

REGION_NAMES_PRINCIPLE = (
    "**Regions are named by evidence:** a region labelled as what it is (a particle, a "
    "substrate, vacuum, a phase) must be identified from evidence in the data, and the script "
    "states that evidence, never from an assumed contrast such as the brighter side being the "
    "material; when the data cannot tell which is which, name the regions by their signal "
    "(high / low) instead."
)

assert "{" not in REGION_NAMES_PRINCIPLE and "}" not in REGION_NAMES_PRINCIPLE
