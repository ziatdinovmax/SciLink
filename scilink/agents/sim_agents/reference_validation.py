"""Engine-neutral reference-property validation (pre-run).

Before trusting a novel prediction, validate the model against independently-
known reference quantities of the system's constituents — the "validate before
you predict" discipline (see
``docs/proposals/reference-property-validation-design.md``).

This module is the engine- and force-field-neutral stage. It walks the DISTINCT
components of a system and asks a supplied ``measure_fn`` for each one's
reference property, then assembles a report the physics critic reasons over. It
names no force field and no engine: ``measure_fn`` does the heavy work through
the existing ``ParameterizedSystem`` contract (parameterize -> write_md_inputs
-> run -> read the property), so adding a new backend or engine needs no change
here.
"""

from typing import Any, Callable, Dict, List, Optional


def _component_key(component: Dict[str, Any]) -> str:
    """Identity of a component for de-duplication: SMILES if present, else name."""
    return str(component.get("smiles") or component.get("name") or "").strip()


def validate_component_properties(
    components: List[Dict[str, Any]],
    measure_fn: Callable[[Dict[str, Any]], Optional[Dict[str, Any]]],
    *,
    reference_property: str = "density",
) -> Dict[str, Any]:
    """Measure each distinct component's reference property for the critic.

    Args:
        components: ``[{"name", "smiles", "count"}, ...]`` — the system's
            component manifest. Repeated species (same SMILES/name) are
            measured once.
        measure_fn: Backend-agnostic measurement of ONE component, returning
            ``{"value": float, "units": str, ...}`` or ``None`` if it could not
            measure. May raise. A component that raises or returns no value is
            recorded as unmeasured, never fatal — partial evidence still helps
            the critic reason.
        reference_property: What ``measure_fn`` measures (default ``"density"``).

    Returns:
        ``{"status", "reference_property", "measurements": [...]}``. Each
        measurement is ``{"component", "smiles", "status", ...}`` with
        ``status`` either ``"measured"`` (carrying ``value``/``units`` plus any
        extra fields the measurer recorded) or ``"unmeasured"`` (carrying
        ``error``). Top-level ``status`` is ``"success"`` if any component was
        measured, else ``"no_measurements"``.
    """
    seen = set()
    measurements: List[Dict[str, Any]] = []
    for comp in components or []:
        key = _component_key(comp)
        if not key or key in seen:
            continue
        seen.add(key)
        entry: Dict[str, Any] = {
            "component": comp.get("name") or key,
            "smiles": comp.get("smiles"),
        }
        try:
            result = measure_fn(comp)
        except Exception as e:  # a measurement failure must not sink the stage
            entry.update(status="unmeasured", error=str(e))
            measurements.append(entry)
            continue
        if not result or result.get("value") is None:
            entry.update(status="unmeasured",
                         error=(result or {}).get("error", "no value returned"))
        else:
            entry.update(status="measured", value=result["value"],
                         units=result.get("units"))
            for k, v in result.items():  # carry any extra evidence recorded
                if k not in ("value", "units", "error"):
                    entry[k] = v
        measurements.append(entry)

    any_measured = any(m["status"] == "measured" for m in measurements)
    return {
        "status": "success" if any_measured else "no_measurements",
        "reference_property": reference_property,
        "measurements": measurements,
    }
