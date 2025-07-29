import re
from pathlib import Path
from typing import Dict, Any

from .vasp_agent import VaspInputAgent

class VaspErrorUpdaterAgent:
    """
    Inline‐updater: takes the VASP error log + old INCAR/KPOINTS,
    and uses VaspInputAgent.generate_vasp_inputs to propose fixes.
    """
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-pro-preview-06-05"):
        self.vasp_agent = VaspInputAgent(api_key, model_name)

    def _extract_errors(self, log: str) -> str:
        patterns = [r"Fatal error.*", r"ERROR.*", r"KPAR.*", r"too many k-points.*"]
        errs = []
        for pat in patterns:
            errs += [m.strip() for m in re.findall(pat, log, flags=re.IGNORECASE)]
        return "\n".join(errs) or "\n".join(log.splitlines()[:20])

    def refine_inputs(
        self,
        poscar_path: str,
        incar_path: str,
        kpoints_path: str,
        vasp_log: str,
        original_request: str
    ) -> Dict[str, Any]:
        # Read old files
        incar_txt   = Path(incar_path).read_text()
        kpoints_txt = Path(kpoints_path).read_text()

        # Build allowed‑keys guard so the LLM can’t invent new parameters
        allowed_keys = [
            line.split('=')[0].strip()
            for line in incar_txt.splitlines()
            if '=' in line and not line.strip().startswith('#')
        ]
        
        allowed_line = (
            "You may only modify these INCAR parameters: "
            + ", ".join(allowed_keys)
            + ". Do not add any keys not in this list.\n\n"
        )

        # Error snippet
        snippet = self._extract_errors(vasp_log)

        # Pull out any “please specify X in the INCAR file” advice
        advice_match = re.search(
            r"please specify ([A-Za-z0-9_,\s]+?) in the INCAR file",
            snippet,
            flags=re.IGNORECASE
        )
        advice_line = ""
        if advice_match:
            advice_params = advice_match.group(1).strip()
            advice_line = (
                f"Based on the VASP log, please include these INCAR parameters: "
                f"{advice_params}.\n\n"
            )

        # Build prompt (you can also inject a full JSON constraints blob here)
        prompt = (
            allowed_line
            + advice_line
            + f"The VASP run for \"{original_request}\" failed with:\n\n{snippet}\n\n"
            + f"Original INCAR:\n{incar_txt}\n\n"
            + f"Original KPOINTS:\n{kpoints_txt}\n\n"
            + "Please reply with a JSON object with keys:\n"
            '  "suggested_incar": full revised INCAR,\n'
            '  "suggested_kpoints": full revised KPOINTS,\n'
            '  "explanation": rationale for each change.\n'
        )

        vasp_res = self.vasp_agent.generate_vasp_inputs(
            poscar_path=poscar_path,
            original_request=prompt
        )
        
        if vasp_res.get("status") != "success":
            return {"status": "error", "message": vasp_res.get("message", "")}

        return {
            "status":            "success",
            "suggested_incar":   vasp_res["incar"],
            "suggested_kpoints": vasp_res["kpoints"],
            "explanation":       vasp_res.get("explanation", "")
        }
        # now ask the LLM itself to explain those changes
        # import the same client your other agents use:
        from scilink.agents.sim_agents.val_agent import LLMClient
        llm = LLMClient(api_key=self.vasp_agent.api_key, model=self.vasp_agent.model_name)

        rationale_prompt = (
            f"I just proposed the following INCAR/KPOINTS updates for “{original_request}”:\n\n"
            f"--- INCAR ---\n{plan['suggested_incar']}\n\n"
            f"--- KPOINTS ---\n{plan['suggested_kpoints']}\n\n"
            "Please give me, in plain text, the reason for each change."
        )
        explanation = llm.generate(prompt=rationale_prompt, model=self.vasp_agent.model_name)
        plan["explanation"] = explanation.strip()

        return plan
        
