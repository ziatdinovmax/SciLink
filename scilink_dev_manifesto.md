# SciLink Development Manifesto: Core Principles


![Architecture](architecture.jpg)


### 1. The Principle of Stateful Memory
**Philosophy:** Science requires reproducibility. An agent’s decision is worthless if the reasoning behind it is lost. Every agent acts as a "Black Box Recorder," persisting its entire lifecycle to disk.

* **Requirement:** Every agent must maintain a `self.state` dictionary (can be inherited from `BaseAgent`)
* **Persistence:** State is saved to JSON after every atomic action.
* **Traceability:** The state record must capture the **Full Context Chain**:
    * *Input:* "What was asked?"
    * *Rationale:* "Why did the LLM choose this path?" (Chain-of-Thought).
    * *Action:* "What code or tool was executed?"
    * *Result:* "What was the raw output?"
    * *Feedback:* "Did a human intervene or correct the action?"

### 2. The Principle of Atomic Tooling
**Philosophy:** Complex workflows are built from small, testable units of discrete, stateless functions. The system supports both strict protocols for reproducibility and dynamic chaining for exploration.

* **Structure:** Agents expose capabilities as clearly defined **Tools** (e.g. `generate_input_structure()`, `generate_force_fields()`) rather than giant `run()` methods.
* **Mode 1 - Strict:** The system maintains a Pipeline Registry of hard-coded sequences (SOPs). The AI can browse this registry and execute a verified pipeline without deviation.
* **Mode 2 - Dynamic:** If no matching pipeline exists, the AI is permitted to compose atomic tools into a custom sequence to solve the problem. If a specific tool fails, the agent can retry with different parameters or swap tools.

### 3. The Principle of Standardized I/O
**Philosophy:** In a multi-agent system, ambiguity is the enemy. Data passed between agents must follow strict schemas to ensure the "Simulation Agent" speaks the same language as the "Experimental Agent."

* **Typed Interfaces:** All inputs and outputs must be validated using strictly typed dictionaries (we may consider using Pdantic models).
* **Explicit Metadata:** Numeric outputs must never be "naked numbers." They must include units and context (e.g., `{"value": 50, "unit": "MPa", "method": "tensile_test"}`).
* **Schema Registry:** A shared registry defines common data structures (e.g., `CrystalStructure`, `Spectra1D`) to ensure interoperability across domain boundaries.

### 4. The Principle of Human-in-the-Loop
**Philosophy:** The AI is a partner, not a replacement. The system must be designed to yield control to the human scientist at critical junctions, ensuring safety and trust.

* **Breakpoint Architecture:** Agents must support "Dry Run" modes where they generate a plan or code but pause for approval before execution.
* **Feedback Integration:** Human feedback is treated as a first-class input. It is not just an interruption; it is recorded in the state and used to immediately refine the agent's context (e.g., "The user said this precursor is toxic; replan synthesis").
* **Override History:** Every human override is logged. This creates a dataset of "AI Failure -> Human Correction" that is invaluable for future fine-tuning.

### 5. The Principle of Decoupled Development
**Philosophy:** Different domains (Experiment, Theory, Planning) are built by separate teams with distinct expertise. Agents are self-contained packages that share no domain logic, but strictly adhere to the **Shared Architectural Contract** (Principles 1-4).

* **Independence:** Agents in the "Simulation Pool" must never import code from the "Experimental Pool".
* **Orchestrated Integration:** Integration happens *only* at the Orchestrator level. The Orchestrator acts as the "API Glue" that translates a "Simulation Result" into a "Planning Input", etc.
* **Parallel Evolution:** A team can upgrade the internal physics of a `DFT_Agent` without breaking the `Planning_Agent`, as long as they maintain the standard I/O and statefulness required by the framework.

### 6. TBA