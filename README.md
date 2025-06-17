# SciLink

A Python framework for connecting experimental materials science (microscopy, spectroscopy) with computational modeling (DFT) and automated literature analysis using Large Language Models.

![SciLink Logo](misc/scilink_logo_v2.svg)

## Overview

SciLink uses a system of intelligent agents to automate the research cycle from experimental observation to computational insight. It streamlines the process of analyzing experimental data, assessing the novelty of findings against the scientific literature, and setting up computational simulations to investigate those findings.

## Core Concepts

- 🤖 Agent-Based Architecture: The framework is built on a collection of specialized agents, each designed for a specific scientific task:

   - 🔬 Experimental Agents: Analyze microscopy images or spectroscopy data to extract features and generate scientific claims.
   - 📚 Literature Agent: Queries scientific databases (via FutureHouse's OWL) to validate claims and assess novelty.
   - ⚛️ Simulation Agents: Generate, validate, and refine atomic structures (using ASE) and prepare input files for DFT calculations.

- 🔄 Automated Workflows: High-level workflows chain these agents together to perform complex tasks with minimal user intervention.


## Installation

### Standard Installation
```bash
pip install scilink
```

### Development Installation
```bash
git clone https://github.com/scilink/scilink.git
cd scilink
pip install -e .[full]
```

## Quick Start

### 1. Configure API Keys

```python
import scilink

# Configure required API keys
scilink.configure('google', 'your-google-api-key')
scilink.configure('futurehouse', 'your-futurehouse-key')  # Optional
```

### 2. Analyze Microscopy Data

```python
from scilink import MicroscopyNoveltyAssessmentWorkflow

# Create workflow
workflow = MicroscopyNoveltyAssessmentWorkflow(
    output_dir="microscopy_analysis"
)

# Run complete analysis
result = workflow.run_complete_workflow(
    image_path="my_stem_image.png",
    system_info={"material": "MoS2", "measurements": "HAADF-STEM"}
)

print(f"Status: {result['final_status']}")
print(f"Novel findings: {len(result['novelty_assessment']['potentially_novel'])}")
```

### 3. Generate DFT Structures

```python
from scilink import DFTWorkflow

# Create DFT workflow
dft_workflow = DFTWorkflow(output_dir="dft_structures")

# Generate structure from description
result = dft_workflow.run_complete_workflow(
    "3x3 Si supercell with oxygen interstitial defect"
)

print(f"Structure generated: {result['final_status']}")
print(f"Files: POSCAR, INCAR, KPOINTS ready for VASP")
```

### 4. Complete Experimental Pipeline

```python
from scilink import Experimental2DFT

# Complete pipeline from experimental data to structures
pipeline = Experimental2DFT(output_dir="complete_analysis")

# Run end-to-end workflow
result = pipeline.run_complete_pipeline(
    data_path="microscopy_image.png",
    data_type="microscopy", 
    system_info={"material": "MoS2", "growth_method": "MOCVD", "measurements": "HAADF-STEM"},
    interactive=True
)

print(f"Pipeline status: {result['final_status']}")
print(f"Structures generated: {len(result['generated_structures'])}")
```

## Workflows

### Microscopy Novelty Assessment
Analyzes microscopy images to identify novel features and validate against literature:

1. **Image Analysis**: Identifies defects, interfaces, and other structural features
2. **Claims Generation**: Extracts specific scientific observations
3. **Literature Search**: Validates claims against existing publications
4. **Novelty Assessment**: Produces a report of potentially novel findings.

### Spectroscopy Novelty Assessment  
Processes hyperspectral data with advanced spectral unmixing:

1. **Spectral Analysis**: Automated component identification and spatial mapping
2. **Claims Extraction**: Generates spectroscopy-specific scientific claims
3. **Literature Validation**: Searches for similar reports in the literature
4. **Novelty Evaluation**: Highlights unique aspects (if any) of current observations

### DFT Structure Generation
Converts experimental observations into computational models:

1. **Structure Building**: Automated generation using ASE and specialized tools
2. **Structure Validation**: Multi-modal structure validation with automatic improvements
2. **VASP Input Creation**: INCAR and KPOINTS for different calculation types
3. **VASP Inputs Validation**: Cross-references INCAR parameters with literature to ensure best practices.

### Complete Experimental Pipeline
End-to-end automation from experimental data to computational structures:

1. **Multi-modal Analysis**: Supports both microscopy and spectroscopy data
2. **Novelty Assessment**: Literature-informed evaluation of findings
3. **Structure Selection**: Interactive or automated selection of structures to generate
4. **Batch Generation**: Creates multiple structures with complete VASP input files


## API Reference

### Core Classes

- `MicroscopyNoveltyAssessmentWorkflow`: Complete microscopy analysis pipeline
- `SpectroscopyNoveltyAssessmentWorkflow`: Hyperspectral data analysis
- `DFTRecommendationsWorkflow`: Generate structure recommendations
- `DFTWorkflow`: Complete structure generation pipeline  
- `Experimental2DFT`: End-to-end experimental to computational workflow

### Agents

- `GeminiMicroscopyAnalysisAgent`: AI-powered microscopy analysis
- `GeminiSpectroscopyAnalysisAgent`: Intelligent spectroscopy processing
- `StructureGenerator`: Automated atomic structure generation
- `StructureValidatorAgent`: AI structure validation
- `OwlLiteratureAgent`: Literature search and validation

## Requirements

- Python ≥ 3.11
- Google Generative AI API key (required)
- FutureHouse API key (optional, for literature search)
- Materials Project API key (optional)