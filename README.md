# SciLinkLLM
A framework aimed at bridging experimental observations with computational materials modeling and literature analysis using large language models. For now it is limited to microscopy images.

## Workflows

### Experiment to DFT Workflow

1. **Image Analysis**: The Microscopy Agent analyzes the uploaded image with contextual metadata and recommends structures for DFT simulations.
2. **Recommendation Selection**: User selects from structural recommendations
3. **Structure Generation**: Structure Generator Agent generates and executes ASE-based scripts to create the selected atomic structures

### Experiment to Literature Workflow

1. **Image Analysis for Claims**: The Microscopy Agent analyzes the uploaded image with contextual metadata and generates scientific claims based on observed features.
2. **Claim Selection**: User selects which claims to validate against existing literature
3. **Literature Search**: The OWL Literature Agent queries scientific databases to determine if similar observations have been reported and provides evidence from relevant publications.
4. **Novelty Assessment**: The system identifies potentially novel findings versus those already reported in literature.

### Experiment to Claims Workflow

1. **Image Analysis for Claims**: The Microscopy Agent analyzes the uploaded image with contextual metadata and generates scientific claims based on observed features.
2. **Claims Output**: The system creates a structured report of the claims for further use.

## Requirements

- Python 3.11+
- Google Generative AI API access (see [here](https://ai.google.dev/gemini-api/docs/api-key))
- ASE (Atomic Simulation Environment)
- OpenCV and Pillow for image processing
- FutureHouse API key for literature search workflow (only needed for exp2lit.py)

## How to use

### For Experiment to DFT Workflow:

1. Edit ```config.py``` to set:
   - Image and experimental metadata file paths
   - Model selection for analysis and generation
   - Additional generation instructions

2. Run ```python exp2dft.py```

**Note**: In `exp2dft.py`, LLM generates and executes ASE code to build atomic structures. As a good general practice, it's recommended to run LLM-generated code in an isolated environment such as a Docker container or a virtual machine like Google Colab. The `example.ipynb` notebook is provided specifically for running the workflow in Google Colab's isolated environment.

### For Experiment to Literature Workflow:

1. Edit ```config.py``` to set:
   - Image and experimental metadata file paths
   - Model selection for analysis
   - FutureHouse API key for OWL literature agent

2. Run ```python exp2lit.py```

### For Experiment to Claims Workflow:

1. Edit ```config.py``` as above for image and model settings

2. Run ```python exp2claims.py```
