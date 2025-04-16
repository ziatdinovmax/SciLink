# SciLinkLLM
A framework aimed at bridging experimental osbervations with computational materials modeling using large language models. For now it is limited to microsocpy images and inputs to DFT simulations.

### Workflow

1. Image Analysis: The Microscopy Agent analyzes the uploaded image with contextual metadata and recommends structures for DFT simulations.
2. Recommendation Selection: User selects from structural recommendations
3. Structure Generation: Strucute Generator Agent generates and executes ASE-based scripts scripts to create the selected atomic structures

### Requirements

- Python 3.11+
- Google Generative AI API access (see [here](https://ai.google.dev/gemini-api/docs/api-key))
- ASE (Atomic Simulation Environment)
- OpenCV and Pillow for image processing

### How to use
1. Edit ```config.py``` to set:

- Image and experimental metadata file paths
- Model selection for analysis and generation
- Additional generation instructions

2. Run ```python exp2dft.py```