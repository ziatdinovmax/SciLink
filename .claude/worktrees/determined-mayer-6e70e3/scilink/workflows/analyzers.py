import logging
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

from ..agents.exp_agents.orchestrator_agent import OrchestratorAgent, AGENT_MAP
from ..agents.exp_agents.hyperspectral_analysis_agent import HyperspectralAnalysisAgent
from ..agents.exp_agents.curve_fitting_agent import CurveFittingAgent 


class BaseExperimentAnalyzer(ABC):
    """Abstract base class for different experimental data analyzers."""
    
    @abstractmethod
    def analyze_for_claims(self, data_path: str, system_info: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """Analyze experimental data and generate scientific claims."""
        pass
    
    @abstractmethod
    def get_data_type_name(self) -> str:
        """Return the name of the data type for logging/display purposes."""
        pass


class MicroscopyAnalyzer(BaseExperimentAnalyzer):
    """Analyzer for microscopy data."""
    
    def __init__(self, agent_id: Optional[int] = None, google_api_key: str = None, 
                 analysis_model: str = "gemini-2.5-pro-preview-06-05", output_dir: str = "",
                 local_model: str = None,
                 enable_human_feedback: bool = False,
                 **kwargs):
        self.agent_id = agent_id
        self.google_api_key = google_api_key
        self.analysis_model = analysis_model
        self.local_model = local_model
        self.output_dir = output_dir
        self.enable_human_feedback = enable_human_feedback
        self.analysis_agent = None
        self.analyzer_kwargs = kwargs
        # if kwargs:
        #     logging.warning(f"Unused arguments passed to MicroscopyAnalyzer: {kwargs}")
    
    def analyze_for_claims(self, data_path: str, system_info: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """Analyze microscopy image and generate scientific claims."""
        
        # Agent selection logic (from microscopy workflow)
        if self.agent_id is not None:
            if self.agent_id not in AGENT_MAP:
                raise ValueError(f"Invalid agent_id: {self.agent_id}. Available agents are: {list(AGENT_MAP.keys())}")
            selected_agent_id = self.agent_id
            logging.info(f"Using manually selected agent: {AGENT_MAP[selected_agent_id].__name__}")
        else:
            logging.info("Using orchestrator to select the best analysis agent...")
            orchestrator = OrchestratorAgent(
                google_api_key=self.google_api_key,
                model_name=self.analysis_model,
                local_model = self.local_model
            )
            selected_agent_id, reasoning = orchestrator.select_agent(
                data_type="microscopy",
                system_info=system_info,
                image_path=data_path
            )
            
            if selected_agent_id == -1:
                return {"error": f"Agent selection failed. Reason: {reasoning}"}

        # Instantiate and run analysis agent
        AnalysisAgentClass = AGENT_MAP[selected_agent_id]
        logging.info(f"✅ Running analysis with: {AnalysisAgentClass.__name__}")
        
        agent_kwargs = {'model_name': self.analysis_model, 
                        'local_model': self.local_model,
                        'enable_human_feedback': self.enable_human_feedback
        }
        if selected_agent_id == 0:  # MicroscopyAnalysisAgent
            agent_kwargs['fft_nmf_settings'] = {
                'FFT_NMF_ENABLED': True,
                'FFT_NMF_AUTO_PARAMS': True,
                'components': 3,
                'output_dir': self.output_dir
            }
        
        if selected_agent_id == 1 and 'sam_settings' in self.analyzer_kwargs: # SAM agent
            agent_kwargs['sam_settings'] = self.analyzer_kwargs['sam_settings']
        
        self.analysis_agent = AnalysisAgentClass(**agent_kwargs)
        
        return self.analysis_agent.analyze_for_claims(data_path, system_info=system_info, **kwargs)
        
    def get_data_type_name(self) -> str:
        return "microscopy"


class SpectroscopyAnalyzer(BaseExperimentAnalyzer):
    """Analyzer for spectroscopy data."""
    
    def __init__(self, google_api_key: str = None, analysis_model: str = "gemini-2.5-pro-preview-06-05", 
                 local_model: str = None,
                 output_dir: str = "", spectral_unmixing_enabled: bool = True,
                 enable_human_feedback: bool = False,
                 run_preprocessing: bool = True,
                 **kwargs):
        self.google_api_key = google_api_key
        self.analysis_model = analysis_model
        self.local_model = local_model,
        self.output_dir = output_dir
        
        # Fixed spectral unmixing settings
        spectral_settings = {
            'method': 'nmf',
            'n_components': 4,
            'normalize': True,
            'enabled': spectral_unmixing_enabled,
            'auto_components': True,
            'max_iter': 500
        }
        
        self.analysis_agent = HyperspectralAnalysisAgent(
            google_api_key=google_api_key,
            model_name=analysis_model,
            local_model=local_model,
            spectral_unmixing_settings=spectral_settings,
            output_dir=output_dir,
            enable_human_feedback=enable_human_feedback,
            run_preprocessing=run_preprocessing
        )

        # if kwargs:
        #     logging.warning(f"Unused arguments passed to SpectroscopyAnalyzer: {kwargs}")
    
    def analyze_for_claims(self, data_path: str, system_info: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """Analyze spectroscopy data and generate scientific claims."""
        structure_image_path = kwargs.get('structure_image_path')
        structure_system_info = kwargs.get('structure_system_info')
        
        return self.analysis_agent.analyze_for_claims(
            data_path,
            metadata_path=system_info,
            structure_image_path=structure_image_path,
            structure_system_info=structure_system_info
        )
    
    def get_data_type_name(self) -> str:
        return "spectroscopy"
    

class CurveAnalyzer(BaseExperimentAnalyzer):
    """Analyzer for 1D curve data."""
    
    def __init__(self, google_api_key: str = None, futurehouse_api_key: str = None, 
                 analysis_model: str = "gemini-2.5-pro-preview-06-05", 
                 output_dir: str = "",
                 run_preprocessing: bool = True,
                 **kwargs):
        self.analysis_agent = CurveFittingAgent(
            google_api_key=google_api_key,
            futurehouse_api_key=futurehouse_api_key,
            model_name=analysis_model,
            output_dir=output_dir,
            run_preprocessing=run_preprocessing,
            **kwargs
        )
    
    def analyze_for_claims(self, data_path: str, system_info: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """Analyze 1D curve data and generate scientific claims."""
        return self.analysis_agent.analyze_for_claims(
            data_path,
            system_info=system_info,
            output_dir=self.analysis_agent.output_dir,
            **kwargs
        )

    def get_data_type_name(self) -> str:
        return "1D Curve"
