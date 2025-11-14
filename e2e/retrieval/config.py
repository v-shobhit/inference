"""
Configuration dataclasses for the retrieval pipeline.

Provides type-safe configuration with automatic validation, replacing 
dictionary-based config with proper dataclasses.
"""

from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class DataConfig:
    """Data source configuration."""
    vector_store: Optional[str] = None
    passages: Optional[str] = None
    tsv_path: Optional[str] = None
    dataset_split: str = "test"
    num_prompts: Optional[int] = None
    passage_count: Optional[int] = None  # For limiting passages during ingestion
    
    def validate(self, check_files: bool = False):
        """
        Validate data configuration.
        
        Args:
            check_files: If True, check if files exist on filesystem.
                        Set to False for testing or when files will be created later.
        """
        if not self.vector_store and not self.passages:
            raise ValueError("Must specify either 'vector_store' or 'passages'")
        
        if check_files:
            if self.vector_store and not Path(self.vector_store).exists():
                raise FileNotFoundError(f"Vector store not found: {self.vector_store}")
            
            if self.passages and not Path(self.passages).exists():
                raise FileNotFoundError(f"Passages file not found: {self.passages}")
        
        if self.num_prompts is not None and self.num_prompts <= 0:
            raise ValueError(f"num_prompts must be positive, got {self.num_prompts}")


@dataclass
class RetrievalConfig:
    """Retrieval configuration."""
    model: str = "intfloat/e5-base-v2"
    top_k: int = 20
    
    def validate(self):
        """Validate retrieval configuration."""
        if self.top_k <= 0:
            raise ValueError(f"top_k must be positive, got {self.top_k}")


@dataclass
class RerankerConfig:
    """Reranker configuration."""
    enabled: bool = True
    model: str = "colbert-ir/colbertv2.0"
    top_p: Optional[float] = None
    
    def validate(self):
        """Validate reranker configuration."""
        if self.top_p is not None:
            if not self.enabled:
                raise ValueError("top_p requires reranker to be enabled")
            if not 0.0 < self.top_p <= 1.0:
                raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")


@dataclass
class RewriterConfig:
    """Query rewriter configuration."""
    enabled: bool = False
    endpoint: Optional[str] = None
    model: Optional[str] = None
    steps: int = 1
    queries_per_step: int = 3
    temperature: float = 0.7
    max_tokens: int = 500
    api_key: str = "EMPTY"
    
    def validate(self):
        """Validate rewriter configuration."""
        if self.enabled:
            if not self.endpoint:
                raise ValueError("Rewriter requires 'endpoint' when enabled")
            if not self.model:
                raise ValueError("Rewriter requires 'model' when enabled")
        
        if self.steps <= 0:
            raise ValueError(f"steps must be positive, got {self.steps}")
        
        if self.queries_per_step <= 0:
            raise ValueError(f"queries_per_step must be positive, got {self.queries_per_step}")
        
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"temperature must be in [0, 2], got {self.temperature}")
        
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")


@dataclass
class OutputConfig:
    """Output configuration."""
    results: str = "retrieval_results.pkl"
    rewriter_io: Optional[str] = None
    retriever_io: Optional[str] = None
    reranker_io: Optional[str] = None
    save_json: bool = False
    save_csv: bool = False
    verbose: bool = False


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""
    data: DataConfig
    device: Optional[str] = None
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    rewriter: RewriterConfig = field(default_factory=RewriterConfig)
    parallel: Optional[int] = None  # max_workers (None or 1 = sequential)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'PipelineConfig':
        """
        Create PipelineConfig from dictionary (YAML/JSON).
        
        Args:
            config_dict: Configuration dictionary from YAML file
            
        Returns:
            PipelineConfig instance
        """
        # Extract sections with defaults
        data_dict = config_dict.get('data', {})
        retrieval_dict = config_dict.get('retrieval', {})
        reranker_dict = config_dict.get('reranker', {})
        rewriter_dict = config_dict.get('rewriter', {})
        output_dict = config_dict.get('output', {})
        parallel_dict = config_dict.get('parallel', {})
        
        return cls(
            data=DataConfig(**data_dict),
            device=config_dict.get('device'),
            retrieval=RetrievalConfig(**retrieval_dict),
            reranker=RerankerConfig(**reranker_dict),
            rewriter=RewriterConfig(**rewriter_dict),
            parallel=parallel_dict.get('max_workers') if parallel_dict else None,
            output=OutputConfig(**output_dict)
        )
    
    def to_dict(self) -> dict:
        """
        Convert to dictionary for backward compatibility.
        
        Returns:
            Dictionary representation matching old config format
        """
        return {
            'data': {
                'vector_store': self.data.vector_store,
                'passages': self.data.passages,
                'tsv_path': self.data.tsv_path,
                'dataset_split': self.data.dataset_split,
                'num_prompts': self.data.num_prompts,
                'passage_count': self.data.passage_count
            },
            'device': self.device,
            'retrieval': {
                'model': self.retrieval.model,
                'top_k': self.retrieval.top_k
            },
            'reranker': {
                'enabled': self.reranker.enabled,
                'model': self.reranker.model,
                'top_p': self.reranker.top_p
            },
            'rewriter': {
                'enabled': self.rewriter.enabled,
                'endpoint': self.rewriter.endpoint,
                'model': self.rewriter.model,
                'steps': self.rewriter.steps,
                'queries_per_step': self.rewriter.queries_per_step,
                'temperature': self.rewriter.temperature,
                'max_tokens': self.rewriter.max_tokens,
                'api_key': self.rewriter.api_key
            },
            'parallel': {
                'max_workers': self.parallel
            } if self.parallel else {},
            'output': {
                'results': self.output.results,
                'rewriter_io': self.output.rewriter_io,
                'retriever_io': self.output.retriever_io,
                'reranker_io': self.output.reranker_io,
                'save_json': self.output.save_json,
                'save_csv': self.output.save_csv,
                'verbose': self.output.verbose
            }
        }
    
    def validate(self, check_files: bool = False):
        """
        Validate all configuration sections.
        
        Args:
            check_files: If True, check if data files exist on filesystem.
        """
        self.data.validate(check_files=check_files)
        self.retrieval.validate()
        self.reranker.validate()
        self.rewriter.validate()
        # output doesn't need validation
    
    def apply_overrides(self, args) -> None:
        """
        Apply command-line argument overrides.
        
        Args:
            args: Parsed argparse Namespace with override values
        """
        if hasattr(args, 'num_prompts') and args.num_prompts is not None:
            self.data.num_prompts = args.num_prompts
        
        if hasattr(args, 'verbose') and args.verbose:
            self.output.verbose = True
        
        if hasattr(args, 'output') and args.output:
            self.output.results = args.output
        
        if hasattr(args, 'max_workers') and args.max_workers is not None:
            self.parallel = args.max_workers

