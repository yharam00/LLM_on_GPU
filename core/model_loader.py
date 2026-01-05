"""
대규모 언어 모델(LLM)의 로딩 및 관리를 담당하는 모듈

이 모듈은 Hugging Face Transformers 라이브러리를 활용하여
다양한 크기의 언어 모델을 단일 GPU 또는 다중 GPU 환경에서
효율적으로 로드하고 관리하는 기능을 제공합니다.

주요 기능:
- 양자화를 통한 메모리 효율적인 모델 로딩
- 다중 GPU 자동 분산
- 토크나이저 설정 자동화
- 모델 파라미터 분산 상태 확인
"""
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class QuantizationConfiguration:
    """
    4-bit 양자화 설정을 담는 불변 데이터 클래스
    
    Attributes:
        load_in_4bit: 4-bit 양자화 사용 여부
        compute_dtype: 양자화된 모델의 계산 데이터 타입
        use_double_quant: 이중 양자화 사용 여부
        quant_type: 양자화 타입 (기본값: "nf4")
    """
    load_in_4bit: bool
    compute_dtype: torch.dtype = torch.bfloat16
    use_double_quant: bool = True
    quant_type: str = "nf4"
    
    def to_bitsandbytes_config(self) -> BitsAndBytesConfig:
        """
        BitsAndBytesConfig 객체로 변환
        
        Returns:
            BitsAndBytesConfig: Hugging Face 양자화 설정 객체
        """
        return BitsAndBytesConfig(
            load_in_4bit=self.load_in_4bit,
            bnb_4bit_compute_dtype=self.compute_dtype,
            bnb_4bit_use_double_quant=self.use_double_quant,
            bnb_4bit_quant_type=self.quant_type,
        )


class ModelLoader:
    """
    대규모 언어 모델을 로드하고 관리하는 클래스
    
    이 클래스는 모델 로딩의 복잡성을 캡슐화하여
    단순한 인터페이스로 다양한 설정의 모델을 로드할 수 있게 합니다.
    
    Attributes:
        model_name: Hugging Face 모델 식별자 또는 로컬 경로
        tokenizer: 로드된 토크나이저 객체
        model: 로드된 모델 객체
        device_map: 모델이 분산된 GPU 매핑 정보
    """
    
    def __init__(
        self,
        model_name: str,
        quantization_config: Optional[QuantizationConfiguration] = None,
        dtype: torch.dtype = torch.bfloat16,
        trust_remote_code: bool = True,
    ) -> None:
        """
        ModelLoader 인스턴스 초기화
        
        Args:
            model_name: Hugging Face 모델 이름 또는 로컬 경로
            quantization_config: 양자화 설정 (None이면 양자화 미사용)
            dtype: 모델의 데이터 타입 (양자화 사용 시 무시됨)
            trust_remote_code: 원격 코드 실행 허용 여부
        """
        self.model_name: str = model_name
        self.quantization_config: Optional[QuantizationConfiguration] = quantization_config
        self.dtype: torch.dtype = dtype
        self.trust_remote_code: bool = trust_remote_code
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForCausalLM] = None
        self.device_map: Dict[str, int] = {}
    
    def load(self) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
        """
        모델과 토크나이저를 로드합니다.
        
        Returns:
            Tuple[AutoTokenizer, AutoModelForCausalLM]: 
                로드된 토크나이저와 모델의 튜플
            
        Raises:
            RuntimeError: 모델 로딩 실패 시
        """
        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
        )
        
        # pad_token이 없으면 eos_token으로 설정
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 양자화 설정 준비
        quantization_config_object = None
        if self.quantization_config is not None:
            quantization_config_object = self.quantization_config.to_bitsandbytes_config()
        
        # 모델 로드
        model_kwargs: Dict[str, Any] = {
            "device_map": "auto",
            "trust_remote_code": self.trust_remote_code,
            "low_cpu_mem_usage": True,
        }
        
        if quantization_config_object is not None:
            model_kwargs["quantization_config"] = quantization_config_object
        else:
            model_kwargs["dtype"] = self.dtype
            model_kwargs["torch_dtype"] = self.dtype
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs,
            )
        except Exception as error:
            raise RuntimeError(
                f"모델 로딩 실패: {self.model_name}"
            ) from error
        
        # 디바이스 매핑 정보 수집
        self._collect_device_map()
        
        return self.tokenizer, self.model
    
    def _collect_device_map(self) -> None:
        """
        모델 파라미터가 분산된 GPU 정보를 수집합니다.
        
        각 GPU에 할당된 파라미터 수를 계산하여
        모델의 분산 상태를 파악할 수 있게 합니다.
        """
        if self.model is None:
            return
        
        device_param_count: Dict[str, int] = {}
        for param in self.model.parameters():
            device_str = str(param.device)
            if device_str not in device_param_count:
                device_param_count[device_str] = 0
            device_param_count[device_str] += param.numel()
        
        self.device_map = device_param_count
    
    def get_device_map_summary(self) -> Dict[str, Dict[str, Any]]:
        """
        모델의 GPU 분산 상태를 요약한 딕셔너리를 반환합니다.
        
        Returns:
            Dict[str, Dict[str, Any]]: 
                각 디바이스별 파라미터 수와 비율을 담은 딕셔너리
                키: 디바이스 문자열 (예: "cuda:0")
                값: {"parameter_count": int, "percentage": float}
        """
        if not self.device_map:
            return {}
        
        total_parameters = sum(self.device_map.values())
        summary: Dict[str, Dict[str, Any]] = {}
        
        for device, param_count in self.device_map.items():
            percentage = (param_count / total_parameters) * 100.0
            summary[device] = {
                "parameter_count": param_count,
                "percentage": percentage,
            }
        
        return summary
    
    def get_input_device(self) -> torch.device:
        """
        모델의 입력 텐서가 위치해야 할 디바이스를 반환합니다.
        
        Returns:
            torch.device: 모델의 첫 번째 파라미터가 위치한 디바이스
        """
        if self.model is None:
            raise RuntimeError("모델이 로드되지 않았습니다.")
        return next(self.model.parameters()).device
    
    def cleanup(self) -> None:
        """
        모델과 토크나이저를 메모리에서 해제하고 GPU 캐시를 비웁니다.
        
        대규모 모델을 사용한 후 메모리 정리를 위해 호출합니다.
        """
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        torch.cuda.empty_cache()

