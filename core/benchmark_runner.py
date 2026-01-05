"""
벤치마크 실행을 담당하는 모듈

이 모듈은 로드된 모델과 벤치마크 데이터를 사용하여
실제 벤치마크를 실행하고 결과를 수집합니다.

주요 기능:
- 벤치마크 항목별 텍스트 생성
- 생성 시간 및 메모리 사용량 측정
- 답변 정확도 평가
- 결과 데이터 수집 및 통계 계산
"""
import time
import torch
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM

from .answer_evaluator import AnswerEvaluator, EvaluationResult
from .gpu_monitor import GPUMonitor

# config 모듈 import (상대/절대 import 모두 지원)
try:
    from ..config.settings import DEFAULT_GENERATION_PARAMETERS
except ImportError:
    from config.settings import DEFAULT_GENERATION_PARAMETERS


@dataclass
class BenchmarkExecutionResult:
    """
    벤치마크 실행 결과를 담는 데이터 클래스
    
    Attributes:
        model_name: 사용된 모델 이름
        model_type: 모델 타입 설명
        use_quantization: 양자화 사용 여부
        total_questions: 전체 문제 수
        correct_answers: 정답 수
        accuracy_percentage: 정확도 (백분율)
        average_generation_time_seconds: 평균 생성 시간 (초)
        total_execution_time_seconds: 총 실행 시간 (초)
        memory_usage_by_gpu: GPU별 메모리 사용량 (GB)
        individual_results: 각 문제별 상세 결과 리스트
    """
    model_name: str
    model_type: str
    use_quantization: bool
    total_questions: int
    correct_answers: int
    accuracy_percentage: float
    average_generation_time_seconds: float
    total_execution_time_seconds: float
    memory_usage_by_gpu: Dict[int, float] = field(default_factory=dict)
    individual_results: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        결과를 딕셔너리로 변환합니다.
        
        Returns:
            Dict[str, Any]: 결과 딕셔너리
        """
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "use_quantization": self.use_quantization,
            "total_questions": self.total_questions,
            "correct_answers": self.correct_answers,
            "accuracy": self.accuracy_percentage,
            "avg_generation_time": self.average_generation_time_seconds,
            "total_time": self.total_execution_time_seconds,
            "memory_usage": self.memory_usage_by_gpu,
            "results": self.individual_results,
        }


class BenchmarkRunner:
    """
    벤치마크를 실행하는 클래스
    
    이 클래스는 모델과 벤치마크 데이터를 받아서
    각 문제에 대해 텍스트를 생성하고 평가합니다.
    """
    
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        model: AutoModelForCausalLM,
        model_name: str,
        model_type: str = "unknown",
        use_quantization: bool = False,
        generation_parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        BenchmarkRunner 인스턴스 초기화
        
        Args:
            tokenizer: 사용할 토크나이저
            model: 사용할 모델
            model_name: 모델 이름
            model_type: 모델 타입 설명
            use_quantization: 양자화 사용 여부
            generation_parameters: 텍스트 생성 파라미터 (None이면 기본값 사용)
        """
        self.tokenizer = tokenizer
        self.model = model
        self.model_name = model_name
        self.model_type = model_type
        self.use_quantization = use_quantization
        self.generation_parameters = (
            generation_parameters or DEFAULT_GENERATION_PARAMETERS.copy()
        )
        self.answer_evaluator = AnswerEvaluator()
        self.gpu_monitor = GPUMonitor()
    
    def format_question_prompt(self, benchmark_item: Dict[str, Any]) -> str:
        """
        벤치마크 항목을 프롬프트 형식으로 변환합니다.
        
        Args:
            benchmark_item: 표준화된 벤치마크 항목 딕셔너리
        
        Returns:
            str: 모델에 입력할 프롬프트 문자열
        """
        question = benchmark_item["question"]
        options = benchmark_item.get("options", {})
        context = benchmark_item.get("context", "가장 적합한 답을 하나만 고르시오.")
        
        prompt_parts = [context, "", question]
        
        if options:
            for key in sorted(options.keys()):
                prompt_parts.append(f"{key}. {options[key]}")
        
        prompt_parts.append("")
        prompt_parts.append("답변:")
        
        return "\n".join(prompt_parts)
    
    def generate_answer(
        self, prompt: str
    ) -> Tuple[str, float]:
        """
        주어진 프롬프트에 대해 답변을 생성합니다.
        
        Args:
            prompt: 입력 프롬프트 문자열
        
        Returns:
            tuple[str, float]: (생성된 답변 텍스트, 생성 시간(초))
        """
        # 토크나이징
        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        
        # 생성 파라미터 준비
        generation_kwargs = self.generation_parameters.copy()
        generation_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
        
        # 텍스트 생성
        start_time = time.time()
        try:
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generation_kwargs)
            generation_time = time.time() - start_time
            
            # 디코딩
            generated_full_text = self.tokenizer.decode(
                outputs[0], skip_special_tokens=True
            )
            
            # 프롬프트 제거 (생성된 부분만 추출)
            generated_answer = self._extract_answer_from_generated_text(
                generated_full_text, prompt
            )
            
            return generated_answer, generation_time
        
        except Exception as error:
            generation_time = time.time() - start_time
            print(f"  ❌ 생성 실패: {error}")
            return "", generation_time
    
    def _extract_answer_from_generated_text(
        self, full_text: str, original_prompt: str
    ) -> str:
        """
        생성된 전체 텍스트에서 답변 부분만 추출합니다.
        
        Args:
            full_text: 생성된 전체 텍스트 (프롬프트 + 답변)
            original_prompt: 원본 프롬프트
        
        Returns:
            str: 추출된 답변 텍스트
        """
        # 프롬프트 제거
        if original_prompt in full_text:
            answer = full_text.split(original_prompt, 1)[-1].strip()
        else:
            # "답변:" 또는 "답:" 이후 부분 추출
            for separator in ["답변:", "답:"]:
                if separator in full_text:
                    answer = full_text.split(separator, 1)[-1].strip()
                    break
            else:
                answer = full_text.strip()
        
        return answer
    
    def run_benchmark(
        self, benchmark_data: List[Dict[str, Any]]
    ) -> BenchmarkExecutionResult:
        """
        벤치마크를 실행합니다.
        
        Args:
            benchmark_data: 표준화된 벤치마크 데이터 리스트
        
        Returns:
            BenchmarkExecutionResult: 벤치마크 실행 결과
        """
        print(f"\n{'='*80}")
        print(f"{self.model_type} 벤치마크 실행")
        print(f"{'='*80}")
        
        # 초기 메모리 측정
        initial_memory_by_gpu = self._get_gpu_memory_usage()
        
        # 벤치마크 실행
        total_start_time = time.time()
        individual_results: List[Dict[str, Any]] = []
        
        for index, item in enumerate(benchmark_data, start=1):
            print(f"\n[{index}/{len(benchmark_data)}] {item.get('category', 'N/A')}")
            print(f"질문: {item['question'][:80]}...")
            
            # 프롬프트 생성
            prompt = self.format_question_prompt(item)
            
            # 답변 생성
            generated_answer, generation_time = self.generate_answer(prompt)
            
            # 정답 평가
            evaluation_result = self.answer_evaluator.evaluate(
                generated_answer,
                item.get("correct_answer_indices", []),
                item.get("correct_answer_texts", []),
            )
            
            print(f"  정답: {item.get('correct_answer_texts', ['N/A'])[0]}")
            print(f"  생성된 답변: {generated_answer[:100]}...")
            print(f"  정확도: {'✓' if evaluation_result.is_correct else '✗'}")
            print(f"  생성 시간: {generation_time:.2f}초")
            
            # 결과 저장
            individual_results.append({
                "no": item.get("item_number", index),
                "category": item.get("category", "N/A"),
                "question": item["question"],
                "correct_answer_idx": item.get("correct_answer_indices", []),
                "correct_answer": item.get("correct_answer_texts", []),
                "generated_answer": generated_answer,
                "is_correct": evaluation_result.is_correct,
                "generation_time": generation_time,
                "evaluation_method": evaluation_result.evaluation_method,
            })
        
        total_execution_time = time.time() - total_start_time
        
        # 최종 메모리 측정
        final_memory_by_gpu = self._get_gpu_memory_usage()
        
        # 메모리 사용량 계산
        memory_usage_by_gpu = {
            gpu_index: final_memory - initial_memory_by_gpu.get(gpu_index, 0.0)
            for gpu_index, final_memory in final_memory_by_gpu.items()
        }
        
        # 통계 계산
        total_questions = len(individual_results)
        correct_answers = sum(
            1 for result in individual_results if result["is_correct"]
        )
        accuracy_percentage = (
            (correct_answers / total_questions) * 100.0
            if total_questions > 0
            else 0.0
        )
        average_generation_time = (
            sum(result["generation_time"] for result in individual_results)
            / total_questions
            if total_questions > 0
            else 0.0
        )
        
        return BenchmarkExecutionResult(
            model_name=self.model_name,
            model_type=self.model_type,
            use_quantization=self.use_quantization,
            total_questions=total_questions,
            correct_answers=correct_answers,
            accuracy_percentage=accuracy_percentage,
            average_generation_time_seconds=average_generation_time,
            total_execution_time_seconds=total_execution_time,
            memory_usage_by_gpu=memory_usage_by_gpu,
            individual_results=individual_results,
        )
    
    def _get_gpu_memory_usage(self) -> Dict[int, float]:
        """
        현재 GPU별 메모리 사용량을 반환합니다.
        
        Returns:
            Dict[int, float]: GPU 인덱스를 키로 하는 메모리 사용량(GB) 딕셔너리
        """
        memory_usage: Dict[int, float] = {}
        for gpu_index in range(torch.cuda.device_count()):
            memory_usage[gpu_index] = (
                torch.cuda.memory_allocated(gpu_index) / 1e9
            )
        return memory_usage

