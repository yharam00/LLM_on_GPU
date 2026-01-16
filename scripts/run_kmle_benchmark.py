#!/usr/bin/env python3
"""
KMLE 2023 벤치마크 실행 스크립트

이 스크립트는 KMLE 2023 벤치마크 데이터를 사용하여
여러 모델의 성능을 비교 평가합니다.

사용법:
    python run_kmle_benchmark.py

기능:
- TinyLlama 모델 벤치마크 실행
- DeepSeek R1 70B 모델 벤치마크 실행 (4-bit 양자화)
- 결과를 JSON 파일로 저장
- 실행 요약 정보 출력
"""
import json
import sys
import torch
from pathlib import Path
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core import (
    ModelLoader,
    QuantizationConfiguration,
    KMLEBenchmarkDataLoader,
    BenchmarkRunner,
)
from config.settings import (
    DEFAULT_BENCHMARK_PATHS,
    RESULTS_DIRECTORY,
    MODELS_DIRECTORY,
)


def load_and_run_tinyllama_benchmark(
    benchmark_data: list,
) -> dict:
    """
    TinyLlama 모델을 로드하고 벤치마크를 실행합니다.
    
    Args:
        benchmark_data: 표준화된 벤치마크 데이터 리스트
    
    Returns:
        dict: 벤치마크 실행 결과 딕셔너리
    """
    print("\n" + "=" * 80)
    print("1단계: TinyLlama 모델 테스트")
    print("=" * 80)
    
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    try:
        print(f"\n모델 로드: {model_name}")
        model_loader = ModelLoader(
            model_name=model_name,
            quantization_config=None,
            dtype=torch.bfloat16,
            cache_dir=MODELS_DIRECTORY,
        )
        tokenizer, model = model_loader.load()
        
        runner = BenchmarkRunner(
            tokenizer=tokenizer,
            model=model,
            model_name=model_name,
            model_type="TinyLlama",
            use_quantization=False,
        )
        
        result = runner.run_benchmark(benchmark_data)
        result_dict = result.to_dict()
        
        # 메모리 정리
        model_loader.cleanup()
        
        return result_dict
    
    except Exception as error:
        print(f"❌ TinyLlama 테스트 실패: {error}")
        import traceback
        traceback.print_exc()
        return None


def load_and_run_deepseek_benchmark(
    benchmark_data: list,
) -> dict:
    """
    DeepSeek R1 70B 모델을 로드하고 벤치마크를 실행합니다.
    
    Args:
        benchmark_data: 표준화된 벤치마크 데이터 리스트
    
    Returns:
        dict: 벤치마크 실행 결과 딕셔너리
    """
    print("\n" + "=" * 80)
    print("2단계: DeepSeek R1 70B 모델 테스트")
    print("=" * 80)
    
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
    
    try:
        print(f"\n모델 로드: {model_name}")
        print("4-bit 양자화 사용 중...")
        
        quantization_config = QuantizationConfiguration(
            load_in_4bit=True,
            compute_dtype=torch.bfloat16,
            use_double_quant=True,
            quant_type="nf4",
        )
        
        model_loader = ModelLoader(
            model_name=model_name,
            quantization_config=quantization_config,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            cache_dir=MODELS_DIRECTORY,
        )
        tokenizer, model = model_loader.load()
        
        runner = BenchmarkRunner(
            tokenizer=tokenizer,
            model=model,
            model_name=model_name,
            model_type="DeepSeek R1 70B",
            use_quantization=True,
        )
        
        result = runner.run_benchmark(benchmark_data)
        return result.to_dict()
    
    except Exception as error:
        print(f"❌ DeepSeek 테스트 실패: {error}")
        import traceback
        traceback.print_exc()
        return None


def print_summary(all_results: dict) -> None:
    """
    벤치마크 실행 결과 요약을 출력합니다.
    
    Args:
        all_results: 모든 모델의 벤치마크 결과 딕셔너리
    """
    print("\n" + "=" * 80)
    print("벤치마크 요약")
    print("=" * 80)
    
    for model_key, result in all_results.items():
        if result is None:
            continue
        
        print(f"\n{result.get('model_type', 'N/A')}:")
        print(
            f"  정확도: {result.get('accuracy', 0):.2f}% "
            f"({result.get('correct_answers', 0)}/{result.get('total_questions', 0)})"
        )
        print(
            f"  평균 생성 시간: {result.get('avg_generation_time', 0):.2f}초"
        )
        print(
            f"  총 소요 시간: {result.get('total_time', 0)/60:.1f}분"
        )
        total_memory = sum(result.get("memory_usage", {}).values())
        print(f"  총 메모리 사용량: {total_memory:.2f} GB")


def main() -> None:
    """메인 실행 함수"""
    import torch
    
    benchmark_path = DEFAULT_BENCHMARK_PATHS["kmle_2023"]
    
    print("=" * 80)
    print("KMLE 2023 벤치마크 실행")
    print("=" * 80)
    
    # 벤치마크 데이터 로드
    print(f"\n벤치마크 데이터 로드: {benchmark_path}")
    data_loader = KMLEBenchmarkDataLoader()
    benchmark_data = data_loader.load(benchmark_path)
    print(f"총 {len(benchmark_data)}개의 문제")
    
    all_results = {}
    
    # TinyLlama 벤치마크
    tinyllama_result = load_and_run_tinyllama_benchmark(benchmark_data)
    if tinyllama_result is not None:
        all_results["tinyllama"] = tinyllama_result
    
    # DeepSeek R1 70B 벤치마크
    deepseek_result = load_and_run_deepseek_benchmark(benchmark_data)
    if deepseek_result is not None:
        all_results["deepseek"] = deepseek_result
    
    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = RESULTS_DIRECTORY / f"kmle_2023_benchmark_{timestamp}.json"
    
    with open(results_file, "w", encoding="utf-8") as file:
        json.dump(all_results, file, indent=2, ensure_ascii=False)
    
    print(f"\n결과 저장: {results_file}")
    
    # 요약 출력
    print_summary(all_results)


if __name__ == "__main__":
    main()

