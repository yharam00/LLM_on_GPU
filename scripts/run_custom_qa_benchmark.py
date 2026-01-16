#!/usr/bin/env python3
"""
커스텀 QA 벤치마크 실행 스크립트

이 스크립트는 custom_qa.json 벤치마크 데이터를 사용하여
모델의 성능을 평가합니다.

사용법:
    python run_custom_qa_benchmark.py [--model_name MODEL_NAME] [--use_quantization]

옵션:
    --model_name: 사용할 모델 이름 (기본값: TinyLlama/TinyLlama-1.1B-Chat-v1.0)
    --use_quantization: 4-bit 양자화 사용 여부

기능:
- 지정된 모델로 커스텀 QA 벤치마크 실행
- 결과를 JSON 파일로 저장
- 실행 요약 정보 출력
"""
import json
import sys
import argparse
import torch
from pathlib import Path
from datetime import datetime
from typing import Optional

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core import (
    ModelLoader,
    QuantizationConfiguration,
    CustomQABenchmarkDataLoader,
    BenchmarkRunner,
)
from config.settings import (
    DEFAULT_BENCHMARK_PATHS,
    RESULTS_DIRECTORY,
    MODELS_DIRECTORY,
)


def load_and_run_benchmark(
    model_name: str,
    benchmark_data: list,
    use_quantization: bool = False,
    model_type: Optional[str] = None,
) -> Optional[dict]:
    """
    모델을 로드하고 벤치마크를 실행합니다.
    
    Args:
        model_name: Hugging Face 모델 이름 또는 로컬 경로
        benchmark_data: 표준화된 벤치마크 데이터 리스트
        use_quantization: 4-bit 양자화 사용 여부
        model_type: 모델 타입 설명 (None이면 model_name에서 추출)
    
    Returns:
        dict: 벤치마크 실행 결과 딕셔너리, 실패 시 None
    """
    if model_type is None:
        model_type = model_name.split("/")[-1] if "/" in model_name else model_name
    
    print("\n" + "=" * 80)
    print(f"{model_type} 벤치마크 실행")
    print("=" * 80)
    
    try:
        print(f"\n모델 로드: {model_name}")
        
        # 양자화 설정
        quantization_config = None
        if use_quantization:
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
        
        # 벤치마크 실행
        runner = BenchmarkRunner(
            tokenizer=tokenizer,
            model=model,
            model_name=model_name,
            model_type=model_type,
            use_quantization=use_quantization,
        )
        
        result = runner.run_benchmark(benchmark_data)
        result_dict = result.to_dict()
        
        # 메모리 정리
        model_loader.cleanup()
        
        return result_dict
    
    except Exception as error:
        print(f"❌ 벤치마크 실행 실패: {error}")
        import traceback
        traceback.print_exc()
        return None


def print_summary(result: dict) -> None:
    """
    벤치마크 실행 결과 요약을 출력합니다.
    
    Args:
        result: 벤치마크 결과 딕셔너리
    """
    if result is None:
        return
    
    print("\n" + "=" * 80)
    print("벤치마크 요약")
    print("=" * 80)
    
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
    
    # 카테고리별 정확도
    individual_results = result.get("results", [])
    if individual_results:
        from collections import defaultdict
        category_stats = defaultdict(lambda: {"correct": 0, "total": 0})
        
        for item in individual_results:
            category = item.get("category", "unknown")
            category_stats[category]["total"] += 1
            if item.get("is_correct", False):
                category_stats[category]["correct"] += 1
        
        print("\n  카테고리별 정확도:")
        for category, stats in sorted(category_stats.items()):
            accuracy = (
                (stats["correct"] / stats["total"]) * 100.0
                if stats["total"] > 0
                else 0.0
            )
            print(
                f"    {category}: {accuracy:.1f}% "
                f"({stats['correct']}/{stats['total']})"
            )


def main() -> None:
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(
        description="커스텀 QA 벤치마크 실행",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="사용할 모델 이름 (기본값: TinyLlama/TinyLlama-1.1B-Chat-v1.0)",
    )
    parser.add_argument(
        "--use_quantization",
        action="store_true",
        help="4-bit 양자화 사용 (메모리 절약)",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="모델 타입 설명 (기본값: 모델 이름에서 자동 추출)",
    )
    parser.add_argument(
        "--benchmark_path",
        type=str,
        default=None,
        help="벤치마크 파일 경로 (기본값: config/settings.py의 기본값 사용)",
    )
    
    args = parser.parse_args()
    
    # 벤치마크 파일 경로 결정
    if args.benchmark_path:
        benchmark_path = Path(args.benchmark_path)
    else:
        benchmark_path = DEFAULT_BENCHMARK_PATHS["custom_qa"]
    
    print("=" * 80)
    print("커스텀 QA 벤치마크 실행")
    print("=" * 80)
    
    # 벤치마크 데이터 로드
    print(f"\n벤치마크 데이터 로드: {benchmark_path}")
    data_loader = CustomQABenchmarkDataLoader()
    
    try:
        benchmark_data = data_loader.load(benchmark_path)
        print(f"총 {len(benchmark_data)}개의 질문")
    except FileNotFoundError as error:
        print(f"❌ 오류: {error}")
        sys.exit(1)
    except Exception as error:
        print(f"❌ 데이터 로드 실패: {error}")
        sys.exit(1)
    
    # 벤치마크 실행
    result = load_and_run_benchmark(
        model_name=args.model_name,
        benchmark_data=benchmark_data,
        use_quantization=args.use_quantization,
        model_type=args.model_type,
    )
    
    if result is None:
        print("\n❌ 벤치마크 실행 실패")
        sys.exit(1)
    
    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_safe = args.model_name.replace("/", "_").replace("\\", "_")
    results_file = (
        RESULTS_DIRECTORY
        / f"custom_qa_benchmark_{model_name_safe}_{timestamp}.json"
    )
    
    with open(results_file, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
    
    print(f"\n결과 저장: {results_file}")
    
    # 요약 출력
    print_summary(result)


if __name__ == "__main__":
    main()

