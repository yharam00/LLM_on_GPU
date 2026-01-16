#!/usr/bin/env python3
"""
단일 모델 벤치마크 실행 스크립트

이 스크립트는 커맨드라인 파라미터로 지정한 모델 하나만 벤치마크를 실행합니다.

사용법:
    # 기본 사용 (양자화 없이)
    python run_single_model_benchmark.py --model-name openai/gpt-oss-20b
    
    # 4-bit 양자화 사용
    python run_single_model_benchmark.py --model-name openai/gpt-oss-20b --use-quantization
    
    # 커스텀 QA 벤치마크 사용
    python run_single_model_benchmark.py --model-name openai/gpt-oss-20b --benchmark-type custom_qa
    
    # 모델 타입 이름 지정
    python run_single_model_benchmark.py --model-name openai/gpt-oss-20b --model-type "GPT OSS 20B"
    
    # Gated 모델 사용 (토큰 필요)
    python run_single_model_benchmark.py --model-name google/medgemma-27b-text-it --use-quantization --token YOUR_HF_TOKEN
    # 또는 환경 변수 사용: export HF_TOKEN=YOUR_HF_TOKEN

예시:
    python run_single_model_benchmark.py --model-name openai/gpt-oss-20b --use-quantization
    python run_single_model_benchmark.py --model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0
    python run_single_model_benchmark.py --model-name deepseek-ai/DeepSeek-R1-Distill-Llama-70B --use-quantization
    python run_single_model_benchmark.py --model-name google/medgemma-27b-text-it --use-quantization --token YOUR_HF_TOKEN
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
    KMLEBenchmarkDataLoader,
    CustomQABenchmarkDataLoader,
    BenchmarkRunner,
)
from config.settings import (
    DEFAULT_BENCHMARK_PATHS,
    RESULTS_DIRECTORY,
    MODELS_DIRECTORY,
)


def extract_model_type_from_name(model_name: str) -> str:
    """
    모델 이름에서 모델 타입을 추출합니다.
    
    Args:
        model_name: Hugging Face 모델 이름 (예: "openai/gpt-oss-20b")
    
    Returns:
        str: 추출된 모델 타입 이름
    """
    # 모델 이름에서 마지막 부분 추출
    parts = model_name.split("/")
    model_id = parts[-1] if len(parts) > 1 else model_name
    
    # 일반적인 패턴 처리
    if "gpt-oss-20b" in model_id.lower():
        return "GPT OSS 20B"
    elif "gpt-oss-120b" in model_id.lower():
        return "GPT OSS 120B"
    elif "medgemma" in model_id.lower():
        # medgemma 모델 타입 추출 (예: medgemma-27b-text-it -> MedGemma 27B)
        if "27b" in model_id.lower():
            return "MedGemma 27B"
        elif "7b" in model_id.lower():
            return "MedGemma 7B"
        else:
            return "MedGemma"
    elif "tinyllama" in model_id.lower():
        return "TinyLlama"
    elif "deepseek" in model_id.lower():
        return "DeepSeek R1 70B"
    else:
        # 기본값: 모델 ID를 그대로 사용
        return model_id


def run_single_model_benchmark(
    model_name: str,
    benchmark_type: str = "kmle_2023",
    use_quantization: bool = False,
    model_type: str = None,
    trust_remote_code: bool = True,
    token: Optional[str] = None,
) -> dict:
    """
    단일 모델에 대한 벤치마크를 실행합니다.
    
    Args:
        model_name: Hugging Face 모델 이름 또는 로컬 경로
        benchmark_type: 벤치마크 타입 ("kmle_2023" 또는 "custom_qa")
        use_quantization: 4-bit 양자화 사용 여부
        model_type: 모델 타입 이름 (None이면 자동 추출)
        trust_remote_code: 원격 코드 실행 허용 여부
        token: Hugging Face 토큰 (gated 모델 접근용, None이면 환경 변수 사용)
    
    Returns:
        dict: 벤치마크 실행 결과 딕셔너리
    """
    print("=" * 80)
    print(f"벤치마크 실행: {model_name}")
    print("=" * 80)
    
    # 모델 타입 자동 추출
    if model_type is None:
        model_type = extract_model_type_from_name(model_name)
    
    # 벤치마크 데이터 로드
    if benchmark_type not in DEFAULT_BENCHMARK_PATHS:
        raise ValueError(
            f"지원하지 않는 벤치마크 타입: {benchmark_type}. "
            f"지원 타입: {list(DEFAULT_BENCHMARK_PATHS.keys())}"
        )
    
    benchmark_path = DEFAULT_BENCHMARK_PATHS[benchmark_type]
    print(f"\n벤치마크 데이터 로드: {benchmark_path}")
    
    if benchmark_type == "kmle_2023":
        data_loader = KMLEBenchmarkDataLoader()
    elif benchmark_type == "custom_qa":
        data_loader = CustomQABenchmarkDataLoader()
    else:
        raise ValueError(f"알 수 없는 벤치마크 타입: {benchmark_type}")
    
    benchmark_data = data_loader.load(benchmark_path)
    print(f"총 {len(benchmark_data)}개의 문제")
    
    try:
        print(f"\n모델 로드: {model_name}")
        print(f"모델 타입: {model_type}")
        
        # GPT OSS 모델은 이미 MXFP4로 양자화되어 있으므로 양자화 설정을 전달하지 않음
        is_gpt_oss_model = "gpt-oss" in model_name.lower()
        # MedGemma 모델은 양자화와 호환성 문제가 있을 수 있으므로 양자화 비활성화
        is_medgemma_model = "medgemma" in model_name.lower()
        
        # 양자화 설정
        quantization_config = None
        if use_quantization and not is_gpt_oss_model and not is_medgemma_model:
            print("4-bit 양자화 사용 중...")
            quantization_config = QuantizationConfiguration(
                load_in_4bit=True,
                compute_dtype=torch.bfloat16,
                use_double_quant=True,
                quant_type="nf4",
            )
        elif is_gpt_oss_model:
            print("GPT OSS 모델 감지: 이미 MXFP4 양자화되어 있음 (추가 양자화 설정 없이 로드)")
        elif is_medgemma_model and use_quantization:
            print("⚠️  MedGemma 모델 감지: 양자화 호환성 문제로 양자화를 비활성화합니다.")
            print("   (MedGemma 모델은 양자화 없이 실행됩니다)")
        else:
            print("양자화 미사용")
        
        # 모델 로드
        model_loader = ModelLoader(
            model_name=model_name,
            quantization_config=quantization_config,
            dtype=torch.bfloat16,
            trust_remote_code=trust_remote_code,
            token=token,
            cache_dir=MODELS_DIRECTORY,
        )
        tokenizer, model = model_loader.load()
        
        # 벤치마크 실행
        # GPT OSS 모델은 이미 양자화되어 있으므로 use_quantization 플래그 조정
        # MedGemma 모델은 양자화를 사용하지 않으므로 플래그 조정
        actual_use_quantization = (use_quantization or is_gpt_oss_model) and not is_medgemma_model
        
        runner = BenchmarkRunner(
            tokenizer=tokenizer,
            model=model,
            model_name=model_name,
            model_type=model_type,
            use_quantization=actual_use_quantization,
        )
        
        print("\n벤치마크 실행 중...")
        result = runner.run_benchmark(benchmark_data)
        result_dict = result.to_dict()
        
        # 메모리 정리
        model_loader.cleanup()
        
        # 결과 출력
        print("\n" + "=" * 80)
        print("벤치마크 결과")
        print("=" * 80)
        print(f"모델: {model_type}")
        print(
            f"정확도: {result_dict.get('accuracy', 0):.2f}% "
            f"({result_dict.get('correct_answers', 0)}/{result_dict.get('total_questions', 0)})"
        )
        print(
            f"평균 생성 시간: {result_dict.get('avg_generation_time', 0):.2f}초"
        )
        print(
            f"총 소요 시간: {result_dict.get('total_time', 0)/60:.1f}분"
        )
        total_memory = sum(result_dict.get("memory_usage", {}).values())
        print(f"총 메모리 사용량: {total_memory:.2f} GB")
        
        return result_dict
    
    except Exception as error:
        print(f"❌ 벤치마크 실행 실패: {error}")
        import traceback
        traceback.print_exc()
        return None


def main() -> None:
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(
        description="단일 모델 벤치마크 실행 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # GPT OSS 20B 모델 (양자화 사용)
  python run_single_model_benchmark.py --model-name openai/gpt-oss-20b --use-quantization
  
  # TinyLlama 모델 (양자화 없이)
  python run_single_model_benchmark.py --model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0
  
  # MedGemma 27B 모델 (양자화 사용, gated 모델이므로 토큰 필요)
  python run_single_model_benchmark.py --model-name google/medgemma-27b-text-it --use-quantization --token YOUR_HF_TOKEN
  
  # 커스텀 QA 벤치마크 사용
  python run_single_model_benchmark.py --model-name openai/gpt-oss-20b --benchmark-type custom_qa
        """,
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Hugging Face 모델 이름 또는 로컬 경로 (예: openai/gpt-oss-20b, google/medgemma-27b-text-it)",
    )
    
    parser.add_argument(
        "--benchmark-type",
        type=str,
        default="kmle_2023",
        choices=["kmle_2023", "custom_qa"],
        help="벤치마크 타입 (기본값: kmle_2023)",
    )
    
    parser.add_argument(
        "--use-quantization",
        action="store_true",
        help="4-bit 양자화 사용 (대규모 모델에 권장)",
    )
    
    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        help="모델 타입 이름 (지정하지 않으면 모델 이름에서 자동 추출)",
    )
    
    parser.add_argument(
        "--no-trust-remote-code",
        action="store_true",
        help="원격 코드 실행 비활성화",
    )
    
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face 토큰 (gated 모델 접근용, 지정하지 않으면 HF_TOKEN 환경 변수 사용)",
    )
    
    args = parser.parse_args()
    
    # 벤치마크 실행
    result = run_single_model_benchmark(
        model_name=args.model_name,
        benchmark_type=args.benchmark_type,
        use_quantization=args.use_quantization,
        model_type=args.model_type,
        trust_remote_code=not args.no_trust_remote_code,
        token=args.token,
    )
    
    if result is None:
        print("\n❌ 벤치마크 실행 실패")
        sys.exit(1)
    
    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_safe_name = args.model_name.replace("/", "_").replace("\\", "_")
    results_file = RESULTS_DIRECTORY / f"{args.benchmark_type}_benchmark_{model_safe_name}_{timestamp}.json"
    
    # 단일 모델 결과를 딕셔너리로 래핑
    output_data = {
        args.model_name: result
    }
    
    with open(results_file, "w", encoding="utf-8") as file:
        json.dump(output_data, file, indent=2, ensure_ascii=False)
    
    print(f"\n결과 저장: {results_file}")
    print("\n✅ 벤치마크 완료!")


if __name__ == "__main__":
    main()

