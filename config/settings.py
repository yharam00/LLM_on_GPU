"""
LLM 벤치마크 시스템의 중앙화된 설정 관리 모듈

이 모듈은 모든 경로, 기본값, 설정 상수들을 중앙에서 관리하여
코드 전반에 걸친 하드코딩된 경로와 설정값의 중복을 제거합니다.

Attributes:
    BASE_DIRECTORY: 프로젝트의 루트 디렉토리 경로
    BENCHMARKS_DIRECTORY: 벤치마크 데이터 파일들이 저장된 디렉토리
    MODELS_DIRECTORY: 다운로드된 모델들이 저장된 디렉토리
    RESULTS_DIRECTORY: 벤치마크 실행 결과와 리포트가 저장되는 디렉토리
    DEFAULT_BENCHMARK_PATHS: 각 벤치마크 타입별 기본 파일 경로
    DEFAULT_GENERATION_PARAMETERS: 텍스트 생성 시 사용되는 기본 하이퍼파라미터
    GPU_MONITORING_INTERVAL_SECONDS: GPU 모니터링 시 데이터 수집 간격(초)
"""
from pathlib import Path
from typing import Dict, Any

# 프로젝트 루트 디렉토리 경로
BASE_DIRECTORY: Path = Path("/home/ychanmo/llm_testing")

# 주요 디렉토리 경로들
BENCHMARKS_DIRECTORY: Path = BASE_DIRECTORY / "benchmarks"
MODELS_DIRECTORY: Path = BASE_DIRECTORY / "models"
RESULTS_DIRECTORY: Path = BASE_DIRECTORY / "results"

# 벤치마크 데이터 파일 경로
DEFAULT_BENCHMARK_PATHS: Dict[str, Path] = {
    "kmle_2023": BENCHMARKS_DIRECTORY / "kmle_2023.jsonl",
    "custom_qa": BENCHMARKS_DIRECTORY / "custom_qa.json",
}

# 텍스트 생성 시 사용되는 기본 하이퍼파라미터
DEFAULT_GENERATION_PARAMETERS: Dict[str, Any] = {
    "max_new_tokens": 150,
    "num_return_sequences": 1,
    "temperature": 0.3,
    "do_sample": True,
    "top_p": 0.9,
}

# GPU 모니터링 설정
GPU_MONITORING_INTERVAL_SECONDS: float = 0.5
GPU_MONITORING_DEFAULT_DURATION_SECONDS: int = 60

# 결과 파일명 패턴
RESULTS_FILE_PATTERN_PREFIX: str = "benchmark_results"
KMLE_RESULTS_FILE_PATTERN_PREFIX: str = "kmle_2023_benchmark"
REPORT_FILE_PATTERN_PREFIX: str = "KMLE_2023_BENCHMARK_REPORT"
GPU_MONITOR_FILE_PATTERN_PREFIX: str = "gpu_monitor"

# 디렉토리 생성 보장
RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
BENCHMARKS_DIRECTORY.mkdir(parents=True, exist_ok=True)
MODELS_DIRECTORY.mkdir(parents=True, exist_ok=True)

