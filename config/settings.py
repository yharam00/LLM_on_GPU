"""
LLM 벤치마크 시스템의 중앙화된 설정 관리 모듈

이 모듈은 모든 경로, 기본값, 설정 상수들을 중앙에서 관리하여
코드 전반에 걸친 하드코딩된 경로와 설정값의 중복을 제거합니다.

이 모듈은 refactored 폴더 내부에서 실행될 때를 우선 고려하며,
필요한 경우 상위 디렉토리의 기존 파일들도 접근할 수 있습니다.

Attributes:
    REFACTORED_DIRECTORY: refactored 폴더의 경로
    BASE_DIRECTORY: 프로젝트의 루트 디렉토리 경로 (상위 디렉토리)
    BENCHMARKS_DIRECTORY: 벤치마크 데이터 파일들이 저장된 디렉토리
    MODELS_DIRECTORY: 다운로드된 모델들이 저장된 디렉토리
    RESULTS_DIRECTORY: 벤치마크 실행 결과와 리포트가 저장되는 디렉토리
    LOGS_DIRECTORY: 로그 파일들이 저장되는 디렉토리
    DOCS_DIRECTORY: 문서 파일들이 저장되는 디렉토리
    DEFAULT_BENCHMARK_PATHS: 각 벤치마크 타입별 기본 파일 경로
    DEFAULT_GENERATION_PARAMETERS: 텍스트 생성 시 사용되는 기본 하이퍼파라미터
    GPU_MONITORING_INTERVAL_SECONDS: GPU 모니터링 시 데이터 수집 간격(초)
"""
from pathlib import Path
from typing import Dict, Any

# refactored 폴더 경로 (이 파일이 있는 위치의 부모 디렉토리)
REFACTORED_DIRECTORY: Path = Path(__file__).parent.parent.resolve()

# 프로젝트 루트 디렉토리 경로 (refactored의 상위 디렉토리)
BASE_DIRECTORY: Path = REFACTORED_DIRECTORY.parent

# refactored 내부 디렉토리들 (우선 사용)
REFACTORED_BENCHMARKS_DIRECTORY: Path = REFACTORED_DIRECTORY / "benchmarks"
REFACTORED_RESULTS_DIRECTORY: Path = REFACTORED_DIRECTORY / "results"
REFACTORED_LOGS_DIRECTORY: Path = REFACTORED_DIRECTORY / "logs"
REFACTORED_DOCS_DIRECTORY: Path = REFACTORED_DIRECTORY / "docs"
REFACTORED_MODELS_DIRECTORY: Path = REFACTORED_DIRECTORY / "models"

# 상위 디렉토리의 기존 디렉토리들 (fallback)
LEGACY_BENCHMARKS_DIRECTORY: Path = BASE_DIRECTORY / "benchmarks"
LEGACY_RESULTS_DIRECTORY: Path = BASE_DIRECTORY / "results"
LEGACY_MODELS_DIRECTORY: Path = BASE_DIRECTORY / "models"

# 실제 사용할 디렉토리 (refactored 내부가 존재하면 우선, 없으면 상위 디렉토리 사용)
def _get_directory(refactored_path: Path, legacy_path: Path) -> Path:
    """
    refactored 내부 디렉토리가 존재하면 사용하고, 없으면 상위 디렉토리 사용
    
    Args:
        refactored_path: refactored 내부 경로
        legacy_path: 상위 디렉토리 경로
    
    Returns:
        Path: 사용할 디렉토리 경로
    """
    if refactored_path.exists() and any(refactored_path.iterdir()):
        return refactored_path
    return legacy_path

BENCHMARKS_DIRECTORY: Path = _get_directory(
    REFACTORED_BENCHMARKS_DIRECTORY, LEGACY_BENCHMARKS_DIRECTORY
)
RESULTS_DIRECTORY: Path = _get_directory(
    REFACTORED_RESULTS_DIRECTORY, LEGACY_RESULTS_DIRECTORY
)
MODELS_DIRECTORY: Path = _get_directory(
    REFACTORED_MODELS_DIRECTORY, LEGACY_MODELS_DIRECTORY
)
LOGS_DIRECTORY: Path = REFACTORED_LOGS_DIRECTORY
DOCS_DIRECTORY: Path = REFACTORED_DOCS_DIRECTORY

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
LOGS_DIRECTORY.mkdir(parents=True, exist_ok=True)
DOCS_DIRECTORY.mkdir(parents=True, exist_ok=True)
REFACTORED_BENCHMARKS_DIRECTORY.mkdir(parents=True, exist_ok=True)
REFACTORED_RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)

