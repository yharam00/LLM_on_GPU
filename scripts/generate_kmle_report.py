#!/usr/bin/env python3
"""
KMLE 2023 벤치마크 결과 리포트 생성 스크립트

이 스크립트는 실행된 KMLE 벤치마크 결과를 분석하여
마크다운 형식의 상세한 리포트를 생성합니다.

사용법:
    python generate_kmle_report.py [결과_파일_경로]

기능:
- 최신 벤치마크 결과 파일 자동 탐지
- 마크다운 리포트 생성
- 카테고리별 성능 분석
- 모델 간 비교 분석
"""
import json
import sys
import glob
from pathlib import Path
from typing import Optional

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core import ReportGenerator
from config.settings import RESULTS_DIRECTORY


def find_latest_benchmark_file() -> Optional[Path]:
    """
    가장 최신의 KMLE 벤치마크 결과 파일을 찾습니다.
    
    Returns:
        Path | None: 최신 결과 파일 경로, 없으면 None
    """
    pattern = str(RESULTS_DIRECTORY / "kmle_2023_benchmark_*.json")
    files = glob.glob(pattern)
    
    if not files:
        return None
    
    return Path(max(files, key=lambda f: Path(f).stat().st_mtime))


def main() -> None:
    """메인 실행 함수"""
    # 결과 파일 경로 결정
    if len(sys.argv) > 1:
        results_file = Path(sys.argv[1])
    else:
        results_file = find_latest_benchmark_file()
    
    if results_file is None or not results_file.exists():
        print("벤치마크 결과 파일을 찾을 수 없습니다.")
        print("먼저 run_kmle_benchmark.py를 실행해주세요.")
        sys.exit(1)
    
    print(f"벤치마크 결과 파일: {results_file}")
    
    # 결과 데이터 로드
    with open(results_file, "r", encoding="utf-8") as file:
        results_data = json.load(file)
    
    # 리포트 생성
    print("\n📝 리포트 생성 중...")
    report_generator = ReportGenerator()
    report = report_generator.generate_kmle_report(results_data)
    
    print(f"리포트 생성 완료: {report.report_path}")
    print(f"리포트 저장: {report.report_path}")


if __name__ == "__main__":
    main()

