"""
벤치마크 결과 리포트 생성을 담당하는 모듈

이 모듈은 벤치마크 실행 결과를 분석하여
마크다운 형식의 상세한 리포트를 생성합니다.

주요 기능:
- 벤치마크 결과 데이터 분석
- 마크다운 리포트 생성
- 카테고리별 성능 분석
- 모델 간 비교 분석
"""
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass

from .benchmark_runner import BenchmarkExecutionResult

# config 모듈 import (상대/절대 import 모두 지원)
try:
    from ..config.settings import RESULTS_DIRECTORY, DOCS_DIRECTORY
except ImportError:
    from config.settings import RESULTS_DIRECTORY, DOCS_DIRECTORY


@dataclass
class BenchmarkReport:
    """
    생성된 벤치마크 리포트 정보를 담는 데이터 클래스
    
    Attributes:
        report_path: 리포트 파일 경로
        report_content: 리포트 내용 (마크다운)
        generation_timestamp: 리포트 생성 시각
    """
    report_path: Path
    report_content: str
    generation_timestamp: datetime


class ReportGenerator:
    """
    벤치마크 결과 리포트를 생성하는 클래스
    
    이 클래스는 벤치마크 실행 결과를 받아서
    다양한 형식의 리포트를 생성할 수 있습니다.
    """
    
    def __init__(
        self,
        output_directory: Optional[Path] = None,
        docs_directory: Optional[Path] = None,
    ) -> None:
        """
        ReportGenerator 인스턴스 초기화
        
        Args:
            output_directory: 결과 JSON 저장 디렉토리 (None이면 기본값 사용)
            docs_directory: 리포트 마크다운 저장 디렉토리 (None이면 기본값 사용)
        """
        self.output_directory = output_directory or RESULTS_DIRECTORY
        self.docs_directory = docs_directory or DOCS_DIRECTORY
    
    def generate_kmle_report(
        self, results_data: Dict[str, Any]
    ) -> BenchmarkReport:
        """
        KMLE 벤치마크 결과 리포트를 생성합니다.
        
        Args:
            results_data: 벤치마크 결과 데이터 딕셔너리
                (예: {"tinyllama": BenchmarkExecutionResult, ...})
        
        Returns:
            BenchmarkReport: 생성된 리포트 객체
        """
        timestamp = datetime.now()
        report_lines: List[str] = []
        
        # 헤더
        report_lines.extend([
            "# KMLE 2023 벤치마크 비교 리포트\n",
            f"**생성 일시**: {timestamp.strftime('%Y년 %m월 %d일 %H:%M:%S')}\n",
            "**벤치마크 데이터**: kmle_2023.jsonl (314개 문제)\n",
            "---\n",
        ])
        
        # 전체 성능 비교 테이블
        report_lines.extend(self._generate_summary_table(results_data))
        
        # 상세 성능 분석
        report_lines.extend(self._generate_detailed_analysis(results_data))
        
        # 성능 비교 분석
        if len(results_data) >= 2:
            report_lines.extend(
                self._generate_comparison_analysis(results_data)
            )
        
        # 결론
        report_lines.extend(self._generate_conclusion(results_data))
        
        # 푸터
        report_lines.append("\n---\n")
        report_lines.append(
            f"**리포트 생성 시간**: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        
        # 리포트 저장
        report_content = "".join(report_lines)
        report_filename = (
            f"KMLE_2023_BENCHMARK_REPORT_{timestamp.strftime('%Y%m%d_%H%M%S')}.md"
        )
        # 마크다운 리포트는 docs 디렉토리에 저장
        report_path = self.docs_directory / report_filename
        
        with open(report_path, "w", encoding="utf-8") as file:
            file.write(report_content)
        
        return BenchmarkReport(
            report_path=report_path,
            report_content=report_content,
            generation_timestamp=timestamp,
        )
    
    def _generate_summary_table(
        self, results_data: Dict[str, Any]
    ) -> List[str]:
        """
        전체 성능 비교 테이블을 생성합니다.
        
        Args:
            results_data: 벤치마크 결과 데이터
        
        Returns:
            List[str]: 마크다운 테이블 라인 리스트
        """
        lines = [
            "## 📊 전체 성능 비교\n",
            "| 모델 | 정확도 | 평균 생성 시간 | 총 소요 시간 | 총 메모리 사용량 |\n",
            "|------|--------|---------------|-------------|-----------------|\n",
        ]
        
        for model_key in ["tinyllama", "deepseek"]:
            if model_key in results_data:
                result = results_data[model_key]
                total_memory = sum(result.get("memory_usage", {}).values())
                lines.append(
                    f"| {result.get('model_type', 'N/A')} | "
                    f"{result.get('accuracy', 0):.2f}% "
                    f"({result.get('correct_answers', 0)}/{result.get('total_questions', 0)}) | "
                    f"{result.get('avg_generation_time', 0):.2f}초 | "
                    f"{result.get('total_time', 0)/60:.1f}분 | "
                    f"{total_memory:.2f} GB |\n"
                )
        
        lines.append("\n---\n")
        return lines
    
    def _generate_detailed_analysis(
        self, results_data: Dict[str, Any]
    ) -> List[str]:
        """
        상세 성능 분석 섹션을 생성합니다.
        
        Args:
            results_data: 벤치마크 결과 데이터
        
        Returns:
            List[str]: 마크다운 라인 리스트
        """
        lines = ["## 🔍 상세 성능 분석\n"]
        
        for model_key in ["tinyllama", "deepseek"]:
            if model_key not in results_data:
                continue
            
            result = results_data[model_key]
            lines.extend([
                f"### {result.get('model_type', 'N/A')}\n\n",
                f"- **모델 이름**: {result.get('model_name', 'N/A')}\n",
                f"- **양자화 사용**: "
                f"{'예 (4-bit)' if result.get('use_quantization', False) else '아니오'}\n",
                f"- **전체 정확도**: "
                f"{result.get('accuracy', 0):.2f}% "
                f"({result.get('correct_answers', 0)}/{result.get('total_questions', 0)})\n",
                f"- **평균 생성 시간**: "
                f"{result.get('avg_generation_time', 0):.2f}초\n",
                f"- **총 소요 시간**: "
                f"{result.get('total_time', 0)/60:.1f}분 "
                f"({result.get('total_time', 0):.0f}초)\n",
            ])
            
            # 메모리 사용량
            lines.append("- **GPU 메모리 사용량**:\n")
            total_memory = 0.0
            for gpu_idx, memory in sorted(
                result.get("memory_usage", {}).items()
            ):
                lines.append(f"  - GPU {gpu_idx}: {memory:.2f} GB\n")
                total_memory += memory
            lines.append(f"  - **총합**: {total_memory:.2f} GB\n")
            
            # 카테고리별 정확도
            category_stats = self._calculate_category_statistics(
                result.get("results", [])
            )
            if category_stats:
                lines.extend([
                    "- **카테고리별 정확도**:\n",
                    "  | 카테고리 | 정확도 | 정답/전체 |\n",
                    "  |---------|--------|----------|\n",
                ])
                for category, stats in sorted(category_stats.items()):
                    accuracy = (
                        (stats["correct"] / stats["total"]) * 100.0
                        if stats["total"] > 0
                        else 0.0
                    )
                    lines.append(
                        f"  | {category} | {accuracy:.1f}% | "
                        f"{stats['correct']}/{stats['total']} |\n"
                    )
            
            lines.append("\n")
        
        return lines
    
    def _calculate_category_statistics(
        self, individual_results: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, int]]:
        """
        카테고리별 통계를 계산합니다.
        
        Args:
            individual_results: 개별 결과 리스트
        
        Returns:
            Dict[str, Dict[str, int]]: 카테고리별 통계
        """
        category_stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"correct": 0, "total": 0}
        )
        
        for result in individual_results:
            category = result.get("category", "N/A")
            category_stats[category]["total"] += 1
            if result.get("is_correct", False):
                category_stats[category]["correct"] += 1
        
        return dict(category_stats)
    
    def _generate_comparison_analysis(
        self, results_data: Dict[str, Any]
    ) -> List[str]:
        """
        모델 간 비교 분석 섹션을 생성합니다.
        
        Args:
            results_data: 벤치마크 결과 데이터
        
        Returns:
            List[str]: 마크다운 라인 리스트
        """
        if "tinyllama" not in results_data or "deepseek" not in results_data:
            return []
        
        tiny = results_data["tinyllama"]
        deepseek = results_data["deepseek"]
        
        lines = ["## 📈 성능 비교 분석\n"]
        
        # 정확도 비교
        acc_diff = deepseek.get("accuracy", 0) - tiny.get("accuracy", 0)
        lines.extend([
            "### 정확도\n",
            f"- **DeepSeek R1 70B**: {deepseek.get('accuracy', 0):.2f}%\n",
            f"- **TinyLlama**: {tiny.get('accuracy', 0):.2f}%\n",
            f"- **차이**: {acc_diff:+.2f}%p\n\n",
        ])
        
        # 속도 비교
        tiny_time = tiny.get("avg_generation_time", 0)
        deepseek_time = deepseek.get("avg_generation_time", 0)
        speed_ratio = (
            tiny_time / deepseek_time if deepseek_time > 0 else 0.0
        )
        lines.extend([
            "### 생성 속도\n",
            f"- **DeepSeek R1 70B**: {deepseek_time:.2f}초/문제\n",
            f"- **TinyLlama**: {tiny_time:.2f}초/문제\n",
        ])
        if speed_ratio > 0:
            lines.append(f"- **TinyLlama가 {speed_ratio:.1f}배 빠름**\n\n")
        
        # 메모리 비교
        tiny_memory = sum(tiny.get("memory_usage", {}).values())
        deepseek_memory = sum(deepseek.get("memory_usage", {}).values())
        memory_ratio = (
            deepseek_memory / tiny_memory if tiny_memory > 0 else 0.0
        )
        lines.extend([
            "### 메모리 사용량\n",
            f"- **DeepSeek R1 70B**: {deepseek_memory:.2f} GB\n",
            f"- **TinyLlama**: {tiny_memory:.2f} GB\n",
        ])
        if memory_ratio > 0:
            lines.append(
                f"- **DeepSeek이 {memory_ratio:.1f}배 더 많은 메모리 사용**\n\n"
            )
        
        return lines
    
    def _generate_conclusion(
        self, results_data: Dict[str, Any]
    ) -> List[str]:
        """
        결론 섹션을 생성합니다.
        
        Args:
            results_data: 벤치마크 결과 데이터
        
        Returns:
            List[str]: 마크다운 라인 리스트
        """
        lines = ["## 💡 결론\n"]
        
        if "tinyllama" in results_data and "deepseek" in results_data:
            tiny = results_data["tinyllama"]
            deepseek = results_data["deepseek"]
            
            lines.append("### 주요 발견사항\n\n")
            
            # 정확도 비교
            acc_diff = deepseek.get("accuracy", 0) - tiny.get("accuracy", 0)
            if acc_diff > 0:
                lines.append(
                    f"1. **정확도**: DeepSeek R1 70B가 TinyLlama보다 "
                    f"{acc_diff:.2f}%p 높은 정확도를 보였습니다.\n"
                )
            else:
                lines.append(
                    f"1. **정확도**: TinyLlama가 DeepSeek R1 70B보다 "
                    f"{abs(acc_diff):.2f}%p 높은 정확도를 보였습니다.\n"
                )
            
            # 속도 비교
            tiny_time = tiny.get("avg_generation_time", 0)
            deepseek_time = deepseek.get("avg_generation_time", 0)
            if tiny_time < deepseek_time:
                speed_ratio = deepseek_time / tiny_time if tiny_time > 0 else 0
                lines.append(
                    f"2. **속도**: TinyLlama가 DeepSeek R1 70B보다 약 "
                    f"{speed_ratio:.1f}배 빠르게 생성했습니다.\n"
                )
            else:
                speed_ratio = tiny_time / deepseek_time if deepseek_time > 0 else 0
                lines.append(
                    f"2. **속도**: DeepSeek R1 70B가 TinyLlama보다 약 "
                    f"{speed_ratio:.1f}배 빠르게 생성했습니다.\n"
                )
            
            # 메모리 비교
            tiny_memory = sum(tiny.get("memory_usage", {}).values())
            deepseek_memory = sum(deepseek.get("memory_usage", {}).values())
            memory_ratio = (
                deepseek_memory / tiny_memory if tiny_memory > 0 else 0.0
            )
            lines.append(
                f"3. **메모리**: DeepSeek R1 70B는 TinyLlama보다 약 "
                f"{memory_ratio:.1f}배 많은 메모리를 사용했습니다.\n"
            )
            
            lines.append("\n### 권장사항\n\n")
            if deepseek.get("accuracy", 0) > tiny.get("accuracy", 0) + 5:
                lines.append(
                    "- **높은 정확도가 필요한 경우**: DeepSeek R1 70B 사용 권장\n"
                )
            if tiny_time < deepseek_time / 2:
                lines.append(
                    "- **빠른 응답이 필요한 경우**: TinyLlama 사용 권장\n"
                )
            lines.append(
                "- **리소스 제약이 있는 경우**: TinyLlama 사용 권장 "
                "(메모리 사용량이 적음)\n"
            )
        
        return lines

