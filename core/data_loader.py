"""
벤치마크 데이터 로딩을 담당하는 모듈

이 모듈은 다양한 형식의 벤치마크 데이터를 로드하고 표준화된 형식으로
변환하는 기능을 제공합니다. 각 벤치마크 타입별로 전용 로더를 제공하여
데이터 형식의 차이를 추상화합니다.

주요 기능:
- JSONL 형식의 KMLE 벤치마크 데이터 로딩
- JSON 형식의 커스텀 QA 벤치마크 데이터 로딩
- 데이터 형식의 표준화 및 검증
"""
import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path


class BenchmarkDataLoader(ABC):
    """
    벤치마크 데이터 로더의 추상 기본 클래스
    
    모든 벤치마크 데이터 로더는 이 클래스를 상속받아
    표준화된 인터페이스를 제공합니다.
    """
    
    @abstractmethod
    def load(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        벤치마크 데이터 파일을 로드합니다.
        
        Args:
            file_path: 벤치마크 데이터 파일의 경로
        
        Returns:
            List[Dict[str, Any]]: 표준화된 형식의 벤치마크 항목 리스트
        
        Raises:
            FileNotFoundError: 파일이 존재하지 않을 때
            ValueError: 파일 형식이 올바르지 않을 때
        """
        pass
    
    @staticmethod
    def validate_benchmark_item(item: Dict[str, Any]) -> bool:
        """
        벤치마크 항목의 필수 필드 존재 여부를 검증합니다.
        
        Args:
            item: 검증할 벤치마크 항목 딕셔너리
        
        Returns:
            bool: 필수 필드가 모두 존재하면 True
        """
        required_fields = {"question"}
        return all(field in item for field in required_fields)


class KMLEBenchmarkDataLoader(BenchmarkDataLoader):
    """
    KMLE 2023 벤치마크 데이터를 로드하는 클래스
    
    KMLE 벤치마크는 JSONL 형식으로 저장되며, 각 라인은
    하나의 문제를 나타내는 JSON 객체입니다.
    """
    
    def load(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        JSONL 형식의 KMLE 벤치마크 데이터를 로드합니다.
        
        Args:
            file_path: JSONL 파일의 경로
        
        Returns:
            List[Dict[str, Any]]: 표준화된 KMLE 벤치마크 항목 리스트
        
        Raises:
            FileNotFoundError: 파일이 존재하지 않을 때
            json.JSONDecodeError: JSON 파싱 실패 시
        """
        if not file_path.exists():
            raise FileNotFoundError(f"벤치마크 파일을 찾을 수 없습니다: {file_path}")
        
        benchmark_items: List[Dict[str, Any]] = []
        
        with open(file_path, "r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    item = json.loads(line)
                    if self.validate_benchmark_item(item):
                        standardized_item = self._standardize_item(item, line_number)
                        benchmark_items.append(standardized_item)
                except json.JSONDecodeError as error:
                    print(
                        f"경고: 라인 {line_number} 파싱 실패 - {error}"
                    )
                    continue
        
        return benchmark_items
    
    def _standardize_item(
        self, item: Dict[str, Any], item_number: int
    ) -> Dict[str, Any]:
        """
        KMLE 벤치마크 항목을 표준화된 형식으로 변환합니다.
        
        Args:
            item: 원본 벤치마크 항목 딕셔너리
            item_number: 항목 번호 (라인 번호)
        
        Returns:
            Dict[str, Any]: 표준화된 벤치마크 항목
        """
        return {
            "item_number": item.get("no", item_number),
            "category": item.get("problem_category", "N/A"),
            "question": item["question"],
            "options": item.get("options", {}),
            "correct_answer_indices": item.get("answer_idx", []),
            "correct_answer_texts": item.get("answer", []),
            "context": item.get("context", "가장 적합한 답을 하나만 고르시오."),
            "raw_data": item,  # 원본 데이터 보존
        }


class CustomQABenchmarkDataLoader(BenchmarkDataLoader):
    """
    커스텀 JSON 형식의 QA 벤치마크 데이터를 로드하는 클래스
    
    이 로더는 간단한 질문-답변 형식의 벤치마크 데이터를 처리합니다.
    """
    
    def load(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        JSON 형식의 커스텀 QA 벤치마크 데이터를 로드합니다.
        
        Args:
            file_path: JSON 파일의 경로
        
        Returns:
            List[Dict[str, Any]]: 표준화된 QA 벤치마크 항목 리스트
        
        Raises:
            FileNotFoundError: 파일이 존재하지 않을 때
            json.JSONDecodeError: JSON 파싱 실패 시
        """
        if not file_path.exists():
            raise FileNotFoundError(f"벤치마크 파일을 찾을 수 없습니다: {file_path}")
        
        with open(file_path, "r", encoding="utf-8") as file:
            raw_data = json.load(file)
        
        if not isinstance(raw_data, list):
            raise ValueError("벤치마크 데이터는 리스트 형식이어야 합니다.")
        
        benchmark_items: List[Dict[str, Any]] = []
        for index, item in enumerate(raw_data, start=1):
            if self.validate_benchmark_item(item):
                standardized_item = self._standardize_item(item, index)
                benchmark_items.append(standardized_item)
        
        return benchmark_items
    
    def _standardize_item(
        self, item: Dict[str, Any], item_number: int
    ) -> Dict[str, Any]:
        """
        커스텀 QA 벤치마크 항목을 표준화된 형식으로 변환합니다.
        
        Args:
            item: 원본 벤치마크 항목 딕셔너리
            item_number: 항목 번호
        
        Returns:
            Dict[str, Any]: 표준화된 벤치마크 항목
        """
        return {
            "item_number": item.get("id", item_number),
            "category": item.get("category", "unknown"),
            "question": item["question"],
            "correct_answer_texts": [item.get("answer", "")],
            "correct_answer_indices": [],
            "options": {},
            "context": "",
            "raw_data": item,  # 원본 데이터 보존
        }

