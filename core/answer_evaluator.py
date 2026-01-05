"""
생성된 답변의 정확도를 평가하는 모듈

이 모듈은 모델이 생성한 답변을 정답과 비교하여
정확도를 판단하는 다양한 평가 전략을 제공합니다.

주요 기능:
- 인덱스 기반 답변 평가 (1, 2, 3, 4, 5 등)
- 텍스트 기반 답변 평가 (부분 일치, 대소문자 무시 등)
- 복합 평가 전략 (인덱스와 텍스트 모두 확인)
"""
import re
from typing import List, Optional
from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationResult:
    """
    답변 평가 결과를 담는 불변 데이터 클래스
    
    Attributes:
        is_correct: 정답 여부
        matched_indices: 매칭된 정답 인덱스 리스트
        matched_texts: 매칭된 정답 텍스트 리스트
        evaluation_method: 사용된 평가 방법
    """
    is_correct: bool
    matched_indices: List[int] = None
    matched_texts: List[str] = None
    evaluation_method: str = "unknown"
    
    def __post_init__(self) -> None:
        """초기화 후 기본값 설정"""
        if self.matched_indices is None:
            object.__setattr__(self, "matched_indices", [])
        if self.matched_texts is None:
            object.__setattr__(self, "matched_texts", [])


class AnswerEvaluator:
    """
    생성된 답변을 평가하는 클래스
    
    이 클래스는 다양한 패턴과 전략을 사용하여
    모델이 생성한 답변이 정답과 일치하는지 판단합니다.
    """
    
    # 답변 인덱스를 찾기 위한 정규표현식 패턴들
    INDEX_PATTERNS = [
        r"답:\s*(\d+)",
        r"답변:\s*(\d+)",
        r"(\d+)번",
        r"선택지\s*(\d+)",
        r"옵션\s*(\d+)",
        r"정답은\s*(\d+)",
        r"정답:\s*(\d+)",
    ]
    
    def evaluate(
        self,
        generated_text: str,
        correct_answer_indices: List[int],
        correct_answer_texts: List[str],
    ) -> EvaluationResult:
        """
        생성된 답변을 평가합니다.
        
        먼저 인덱스 기반 평가를 시도하고, 실패하면 텍스트 기반 평가를 수행합니다.
        
        Args:
            generated_text: 모델이 생성한 답변 텍스트
            correct_answer_indices: 정답 인덱스 리스트 (예: [1, 2])
            correct_answer_texts: 정답 텍스트 리스트 (예: ["파리", "Paris"])
        
        Returns:
            EvaluationResult: 평가 결과 객체
        """
        # 인덱스 기반 평가
        if correct_answer_indices:
            index_result = self._evaluate_by_index(
                generated_text, correct_answer_indices
            )
            if index_result.is_correct:
                return index_result
        
        # 텍스트 기반 평가
        if correct_answer_texts:
            text_result = self._evaluate_by_text(
                generated_text, correct_answer_texts
            )
            if text_result.is_correct:
                return text_result
        
        # 모두 실패
        return EvaluationResult(
            is_correct=False,
            evaluation_method="none",
        )
    
    def _evaluate_by_index(
        self, generated_text: str, correct_indices: List[int]
    ) -> EvaluationResult:
        """
        인덱스 기반으로 답변을 평가합니다.
        
        생성된 텍스트에서 숫자 인덱스를 추출하여
        정답 인덱스와 비교합니다.
        
        Args:
            generated_text: 생성된 답변 텍스트
            correct_indices: 정답 인덱스 리스트
        
        Returns:
            EvaluationResult: 평가 결과
        """
        matched_indices: List[int] = []
        
        for pattern in self.INDEX_PATTERNS:
            matches = re.findall(pattern, generated_text, re.IGNORECASE)
            for match in matches:
                try:
                    index = int(match)
                    if index in correct_indices:
                        matched_indices.append(index)
                except ValueError:
                    continue
        
        is_correct = len(matched_indices) > 0
        
        return EvaluationResult(
            is_correct=is_correct,
            matched_indices=matched_indices,
            evaluation_method="index_pattern_matching",
        )
    
    def _evaluate_by_text(
        self, generated_text: str, correct_texts: List[str]
    ) -> EvaluationResult:
        """
        텍스트 기반으로 답변을 평가합니다.
        
        생성된 텍스트에 정답 텍스트가 포함되어 있는지
        대소문자를 무시하고 확인합니다.
        
        Args:
            generated_text: 생성된 답변 텍스트
            correct_texts: 정답 텍스트 리스트
        
        Returns:
            EvaluationResult: 평가 결과
        """
        generated_lower = generated_text.lower()
        matched_texts: List[str] = []
        
        for correct_text in correct_texts:
            correct_lower = correct_text.lower()
            if correct_lower in generated_lower:
                matched_texts.append(correct_text)
        
        is_correct = len(matched_texts) > 0
        
        return EvaluationResult(
            is_correct=is_correct,
            matched_texts=matched_texts,
            evaluation_method="text_substring_matching",
        )

