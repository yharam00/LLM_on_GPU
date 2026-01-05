# LLM 벤치마크 시스템 (리팩토링 버전)

이 디렉토리는 기존 `scripts` 폴더의 코드를 리팩토링한 버전입니다.

## 리팩토링 원칙

1. **언어 기능 최대 활용**: Python의 표준 라이브러리와 언어 기능을 최대한 활용
2. **중복 제거**: 모든 중복 코드를 제거하고 공통 기능을 모듈화
3. **상세한 문서화**: 모든 함수와 클래스에 상세한 DocString 작성
4. **명확한 네이밍**: 문학적이고 명확한 이름으로 코드 가독성 향상
5. **Lint 준수**: 국제 표준에 따른 Lint 규칙 100% 준수

## 디렉토리 구조

```
refactored/
├── core/                    # 핵심 기능 모듈
│   ├── model_loader.py      # 모델 로딩 및 관리
│   ├── gpu_monitor.py       # GPU 모니터링
│   ├── data_loader.py       # 벤치마크 데이터 로딩
│   ├── answer_evaluator.py # 답변 평가
│   ├── benchmark_runner.py # 벤치마크 실행
│   └── report_generator.py # 리포트 생성
├── config/                  # 설정 관리
│   └── settings.py          # 중앙화된 설정
└── scripts/                 # 실행 가능한 스크립트
    ├── run_kmle_benchmark.py
    └── generate_kmle_report.py
```

## 주요 개선사항

### 1. 모듈화 및 재사용성
- 공통 기능을 독립적인 모듈로 분리
- 각 모듈은 단일 책임 원칙을 따름
- 모듈 간 의존성 최소화

### 2. 타입 힌팅
- 모든 함수와 메서드에 타입 힌팅 추가
- `dataclass`를 활용한 데이터 구조 정의
- 타입 안정성 향상

### 3. 설정 중앙화
- 모든 경로와 설정값을 `config/settings.py`에서 관리
- 하드코딩된 경로 제거
- 환경별 설정 변경 용이

### 4. 에러 처리
- 명확한 예외 메시지
- 예외 체이닝을 통한 디버깅 용이성 향상

### 5. 문서화
- 모든 공개 함수/클래스에 상세한 DocString
- 입력/출력 타입 및 설명 포함
- 사용 예시 포함

## 사용법

### KMLE 벤치마크 실행

```bash
cd /home/ychanmo/llm_testing/refactored
python scripts/run_kmle_benchmark.py
```

### 리포트 생성

```bash
python scripts/generate_kmle_report.py
```

또는 특정 결과 파일 지정:

```bash
python scripts/generate_kmle_report.py /path/to/results.json
```

## 모듈 설명

### core/model_loader.py
- `ModelLoader`: 모델과 토크나이저를 로드하고 관리
- `QuantizationConfiguration`: 양자화 설정을 담는 데이터 클래스

### core/gpu_monitor.py
- `GPUMonitor`: GPU 리소스 모니터링
- `GPUStatistics`: GPU 통계 정보를 담는 데이터 클래스

### core/data_loader.py
- `BenchmarkDataLoader`: 벤치마크 데이터 로더 추상 클래스
- `KMLEBenchmarkDataLoader`: KMLE 2023 데이터 로더
- `CustomQABenchmarkDataLoader`: 커스텀 QA 데이터 로더

### core/answer_evaluator.py
- `AnswerEvaluator`: 생성된 답변의 정확도 평가
- `EvaluationResult`: 평가 결과를 담는 데이터 클래스

### core/benchmark_runner.py
- `BenchmarkRunner`: 벤치마크 실행 및 결과 수집
- `BenchmarkExecutionResult`: 벤치마크 실행 결과를 담는 데이터 클래스

### core/report_generator.py
- `ReportGenerator`: 벤치마크 결과 리포트 생성
- `BenchmarkReport`: 생성된 리포트 정보를 담는 데이터 클래스

## 기존 코드와의 차이점

1. **중복 제거**: 기존에 여러 파일에 중복되어 있던 모델 로딩, GPU 모니터링 코드를 단일 모듈로 통합
2. **설정 관리**: 하드코딩된 경로를 설정 파일로 이동
3. **타입 안정성**: 타입 힌팅을 통한 오류 사전 방지
4. **테스트 용이성**: 모듈화를 통한 단위 테스트 작성 용이
5. **확장성**: 새로운 벤치마크 타입이나 모델 추가가 용이한 구조

## 요구사항

- Python 3.8+
- torch
- transformers
- bitsandbytes (양자화 사용 시)

## Lint 검사

모든 코드는 다음 규칙을 준수합니다:
- PEP 8 스타일 가이드
- mypy 타입 체크 (가능한 경우)
- pylint 또는 flake8 검사 통과

