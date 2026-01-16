# LLM 벤치마크 시스템

이 프로젝트는 LLM(Large Language Model)의 성능을 벤치마크하는 시스템입니다.

## 리팩토링 원칙

1. **언어 기능 최대 활용**: Python의 표준 라이브러리와 언어 기능을 최대한 활용
2. **중복 제거**: 모든 중복 코드를 제거하고 공통 기능을 모듈화
3. **상세한 문서화**: 모든 함수와 클래스에 상세한 DocString 작성
4. **명확한 네이밍**: 문학적이고 명확한 이름으로 코드 가독성 향상
5. **Lint 준수**: 국제 표준에 따른 Lint 규칙 100% 준수

## 디렉토리 구조

```
LLM_on_GPU/
├── core/                    # 핵심 기능 모듈
│   ├── __init__.py
│   ├── model_loader.py      # 모델 로딩 및 관리
│   ├── gpu_monitor.py       # GPU 모니터링
│   ├── data_loader.py       # 벤치마크 데이터 로딩
│   ├── answer_evaluator.py  # 답변 평가
│   ├── benchmark_runner.py  # 벤치마크 실행
│   └── report_generator.py  # 리포트 생성
├── config/                  # 설정 관리
│   ├── __init__.py
│   └── settings.py          # 중앙화된 설정
├── scripts/                 # 실행 가능한 스크립트
│   ├── __init__.py
│   ├── run_kmle_benchmark.py           # KMLE 2023 벤치마크 실행
│   ├── run_custom_qa_benchmark.py      # 커스텀 QA 벤치마크 실행
│   ├── run_single_model_benchmark.py   # 단일 모델 벤치마크 실행
│   └── generate_kmle_report.py         # KMLE 리포트 생성
├── benchmarks/              # 벤치마크 데이터
├── results/                 # 벤치마크 결과 JSON 파일
├── logs/                    # 로그 파일
├── docs/                    # 생성된 리포트 마크다운 파일
└── models/                  # 다운로드된 모델 (선택사항)
```

**참고**: 
- `benchmarks/`, `results/`, `logs/`, `docs/` 디렉토리는 자동으로 생성되며, 프로젝트 내부에 파일이 있으면 우선 사용하고, 없으면 상위 디렉토리의 기존 파일을 사용합니다.
- `models/` 디렉토리는 다운로드된 모델들이 저장되는 위치입니다. 모델은 **항상 프로젝트 내부**(`LLM_on_GPU/models/`)에 저장되어 프로젝트 독립성을 유지합니다. 모델은 Hugging Face Hub 형식으로 저장되며, 첫 실행 시 다운로드되고 이후에는 캐시에서 빠르게 로드됩니다.

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

### 단일 모델 벤치마크 실행 (권장)

가장 유연한 방법으로, 커맨드라인에서 모델을 지정하여 벤치마크를 실행할 수 있습니다.

```bash
# 기본 사용 (양자화 없이)
python scripts/run_single_model_benchmark.py --model-name openai/gpt-oss-20b

# 4-bit 양자화 사용
python scripts/run_single_model_benchmark.py --model-name openai/gpt-oss-20b --use-quantization

# 커스텀 QA 벤치마크 사용
python scripts/run_single_model_benchmark.py --model-name openai/gpt-oss-20b --benchmark-type custom_qa

# Gated 모델 사용 (토큰 필요)
python scripts/run_single_model_benchmark.py --model-name google/medgemma-27b-text-it --use-quantization --token YOUR_HF_TOKEN
# 또는 환경 변수 사용: export HF_TOKEN=YOUR_HF_TOKEN
```

### KMLE 벤치마크 실행

여러 모델을 한 번에 실행하는 스크립트입니다.

```bash
python scripts/run_kmle_benchmark.py
```

### 커스텀 QA 벤치마크 실행

```bash
# 기본 모델 사용
python scripts/run_custom_qa_benchmark.py

# 특정 모델 지정
python scripts/run_custom_qa_benchmark.py --model_name openai/gpt-oss-20b --use_quantization
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


