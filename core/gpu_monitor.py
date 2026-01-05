"""
GPU 리소스 모니터링을 담당하는 모듈

이 모듈은 nvidia-smi를 활용하여 GPU의 사용률, 메모리, 온도, 전력 소비 등의
지표를 실시간으로 수집하고 분석하는 기능을 제공합니다.

주요 기능:
- 실시간 GPU 통계 수집
- 백그라운드 모니터링 스레드 지원
- GPU 통계 데이터의 시계열 저장 및 분석
"""
import subprocess
import time
import threading
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict


@dataclass
class GPUStatistics:
    """
    단일 GPU의 통계 정보를 담는 데이터 클래스
    
    Attributes:
        index: GPU 인덱스 번호
        name: GPU 모델 이름
        gpu_utilization_percent: GPU 사용률 (0-100)
        memory_utilization_percent: 메모리 사용률 (0-100)
        memory_used_megabytes: 사용 중인 메모리 (MB)
        memory_total_megabytes: 전체 메모리 (MB)
        temperature_celsius: GPU 온도 (섭씨)
        power_draw_watts: 전력 소비량 (와트)
        timestamp: 통계 수집 시각
    """
    index: str
    name: str = "N/A"
    gpu_utilization_percent: Optional[float] = None
    memory_utilization_percent: Optional[float] = None
    memory_used_megabytes: Optional[float] = None
    memory_total_megabytes: Optional[float] = None
    temperature_celsius: Optional[float] = None
    power_draw_watts: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def memory_usage_percentage(self) -> Optional[float]:
        """
        메모리 사용률을 백분율로 계산합니다.
        
        Returns:
            Optional[float]: 메모리 사용률 (0-100), 계산 불가 시 None
        """
        if (
            self.memory_used_megabytes is not None
            and self.memory_total_megabytes is not None
            and self.memory_total_megabytes > 0
        ):
            return (self.memory_used_megabytes / self.memory_total_megabytes) * 100.0
        return None


class GPUMonitor:
    """
    GPU 리소스를 모니터링하는 클래스
    
    이 클래스는 nvidia-smi를 통해 GPU 통계를 수집하고,
    필요시 백그라운드 스레드에서 지속적으로 모니터링할 수 있습니다.
    
    Attributes:
        monitoring_active: 모니터링 활성화 여부
        monitoring_thread: 백그라운드 모니터링 스레드
        collected_statistics: 수집된 통계 데이터 리스트
    """
    
    NVIDIA_SMI_QUERY_FIELDS = [
        "index",
        "name",
        "utilization.gpu",
        "utilization.memory",
        "memory.used",
        "memory.total",
        "temperature.gpu",
        "power.draw",
    ]
    
    def __init__(self) -> None:
        """GPUMonitor 인스턴스 초기화"""
        self.monitoring_active: bool = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.collected_statistics: List[GPUStatistics] = []
        self._lock = threading.Lock()
    
    def get_current_gpu_statistics(self) -> List[GPUStatistics]:
        """
        현재 시점의 모든 GPU 통계를 수집합니다.
        
        Returns:
            List[GPUStatistics]: 각 GPU의 통계 정보 리스트
        """
        try:
            query_string = ",".join(self.NVIDIA_SMI_QUERY_FIELDS)
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--query-gpu={query_string}",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
            
            statistics_list: List[GPUStatistics] = []
            lines = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
            
            for line in lines:
                parts = [part.strip() for part in line.split(",")]
                if len(parts) >= len(self.NVIDIA_SMI_QUERY_FIELDS):
                    statistics = self._parse_gpu_statistics_line(parts)
                    if statistics is not None:
                        statistics_list.append(statistics)
            
            return statistics_list
        
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError) as error:
            print(f"GPU 통계 수집 실패: {error}")
            return []
    
    def _parse_gpu_statistics_line(self, parts: List[str]) -> Optional[GPUStatistics]:
        """
        nvidia-smi 출력 라인을 GPUStatistics 객체로 파싱합니다.
        
        Args:
            parts: 쉼표로 분리된 필드 값들의 리스트
        
        Returns:
            Optional[GPUStatistics]: 파싱된 통계 객체, 실패 시 None
        """
        try:
            def safe_float(value: str) -> Optional[float]:
                """안전하게 문자열을 float로 변환"""
                if value in ("[N/A]", "N/A", "") or "Insufficient" in value:
                    return None
                try:
                    return float(value)
                except ValueError:
                    return None
            
            return GPUStatistics(
                index=parts[0],
                name=parts[1] if len(parts) > 1 else "N/A",
                gpu_utilization_percent=safe_float(parts[2]) if len(parts) > 2 else None,
                memory_utilization_percent=safe_float(parts[3]) if len(parts) > 3 else None,
                memory_used_megabytes=safe_float(parts[4]) if len(parts) > 4 else None,
                memory_total_megabytes=safe_float(parts[5]) if len(parts) > 5 else None,
                temperature_celsius=safe_float(parts[6]) if len(parts) > 6 else None,
                power_draw_watts=safe_float(parts[7]) if len(parts) > 7 else None,
                timestamp=datetime.now(),
            )
        except (IndexError, ValueError):
            return None
    
    def start_background_monitoring(
        self, interval_seconds: float = 0.5
    ) -> None:
        """
        백그라운드 스레드에서 GPU 모니터링을 시작합니다.
        
        Args:
            interval_seconds: 통계 수집 간격 (초)
        """
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.collected_statistics.clear()
        
        def monitoring_loop() -> None:
            while self.monitoring_active:
                statistics = self.get_current_gpu_statistics()
                with self._lock:
                    self.collected_statistics.extend(statistics)
                time.sleep(interval_seconds)
        
        self.monitoring_thread = threading.Thread(
            target=monitoring_loop, daemon=True
        )
        self.monitoring_thread.start()
    
    def stop_background_monitoring(self) -> None:
        """백그라운드 모니터링을 중지합니다."""
        self.monitoring_active = False
        if self.monitoring_thread is not None:
            self.monitoring_thread.join(timeout=2.0)
    
    def get_collected_statistics(self) -> List[GPUStatistics]:
        """
        수집된 통계 데이터를 반환합니다.
        
        Returns:
            List[GPUStatistics]: 수집된 모든 통계 데이터의 복사본
        """
        with self._lock:
            return self.collected_statistics.copy()
    
    def get_statistics_summary_by_gpu(
        self,
    ) -> Dict[str, Dict[str, Any]]:
        """
        GPU별로 통계 데이터를 집계하여 요약 정보를 생성합니다.
        
        Returns:
            Dict[str, Dict[str, Any]]: 
                GPU 인덱스를 키로 하는 요약 정보 딕셔너리
                각 값은 평균, 최대값 등의 집계 정보를 포함
        """
        statistics = self.get_collected_statistics()
        if not statistics:
            return {}
        
        gpu_data: Dict[str, List[GPUStatistics]] = defaultdict(list)
        for stat in statistics:
            gpu_data[stat.index].append(stat)
        
        summary: Dict[str, Dict[str, Any]] = {}
        for gpu_index, stats_list in gpu_data.items():
            gpu_utils = [
                s.gpu_utilization_percent
                for s in stats_list
                if s.gpu_utilization_percent is not None
            ]
            mem_utils = [
                s.memory_utilization_percent
                for s in stats_list
                if s.memory_utilization_percent is not None
            ]
            mem_useds = [
                s.memory_used_megabytes
                for s in stats_list
                if s.memory_used_megabytes is not None
            ]
            
            summary[gpu_index] = {
                "sample_count": len(stats_list),
                "average_gpu_utilization": (
                    sum(gpu_utils) / len(gpu_utils) if gpu_utils else None
                ),
                "max_gpu_utilization": max(gpu_utils) if gpu_utils else None,
                "average_memory_utilization": (
                    sum(mem_utils) / len(mem_utils) if mem_utils else None
                ),
                "max_memory_utilization": max(mem_utils) if mem_utils else None,
                "average_memory_used_mb": (
                    sum(mem_useds) / len(mem_useds) if mem_useds else None
                ),
                "max_memory_used_mb": max(mem_useds) if mem_useds else None,
            }
        
        return summary

