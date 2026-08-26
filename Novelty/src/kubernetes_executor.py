"""Both execution backends: paper rolling update and novelty in-place resize."""
from Paper.src.kubernetes_executor import InPlaceResizeExecutor, RollingUpdateExecutor

__all__=['RollingUpdateExecutor','InPlaceResizeExecutor']
