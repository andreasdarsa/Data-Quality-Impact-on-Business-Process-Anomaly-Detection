from .config import DegradationConfig
from .dispatcher import apply_degradation

from .completeness import inject_missing_events
from .accuracy import inject_accuracy_noise
from .consistency import inject_consistency_issues
from .timeliness import inject_timeliness_issues