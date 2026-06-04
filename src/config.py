from __future__ import annotations

import os


# If the exact ISU is needed later, we can override it via environment:
# ISU_ID=123456 python3 -m src.experiments
ISU_ID = int(os.getenv("ISU_ID", "0"))

