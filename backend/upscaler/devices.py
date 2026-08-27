from __future__ import annotations

import os
import platform
import shutil
from typing import Any


def platform_capabilities() -> dict[str, Any]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "vulkan_tools_detected": bool(shutil.which("vulkaninfo")),
    }
