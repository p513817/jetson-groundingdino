from dataclasses import dataclass, field
from typing import List, Optional

from dino import DINO
from utils import AsyncExecutor


@dataclass
class SharedParams:
    exec: Optional[AsyncExecutor] = None
    models: List[DINO] = field(default_factory=list)


shared_params = SharedParams()
