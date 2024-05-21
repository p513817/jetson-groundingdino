from asyncio import AbstractEventLoop
from concurrent.futures import ThreadPoolExecutor
from typing import Callable


class AsyncExecutor:
    def __init__(
        self,
        loop: AbstractEventLoop,
        executor: ThreadPoolExecutor,
    ) -> None:
        self.loop = loop
        self.executor = executor

    async def __call__(self, func: Callable, *args):
        """Execute block function with thread and control by event loop"""
        return await self.loop.run_in_executor(self.executor, func, *args)
