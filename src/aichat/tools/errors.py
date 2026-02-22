from __future__ import annotations


class ToolRequestError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


RETRYABLE_HTTP_STATUSES = {429, 500, 502, 503, 504}


def is_retryable_status(status_code: int | None) -> bool:
    return status_code in RETRYABLE_HTTP_STATUSES
