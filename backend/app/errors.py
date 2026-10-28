"""Errors the frontend knows how to show. Every error response is {"error": code, "message": text}."""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


class ApiError(Exception):
    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


class NoPersonError(ApiError):
    def __init__(self):
        super().__init__("no_person", "No person was found in the video.", 422)


class TooFewFramesError(ApiError):
    def __init__(self, found: int, needed: int):
        super().__init__(
            "too_few_frames",
            f"The person was only visible in {found} frames; at least {needed} are needed.",
            422,
        )


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError):
        return JSONResponse({"error": exc.code, "message": exc.message}, status_code=exc.status)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError):
        first = exc.errors()[0] if exc.errors() else {}
        field = ".".join(str(p) for p in first.get("loc", [])[1:]) or "request"
        return JSONResponse(
            {"error": "invalid_request", "message": f"{field}: {first.get('msg', 'invalid')}"},
            status_code=422,
        )
