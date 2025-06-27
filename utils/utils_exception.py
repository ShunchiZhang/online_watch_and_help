import time
import traceback
from datetime import datetime
from pathlib import Path

import litellm
import openai
from json_repair import repair_json

"""
Use CustomError to abort the program.
Use CustomException to handle the error.

`logger.exception` is equivalent to `logger.error(msg, exc_info=True)`,
which will record the traceback if called in a `try-except` block.
"""


def exception_info(e: Exception):
    return f"{type(e).__module__}.{type(e).__name__}: {e}"


def exception_traceback(e: Exception):
    tb = "".join(traceback.TracebackException.from_exception(e).format())
    curr_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tb_dir = Path() / "logs" / "tracebacks"
    tb_dir.mkdir(parents=True, exist_ok=True)

    tb_file = tb_dir / f"{curr_time}.log"
    tb_file.write_text(tb)

    return f"{exception_info(e)}\nDetailed Traceback: '{tb_file.resolve()}'"


# ! Exception
class CustomException(Exception): ...


class NoneSubgoalsException(CustomException): ...


class DoubleGrabException(CustomException): ...


class CheckProgressException(CustomException): ...


class ParticleLengthException(CustomException): ...


class RateLimitException(CustomException): ...


# ! Error
class CustomError(Exception): ...


class UnityError(CustomError): ...


class QuotaExceededError(CustomError): ...


class UnknownError(CustomError): ...


def handle(e, logger, allow=None, exc_info=False):
    if isinstance(e, CustomException) or (allow and isinstance(e, allow)):
        logger.exception(exception_info(e))
    else:
        if exc_info:
            logger.error(exception_traceback(e))
        else:
            logger.error(exception_info(e))

        if isinstance(e, CustomError):
            raise e
        else:
            raise UnknownError from e


def check_unity_error(e):
    """
    NOTE:
    - Importing the same `Exception` class differently by `pip` and `PYTHONPATH` will cause `isinstance` failure.
    """
    if (
        type(e).__name__ == "UnityCommunicationException"
        and "unity_simulator.comm_unity" in type(e).__module__
    ):
        return UnityError(exception_info(e))
    return e


def get_dict_chain(d: dict, chain: str):
    chain = chain.split(".")
    for key in chain:
        if d is None:
            return None
        d = d.get(key, None)
    return d


def sleep_with_progress(n: int, interval: int = 5):
    print(f"sleep {n}s", end=": ", flush=True)
    for _ in range(0, n, interval):
        print(_, end="...", flush=True)
        time.sleep(interval)


def check_quota_exceeded(e):
    # https://docs.litellm.ai/docs/exception_mapping
    # https://platform.openai.com/docs/guides/error-codes
    # https://ai.google.dev/gemini-api/docs/troubleshooting
    info = repair_json(str(e), return_objects=True)
    if not isinstance(info, dict):
        info = {}

    if isinstance(e, litellm.RateLimitError):  # inherit from openai.RateLimitError
        if "exhausted" or "exceeded your current quota" in str(e):
            return QuotaExceededError(exception_info(e))
        if "rate-limits" in get_dict_chain(info, "error.message"):
            # * sleep if `rate limit` rather than `quota exceeded`
            try:
                details = info["error"]["details"]
                for detail in details:
                    if detail["@type"] == "type.googleapis.com/google.rpc.RetryInfo":
                        delay = detail["retryDelay"]
                        assert isinstance(delay, str) and delay.endswith("s")
                        sleep_with_progress(int(delay[:-1]), interval=5)
                        return RateLimitException(exception_info(e))
            except Exception:
                return e
    elif isinstance(e, openai.RateLimitError):
        if "insufficient_quota" in str(e):
            return QuotaExceededError(exception_info(e))
    return e
