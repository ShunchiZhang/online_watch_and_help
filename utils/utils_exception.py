import time

import litellm
import openai
from json_repair import repair_json


def exception_info(e: Exception):
    return f"{type(e).__module__}.{type(e).__name__}: {e}"


def is_unity_exception(e):
    """
    NOTE:
    - Importing the same `Exception` class differently by `pip` and `PYTHONPATH` will cause `isinstance` failure.
    """
    return (
        type(e).__name__ == "UnityCommunicationException"
        and "unity_simulator.comm_unity" in type(e).__module__
    )


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


def is_quota_exceeded(e):
    # https://docs.litellm.ai/docs/exception_mapping
    # https://platform.openai.com/docs/guides/error-codes
    # https://ai.google.dev/gemini-api/docs/troubleshooting
    info = repair_json(str(e), return_objects=True)
    if isinstance(e, litellm.RateLimitError):  # inherit from openai.RateLimitError
        if "exhausted" in get_dict_chain(info, "error.message"):
            return True
        elif "rate-limits" in get_dict_chain(info, "error.message"):
            # * sleep if `rate limit` rather than `quota exceeded`
            try:
                details = info["error"]["details"]
                for detail in details:
                    if detail["@type"] == "type.googleapis.com/google.rpc.RetryInfo":
                        delay = detail["retryDelay"]
                        assert isinstance(delay, str) and delay.endswith("s")
                        sleep_with_progress(int(delay[:-1]), interval=5)
                        return False
            except Exception:
                ...
    elif isinstance(e, openai.RateLimitError):
        if "insufficient_quota" in str(e):
            return True
    return False


class CustomException(Exception): ...


class NoneSubgoalsException(CustomException): ...


class DoubleGrabException(CustomException): ...


class CheckProgressException(CustomException): ...
