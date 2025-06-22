from openai import RateLimitError


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


def is_openai_quota_exceeded(e):
    return isinstance(e, RateLimitError) and "insufficient_quota" in str(e)


class CustomException(Exception): ...


class NoneSubgoalsException(CustomException): ...


class DoubleGrabException(CustomException): ...


class CheckProgressException(CustomException): ...
