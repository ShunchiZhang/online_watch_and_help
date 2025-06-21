from requests.exceptions import RequestException
from urllib3.exceptions import HTTPError
from virtualhome.simulation.unity_simulator.comm_unity import (
    UnityCommunicationException,
)


def exception_info(e: Exception):
    return f"{type(e).__module__}.{type(e).__name__}: {e}"


class CustomException(Exception): ...


class NoneSubgoalsException(CustomException): ...


class DoubleGrabException(CustomException): ...


UnityExceptions = (
    UnityCommunicationException,
    RequestException,
    HTTPError,
    TimeoutError,
)

AllHandledExceptions = UnityExceptions + (CustomException,)
