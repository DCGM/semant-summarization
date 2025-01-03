from abc import ABC
from typing import Optional

from classconfig import ConfigurableValue
from classconfig.validators import StringValidator


class API(ABC):
    """
    Base class for API.
    """
    api_key: str = ConfigurableValue(desc="OpenAI API key.", validator=StringValidator())
    base_url: Optional[str] = ConfigurableValue(desc="Base URL for API.", user_default=None, voluntary=True)

