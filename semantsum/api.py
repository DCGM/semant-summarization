from classconfig import ConfigurableValue
from classconfig.validators import StringValidator


class API:
    """
    Base class for API.
    """
    api_key: str = ConfigurableValue(desc="OpenAI API key.", validator=StringValidator())
