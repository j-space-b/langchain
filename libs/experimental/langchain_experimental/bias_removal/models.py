"""Model for the Debias Chain."""
from langchain.pydantic_v1 import BaseModel


class Debias(BaseModel):
    """Class for a bias evaluation."""

    bias_critique_request: str
    bias_revision_request: str
    name: str = "Debias"
