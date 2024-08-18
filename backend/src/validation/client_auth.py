from pydantic import BaseModel
from typing import Optional


class ClientAuth(BaseModel):
    """
    ClientAuth class for storing client authentication details.
    """
    uri: Optional[str]
    userName: Optional[str]
    password: Optional[str]
    database: Optional[str]
    openai_key: Optional[str]

    def __str__(self):
        return str(self.dict())

    def __repr__(self):
        return self.__str__()
