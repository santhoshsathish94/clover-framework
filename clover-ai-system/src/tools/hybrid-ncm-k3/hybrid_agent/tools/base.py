"""Tool interface: a tool is triggered by `<tag>content</tag>` in the agent's text and
returns the observation text that is appended to the working memory inside <result>.
"""
from abc import ABC, abstractmethod


class BaseTool(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def trigger_tag(self) -> str:
        """Tag used in model output: <tag>content</tag>."""
        pass

    @abstractmethod
    def execute(self, content: str, **kwargs) -> str:
        pass
