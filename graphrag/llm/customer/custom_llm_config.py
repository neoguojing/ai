# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""OpenAI Configuration class definition."""

import json
from collections.abc import Hashable
from typing import Any, cast

from graphrag.llm.types import LLMConfig


def _non_blank(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return None if stripped == "" else value


class CustomConfiguration(Hashable, LLMConfig):
    """OpenAI Configuration class definition."""

    # Core Configuration
    _access_key: str
    _secret_key: str
    _token: str
    

    def __init__(
        self,
        config: dict,
    ):
        """Init method definition."""

        def lookup_required(key: str) -> str:
            return cast(str, config.get(key))

        def lookup_str(key: str) -> str | None:
            return cast(str | None, config.get(key))

        def lookup_int(key: str) -> int | None:
            result = config.get(key)
            if result is None:
                return None
            return int(cast(int, result))

        def lookup_float(key: str) -> float | None:
            result = config.get(key)
            if result is None:
                return None
            return float(cast(float, result))

        def lookup_dict(key: str) -> dict | None:
            return cast(dict | None, config.get(key))

        def lookup_list(key: str) -> list | None:
            return cast(list | None, config.get(key))

        def lookup_bool(key: str) -> bool | None:
            value = config.get(key)
            if isinstance(value, str):
                return value.upper() == "TRUE"
            if isinstance(value, int):
                return value > 0
            return cast(bool | None, config.get(key))

        self.access_key = lookup_str("ak")
        self.secret_key = lookup_str("sk")
        self.token = lookup_str("token")
        self._raw_config = config

    @property
    def access_key(self) -> str:
        """API key property definition."""
        return self.access_key

    @property
    def secret_key(self) -> str:
        """Model property definition."""
        return self.secret_key

    @property
    def token(self) -> str | None:
        """Deployment name property definition."""
        return self.token
    
    @property
    def raw_config(self) -> dict:
        """Raw config method definition."""
        return self._raw_config

    def lookup(self, name: str, default_value: Any = None) -> Any:
        """Lookup method definition."""
        return self._raw_config.get(name, default_value)

    def __str__(self) -> str:
        """Str method definition."""
        return json.dumps(self.raw_config, indent=4)

    def __repr__(self) -> str:
        """Repr method definition."""
        return f"OpenAIConfiguration({self._raw_config})"

    def __eq__(self, other: object) -> bool:
        """Eq method definition."""
        if not isinstance(other, CustomConfiguration):
            return False
        return self._raw_config == other._raw_config

    def __hash__(self) -> int:
        """Hash method definition."""
        return hash(tuple(sorted(self._raw_config.items())))
