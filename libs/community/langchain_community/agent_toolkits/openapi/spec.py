"""Quick and dirty representation for OpenAPI specs."""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass(frozen=True)
class ReducedOpenAPISpec:
    """A reduced OpenAPI spec.

    This is a quick and dirty representation for OpenAPI specs.

    Attributes:
        servers: The servers in the spec.
        description: The description of the spec.
        endpoints: The endpoints in the spec.
    """

    servers: List[Dict[str, Any]]
    description: str
    endpoints: List[Tuple[str, str, Dict[str, Any]]]


def dereference_refs(docs: Dict[str, Any], full_schema: Dict[str, Any]) -> Dict[str, Any]:
    """Dereference refs in the given docs."""
    def resolve_ref(ref: str) -> Dict[str, Any]:
        ref_parts = ref.split("/")
        pointer = "#/"
        for part in ref_parts[1:]:
            pointer += f"/{part}"
            if part not in full_schema:
              
