"""Power-source contract. Metadata belongs to the source and every raw trace."""

from typing import Protocol, runtime_checkable

from edge_ai.power.trace import PowerMetadata, PowerTrace


@runtime_checkable
class PowerSource(Protocol):
    @property
    def metadata(self) -> PowerMetadata: ...

    def start(self) -> None: ...

    def stop(self) -> PowerTrace: ...
