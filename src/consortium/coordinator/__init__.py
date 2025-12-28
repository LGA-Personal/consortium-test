"""Coordinator: Session management, work dispatch, audit orchestration."""

from consortium.coordinator.auditor import (
    AuditRecord,
    AuditStats,
    AuditStatus,
    Auditor,
    select_verifier_node,
)
from consortium.coordinator.failover import (
    execute_failover,
    FailoverManager,
    FailoverOperation,
    FailoverStats,
    FailoverStatus,
    FailureEvent,
    FailureType,
)
from consortium.coordinator.server import (
    Coordinator,
    CoordinatorServicer,
    CoordinatorState,
    RegisteredNode,
    run_coordinator,
)

__all__ = [
    # Auditor
    "AuditRecord",
    "AuditStats",
    "AuditStatus",
    "Auditor",
    "select_verifier_node",
    # Failover
    "execute_failover",
    "FailoverManager",
    "FailoverOperation",
    "FailoverStats",
    "FailoverStatus",
    "FailureEvent",
    "FailureType",
    # Server
    "Coordinator",
    "CoordinatorServicer",
    "CoordinatorState",
    "RegisteredNode",
    "run_coordinator",
]
