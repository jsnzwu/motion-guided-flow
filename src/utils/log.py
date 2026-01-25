from wickit.utils.log import (
    add_prefix_to_log,
    configure_logging,
    get_local_rank,
    log,
    shutdown_log,
)

configure_logging()

__all__ = [
    "add_prefix_to_log",
    "configure_logging",
    "get_local_rank",
    "log",
    "shutdown_log",
]
