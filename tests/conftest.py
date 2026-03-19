#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared pytest configuration file, used to hook summary report."""

import logging

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import pytest


# get library-specific logger
logger = logging.getLogger(__name__)
"""Logger for Churn Library Test Summary"""


def pytest_terminal_summary(terminalreporter: 'pytest.TerminalReporter', exitstatus: int, config: 'pytest.Config'):
    """Hook ad the end of Terminal Summary to log results
    
    Args:
        terminalreporter: pytest TerminalReporter object
        exitstatus: pytest exit status code
        config: pytest Config object (unused)
    """
    # silence pylint
    _ = config

    def _get_stat_count(stats: dict[str, list[Any]], key: str) -> int:
        """Helper function to extract total number of tests from pytest stats dict
        
        Args:
            stats: pytest stats dictionary, e.g. terminalreporter.stats
            key: name of stat, e.g. passed, failed, error etc.
        """
        # collect requested stat
        results = tuple(x for x in stats.get(key, ())
                        # apparently, there can be stats that are omitted from summary
                        # so omit them too, so log output and pytest output match
                        if getattr(x, 'count_towards_summary', True)
                        )
        # only interested in number of stats
        return len(results)

    # compute stat counts
    keys = ('passed', 'failed', 'skipped', 'xfailed', 'xpassed', 'error')
    count = {key: _get_stat_count(terminalreporter.stats, key) for key in keys}

    logger.info('Pytest Summary Report:')
    logger.info(f'Exit Status: {exitstatus}')
    logger.info(f'Total #Tests: {sum(count.values())}')
    logger.info(f'Passed: {count["passed"]} (X: {count["xpassed"]})')
    logger.info(f'Failed: {count["failed"]} (X: {count["xfailed"]})')
    logger.info(f'Skipped: {count["skipped"]}')
    logger.info(f'Errors: {count["error"]}')
