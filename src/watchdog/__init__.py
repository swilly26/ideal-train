"""Watchdog policy layer.

Holds the restart decision logic for the bash supervisor
(``watchdog.sh``) in testable Python, plus the market-hours
helpers used to exempt stale-log kills while the market is closed.
"""