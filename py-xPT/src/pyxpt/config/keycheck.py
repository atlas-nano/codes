"""Warn about control-file keys that py-xPT does not recognize.

py-xPT reads its INI by testing for each key it knows about (``if "foo" in s:``),
so a key it does not know is silently discarded.  A deck can therefore carry a
setting its author believes is active while the run ignores it, with no warning
and a plausible-looking result.

This module records which keys the parsers actually touch and reports the rest.
The check is self-maintaining: adding a new key to a parser makes it recognized
automatically, with no list to keep in sync.

It is advisory.  Unknown keys are reported, never fatal, and any failure inside
the checker is swallowed so that it can never break a run that would otherwise
have succeeded.
"""

from __future__ import annotations

import configparser
import logging

log = logging.getLogger(__name__)

# Keys accepted anywhere without being read by a parser: comments and the
# section-level defaults configparser itself injects.
_ALWAYS_OK = frozenset({"comment", "note"})


class _RecordingSection:
    """A SectionProxy that remembers which keys were asked for."""

    def __init__(self, section: configparser.SectionProxy, seen: set) -> None:
        object.__setattr__(self, "_section", section)
        object.__setattr__(self, "_seen", seen)

    def _mark(self, key):
        try:
            self._seen.add(str(key).strip().lower())
        except Exception:
            pass

    def __contains__(self, key) -> bool:
        self._mark(key)
        return key in self._section

    def __getitem__(self, key):
        self._mark(key)
        return self._section[key]

    def __setitem__(self, key, value):
        self._mark(key)
        self._section[key] = value

    def __iter__(self):
        # Wholesale iteration: we cannot tell which keys are used, so treat
        # every key as read rather than emit false warnings.
        for key in self._section:
            self._mark(key)
        return iter(self._section)

    def get(self, key, *a, **kw):
        self._mark(key)
        return self._section.get(key, *a, **kw)

    def getint(self, key, *a, **kw):
        self._mark(key)
        return self._section.getint(key, *a, **kw)

    def getfloat(self, key, *a, **kw):
        self._mark(key)
        return self._section.getfloat(key, *a, **kw)

    def getboolean(self, key, *a, **kw):
        self._mark(key)
        return self._section.getboolean(key, *a, **kw)

    def __getattr__(self, name):
        return getattr(self._section, name)


class RecordingParser:
    """Wraps a ConfigParser so every key the parsers touch is recorded."""

    def __init__(self, parser: configparser.ConfigParser) -> None:
        self._parser = parser
        self._seen: dict[str, set] = {}

    def __getitem__(self, name):
        seen = self._seen.setdefault(str(name).strip().lower(), set())
        return _RecordingSection(self._parser[name], seen)

    def __contains__(self, name):
        return name in self._parser

    def __getattr__(self, name):
        return getattr(self._parser, name)

    def unknown_keys(self) -> dict[str, list]:
        """Return {section: [keys present but never read]}."""
        out: dict[str, list] = {}
        for name in self._parser.sections():
            key = name.strip().lower()
            present = {k.strip().lower() for k in self._parser[name]}
            # configparser injects DEFAULT-section keys into every section.
            present -= {k.strip().lower() for k in self._parser.defaults()}
            leftover = sorted(present - self._seen.get(key, set()) - _ALWAYS_OK)
            if leftover:
                out[name] = leftover
        return out


def warn_unknown_keys(parser) -> dict[str, list]:
    """Log a warning for each control-file key py-xPT did not read."""
    try:
        unknown = parser.unknown_keys()
    except Exception:                                    # never break a run
        log.debug("unknown-key check failed", exc_info=True)
        return {}
    for section, keys in unknown.items():
        for key in keys:
            log.warning("[%s] %s is not a py-xPT keyword and was IGNORED. "
                        "Check the spelling, or remove it: a key that is not "
                        "read has no effect on the calculation.", section, key)
    return unknown
