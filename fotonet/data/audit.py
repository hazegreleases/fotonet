"""Dataset annotation auditing.

Object-detection datasets have a particularly nasty failure mode: a typo in a
label path silently turns a positive image into a background image.  The
dataset loader uses this module to make that state explicit and configurable.
"""
from __future__ import annotations

from collections import Counter
from threading import Lock
import warnings


class AnnotationError(ValueError):
    """Raised when an annotation violates the configured dataset policy."""


class AnnotationAudit:
    """Thread-safe, bounded record of annotation/source problems.

    ``policy`` is deliberately explicit:

    - ``"error"`` rejects malformed annotations immediately.
    - ``"warn"`` keeps loading while emitting a bounded warning stream.
    - ``"ignore"`` keeps loading but still records every issue in ``summary``.
    - ``"clamp"`` repairs out-of-bounds boxes but remains strict for all other
      fatal annotation problems.
    - ``"fix"`` repairs or drops invalid rows, records the totals, and never
      aborts loading for an annotation-quality issue. ``"repair"`` and
      ``"sanitize"`` are accepted as compatibility aliases.

    In ``"fix"`` mode, missing label files are recorded and treated as negative
    images. Strict policies require ``allow_missing_labels=True`` to accept
    them. An empty *existing* label file remains the unambiguous YOLO-format
    representation of a negative image.
    """

    _POLICIES = {"error", "warn", "ignore", "clamp", "fix"}
    _POLICY_ALIASES = {"repair": "fix", "sanitize": "fix"}
    _CLAMP_REPAIRABLE = {"box_extends_outside_image"}

    def __init__(self, policy="fix", allow_missing_labels=False, max_examples=20):
        policy = str(policy or "fix").lower()
        policy = self._POLICY_ALIASES.get(policy, policy)
        if policy not in self._POLICIES:
            accepted = sorted(self._POLICIES | set(self._POLICY_ALIASES))
            raise ValueError(f"annotation_policy must be one of {accepted}, got {policy!r}")
        self.policy = policy
        self.allow_missing_labels = bool(allow_missing_labels)
        self.max_examples = max(int(max_examples or 0), 0)
        self._counts = Counter()
        self._examples = []
        self._lock = Lock()

    @staticmethod
    def _message(kind, path, line=None, detail=None):
        location = str(path)
        if line is not None:
            location = f"{location}:{int(line)}"
        suffix = f" ({detail})" if detail else ""
        return f"Dataset annotation issue [{kind}] at {location}{suffix}"

    def issue(self, kind, path, line=None, detail=None, fatal=True):
        """Record one issue and apply the selected policy.

        Returns the formatted message for callers which need to add context.
        """
        message = self._message(kind, path, line=line, detail=detail)
        with self._lock:
            self._counts[str(kind)] += 1
            if len(self._examples) < self.max_examples:
                self._examples.append(message)

        strict_failure = self.policy == "error" or (
            self.policy == "clamp" and str(kind) not in self._CLAMP_REPAIRABLE
        )
        if fatal and strict_failure:
            raise AnnotationError(message)
        if self.policy == "warn":
            # Avoid turning a damaged large dataset into thousands of warnings.
            with self._lock:
                should_warn = self._counts[str(kind)] <= self.max_examples
            if should_warn:
                warnings.warn(message, RuntimeWarning, stacklevel=2)
        return message

    def summary(self):
        with self._lock:
            return {
                "policy": self.policy,
                "allow_missing_labels": self.allow_missing_labels,
                "counts": dict(sorted(self._counts.items())),
                "examples": list(self._examples),
            }

    def restore_summary(self, summary):
        """Restore cached issue counts without changing the active policy."""
        counts = dict((summary or {}).get("counts") or {})
        examples = list((summary or {}).get("examples") or [])
        with self._lock:
            self._counts = Counter(
                {
                    str(kind): int(count)
                    for kind, count in counts.items()
                    if int(count) > 0
                }
            )
            self._examples = [str(example) for example in examples[: self.max_examples]]

    def __getstate__(self):
        """Serialize audit data without the non-picklable thread lock."""
        with self._lock:
            return {
                "policy": self.policy,
                "allow_missing_labels": self.allow_missing_labels,
                "max_examples": self.max_examples,
                "counts": dict(self._counts),
                "examples": list(self._examples),
            }

    def __setstate__(self, state):
        """Recreate synchronization when loaded by a spawned data worker."""
        self.__init__(
            policy=state["policy"],
            allow_missing_labels=state["allow_missing_labels"],
            max_examples=state["max_examples"],
        )
        self.restore_summary(state)

    @property
    def has_issues(self):
        with self._lock:
            return bool(self._counts)
