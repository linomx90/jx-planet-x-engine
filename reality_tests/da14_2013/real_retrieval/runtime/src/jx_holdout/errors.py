class JXHoldoutError(RuntimeError):
    """Base fail-closed error for the sealed-holdout tool."""


class ValidationError(JXHoldoutError):
    """Input or contract validation failed."""


class IntegrityError(JXHoldoutError):
    """A hash, signature-equivalent receipt, or encrypted record failed verification."""


class StateTransitionError(JXHoldoutError):
    """An operation was attempted in the wrong curation/seal state."""
