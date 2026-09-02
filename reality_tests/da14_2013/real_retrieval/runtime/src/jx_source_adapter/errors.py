class SourceAdapterError(Exception):
    """Base fail-closed adapter error."""


class PolicyError(SourceAdapterError):
    pass


class TransportError(SourceAdapterError):
    pass


class SourceSchemaError(SourceAdapterError):
    pass


class IntegrityError(SourceAdapterError):
    pass
