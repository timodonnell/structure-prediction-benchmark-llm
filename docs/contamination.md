# Contamination limits

This standalone extraction retains the original canary GUID and frozen manifest
withholding notices. Generated gold and prompts stay outside Git; the source-input
release deliberately makes their deterministic reconstruction possible. Neither
withholding plaintext from Git nor a canary establishes a secret holdout.

F02's release-date filter and sequence clustering do not establish separation
from model training data. See [the scientific protocol](protocol.md#contamination-and-interpretation)
and [the F02 provenance audit](f02-provenance.md#training-overlap).
