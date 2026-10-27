# Archived HEARD-side rule experiment — metrics removed

> **Non-compliant.** This experiment tuned collision-recovery and STOP-confidence rules against validation after analyzing a prior rule-modified HEARD test. The affected evaluation outputs have been deleted. No SR/SPL metrics are retained here, and this report must not be used for benchmark claims or model selection. See [the cleanup notice](result_avnav/NON_COMPLIANT_RULE_BASED_EVALUATIONS.md).

This archive is retained only to document that a prohibited inference-time action adapter was explored. The former validation table, full-test episode analysis, and recommendation for a recovery/STOP configuration have all been withdrawn. The surviving IQL checkpoint is a train artifact only; it has not been reselected or evaluated under the raw-policy protocol.

Any follow-up must select checkpoints on validation with raw actor argmax, audit training-source episode provenance, then evaluate complete heard and unheard test splits under the same frozen raw-action protocol.
