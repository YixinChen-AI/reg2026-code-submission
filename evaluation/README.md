# Evaluation

The challenge organizers maintain the scoring implementation in the REG2026
repository. It is referenced at a fixed commit rather than copied here because
the upstream repository does not declare a source-code license.

```bash
bash evaluation/bootstrap_official_scorer.sh
cd .cache/REG2026/submission_evaluation_code
```

Follow the scorer README to prepare predictions and ground truth, then run:

```bash
bash do_test_run.sh
```

The scorer evaluates workflow reasoning and visual grounding. The local
contract tests in `tests/` check output shape and fallback behavior but do not
replace the official scorer.
