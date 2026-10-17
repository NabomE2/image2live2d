# PR #3 Ruff audit

Compared `main` (`fe72d4c441ea3832e6ce4ddfa658eb30949010c2`) with the
PR implementation (`8c21d21dec0c6941481f93f021833e82bc962b31`) using Ruff 0.17.1
and the CI command `ruff check src tests`.

| Check | main | PR #3 |
| --- | ---: | ---: |
| Ruff diagnostics | 178 | 178 |
| Affected files | 76 | 76 |
| Newly introduced diagnostics | — | 0 |
| Resolved diagnostics | — | 0 |

All diagnostics match by repository-relative file path, rule, line, column and
message. The 178 findings are existing main-branch lint debt, not new low_vram
errors. The largest groups are I001 (65), UP037 (29), RUF059 (16), RUF100 (16),
and RUF022 (12); the remaining rules account for 40 findings.

The CI lint job failed, although its existing `continue-on-error: true` setting
allowed the overall workflow to succeed. This audit does not change that setting,
ignore rules, suppress diagnostics or apply automatic fixes.

All Python files changed by PR #3 pass the same Ruff version:

```bash
ruff check service/seethrough/app.py service/seethrough/staged_inference.py \
  service/seethrough/low_vram.py tests/test_low_vram.py tests/test_seethrough_memory.py
```

No lint code changes were needed. Inference, application behavior, tests and CI
configuration remain unchanged by this audit commit.

## Reproduce the comparison

From the repository root, with Ruff 0.17.1 installed:

```bash
baseline=$(mktemp -d)
git archive fe72d4c441ea3832e6ce4ddfa658eb30949010c2 | tar -x -C "$baseline"
ruff check "$baseline/src" "$baseline/tests" --output-format json > /tmp/ruff-main.json
ruff check src tests --output-format json > /tmp/ruff-pr.json
```

Both Ruff commands intentionally exit 1 while the existing findings remain.
Strip the respective root prefixes from `filename` and compare the multiset of
`filename`, `code`, `location.row`, `location.column` and `message` from the JSON
outputs. Do not compare absolute paths or treat the nonzero exit as a new finding.

## Validation after audit

- Full test suite: **577 passed, 10 skipped**, with two existing FastAPI
  deprecation warnings. Skips require unavailable corpus, proprietary Core/models
  or a real CUDA inference service.
- Full `ruff check src tests`: **178 existing findings**, exit 1.
- Ruff on all five Python files changed by PR #3: **passed**.
- Application, inference, test and CI files: no changes in this audit commit.
