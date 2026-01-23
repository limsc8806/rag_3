# Contributing Guide

## Documentation Update Rule

When you change ingest/index/retrieval/agent/generation behavior or data
formats, update `docs/ARCHITECTURE.md` in the same change.

## Regression Tests

If retrieval behavior or schema changes, review and update:

- `tests/regression_questions.yaml`
- `tests/test_regression.py`

## Quick Checks

- `python -m unittest tests.test_regression`
