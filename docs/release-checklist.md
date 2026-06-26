# Alpha Release Checklist

- [ ] Full tests pass on a clean clone.
- [ ] README contains no emojis.
- [ ] README hero image loads from the raw banner URL.
- [ ] `.gitignore` excludes datasets, logs, weights, exports, and caches.
- [ ] No local absolute paths appear in public docs or examples.
- [ ] `python -m build` succeeds.
- [ ] `twine check dist/*` succeeds.
- [ ] Editable install works.
- [ ] CLI help works.
- [ ] At least one inference example runs.
- [ ] At least one transform example runs.
- [ ] Weights are attached to GitHub Releases, not committed.
- [ ] SHA256 checksums are published for release weights.
- [ ] GitHub release notes state alpha limitations.
