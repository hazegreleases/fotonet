# Release checklist

## Source release

- [ ] The private non-training regression suite passes against the release tree.
- [ ] `python -m compileall -q fotonet examples` succeeds.
- [ ] Public docs/examples contain no local absolute paths.
- [ ] Dataset, run, checkpoint, export, and cache artifacts are ignored.
- [ ] The downloadable `train.py --dry-run` accepts the active checkpoint and COCO YAML.
- [ ] Active-checkpoint model/optimizer/scaler/EMA/RNG reconstruction succeeds
  without entering the training loop.
- [ ] `python -m build` and `twine check dist/*` succeed.
- [ ] The built wheel installs in a clean environment; imports, CLI help,
  checkpoint inference, and the transform example succeed there.
- [ ] The GitHub-bound diff contains no datasets, runs, weights, exports,
  credentials, private planning documents, or generated build artifacts.

## Weight release (after training)

- [ ] The selected inference checkpoint is self-identifying and tensor-only
  loadable.
- [ ] The checkpoint is attached to GitHub Releases, not committed to Git.
- [ ] SHA256 checksum and byte size are published.
- [ ] Canonical COCO val2017 evaluation passes the private release validator and
  its exact command/output are retained with the release evidence.
- [ ] Any AP claim cites the released weight, dataset split, image size,
  max-detections policy, and evaluator backend.
- [ ] Automatic model-name download points to the immutable release asset and
  verifies its checksum before loading.

Source publication does not imply a weight or AP release. Official weights and
automatic download remain pending while training is in progress.
