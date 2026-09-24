# Models and runtime

`fotonete` is the supported model ID in this release. Load a compatible
checkpoint before running inference; constructing a model without weights only
creates an untrained network.

## Model facts

| Property | Value |
| --- | --- |
| Model ID | `fotonete` |
| Input size used by the packaged model | `640 x 640` |
| Classes | `80` COCO classes |
| Training parameters | `1,723,672` |
| Deployment parameters | `1,698,564` |
| Checkpoint format | PyTorch `.pt` |

The model configuration is fixed for this release. See
[model configuration](model-config.md) for the accepted settings and checkpoint
compatibility rules.

## Runtime measurements

The website benchmark page publishes the current measurements together with the
hardware and software used for the run. Timing, memory, and cold-start values
are averages of 15 independent measurements. COCO mAP is evaluated once on the
published image sample because the same fixed evaluation is deterministic.

For application code, start with the [quickstart](quickstart.md) and
[inference guide](inference.md). For deployment artifacts, see the
[export guide](export.md).