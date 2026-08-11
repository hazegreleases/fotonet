# Model configuration

fotonet exposes ten canonical model IDs: N/S/M/L/X, each with an optional P2
variant. No aliases or arbitrary YAML layer graphs are accepted.

```yaml
nc: 80
profile: n
p2: false
reg_max: 12
quality_head: false
architecture_schema: 1
```

`profile` selects reviewed integer channels, stage depths, downsampling modes,
and neck widths. `p2: true` adds stride-4 neck/head output. `reg_max` selects
direct or distributional localization width, and `quality_head` selects the
declared score-fusion branch.

Normalization produces an exact `model_id`, full graph
`architecture_fingerprint`, backbone/neck output channels, and feature strides.
The fingerprint includes profile, P2, regression/quality settings, and class
count. Checkpoint loading cross-validates this identity before strict tensor
loading; it never guesses a graph from shapes.
