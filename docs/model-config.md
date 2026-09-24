# Model configuration

fotonet exposes one canonical model ID: `fotonete`. No aliases, P2 variants, or
arbitrary YAML layer graphs are accepted.

```yaml
nc: 80
profile: e
p2: false
reg_max: 12
quality_head: false
architecture_schema: 3
```

`profile` is fixed to the reviewed `e` channel layout. `p2` is always `false`.
`reg_max` selects the declared distributional localization width, and
`quality_head` is fixed to the declared score-fusion setting.

The packaged `fotonete` model uses `reg_max: 12` and does not expose arbitrary
regression or head overrides.

## Schema history

`architecture_schema: 3` denotes the all-dense fotonete graph and its explicit
integer-channel contract. Schema-1/2 checkpoints have different tensor shapes
and are rejected with an explicit message rather than partially loaded.

Normalization produces an exact `model_id`, full graph
`architecture_fingerprint`, backbone/neck output channels, and feature strides.
The fingerprint includes the fixed `e` profile, disabled P2 setting,
regression/quality settings, and class count. Checkpoint loading cross-validates
this identity before strict tensor loading; it never guesses a graph from shapes.
