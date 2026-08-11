# Transform zone system

This project turns tracked detections into durable enter and exit events.
Geometry is the primary decision boundary: the bottom anchor approximates
floor contact, BoxTransform defines the normalized zone, and the state
machine emits only edge transitions.

## Run

```bash
python -m zone_system.main --config config/zones.toml
```

## Test

```bash
python -m pytest
```

The example assumes a trusted native checkpoint and an ordered video source.
It does not claim re-identification across camera cuts or long occlusions.
