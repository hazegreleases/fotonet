# Security

Report vulnerabilities through GitHub private security advisories. Do not put
credentials, private checkpoints, or exploit details in public issues.

Native checkpoints are loaded with PyTorch's tensor-only mode. Supported files
must carry explicit checkpoint, architecture, model, and (for resume) training
protocol identity. Missing/unknown identities fail closed; filenames, state
keys, and tensor shapes are never used to guess a graph. There is no public
unsafe-pickle override.

ONNX and TorchScript files are executable programs. Treat them, their required
metadata, calibration inputs, and export toolchains as trusted build inputs.
