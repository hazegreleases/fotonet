"""Inference adapters for exported FOTO-NET artifacts."""
from __future__ import annotations

import json
from numbers import Integral
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn


def _positive_integer(value, *, field):
    """Return a strict positive integer used by serialized artifact metadata."""
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise RuntimeError(f"FOTO-NET artifact metadata field '{field}' must be a positive integer.")
    return int(value)


def _normalize_strides(strides, *, field="strides"):
    """Validate FOTO-NET's feature-stride geometry before it sizes CUDA buffers."""
    if not isinstance(strides, (list, tuple)) or not strides:
        raise RuntimeError(f"FOTO-NET artifact metadata field '{field}' must be a non-empty list of strides.")
    normalized = [_positive_integer(value, field=f"{field}[{index}]") for index, value in enumerate(strides)]
    if any(right <= left for left, right in zip(normalized, normalized[1:])):
        raise RuntimeError(f"FOTO-NET artifact metadata field '{field}' must be strictly increasing.")
    return normalized


def validate_artifact_metadata(metadata, expected_format):
    """Return a shallow, type-safe metadata copy for a known artifact format.

    Sidecars are part of the runtime contract: accepting strings such as
    ``\"false\"`` as truthy or a mismatched format can silently route tensors
    through the wrong static/dynamic preprocessing path.  Class-name semantics
    are intentionally validated by ``normalize_class_schema`` in the public
    model API, where the configured class count is also available.
    """
    if not isinstance(metadata, dict):
        raise RuntimeError("FOTO-NET artifact metadata must be a JSON object.")
    normalized = dict(metadata)
    if normalized.get("export_schema") != 1:
        raise RuntimeError("Artifact metadata has an unsupported or missing export_schema.")
    for field in ("model_id", "architecture_schema", "architecture_fingerprint"):
        if normalized.get(field) in (None, ""):
            raise RuntimeError(f"Artifact metadata is missing required field {field!r}.")
    if int(normalized["architecture_schema"]) != 1:
        raise RuntimeError("Artifact metadata has an unsupported architecture_schema.")
    fmt = normalized.get("format")
    if fmt is not None and fmt != expected_format:
        raise RuntimeError(
            f"Artifact metadata format={fmt!r} does not match the loaded {expected_format!r} artifact."
        )
    if "imgsz" in normalized:
        imgsz = normalized["imgsz"]
        if not isinstance(imgsz, (list, tuple)) or len(imgsz) != 2:
            raise RuntimeError("FOTO-NET artifact metadata field 'imgsz' must be a [height, width] pair.")
        normalized["imgsz"] = [
            _positive_integer(value, field=f"imgsz[{index}]") for index, value in enumerate(imgsz)
        ]
    if "batch" in normalized:
        normalized["batch"] = _positive_integer(normalized["batch"], field="batch")
    if "nc" in normalized:
        normalized["nc"] = _positive_integer(normalized["nc"], field="nc")
    if "dynamic" in normalized:
        if not isinstance(normalized["dynamic"], bool):
            raise RuntimeError("FOTO-NET artifact metadata field 'dynamic' must be a boolean.")
    if "batch_dynamic" in normalized and not isinstance(normalized["batch_dynamic"], bool):
        raise RuntimeError("FOTO-NET artifact metadata field 'batch_dynamic' must be a boolean.")
    if "half" in normalized and not isinstance(normalized["half"], bool):
        raise RuntimeError("FOTO-NET artifact metadata field 'half' must be a boolean.")
    if "strides" in normalized:
        normalized["strides"] = _normalize_strides(normalized["strides"])
    return normalized


class TorchScriptRuntime(nn.Module):
    """TorchScript adapter exposing the small model interface used by predict."""

    def __init__(self, path, nc, device):
        super().__init__()
        extra_files = {"fotonet_metadata.json": ""}
        self.module = torch.jit.load(str(path), map_location=device, _extra_files=extra_files).eval()
        raw_metadata = extra_files["fotonet_metadata.json"]
        if raw_metadata:
            try:
                decoded = raw_metadata.decode("utf-8") if isinstance(raw_metadata, bytes) else str(raw_metadata)
                metadata = json.loads(decoded)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    "TorchScript artifact contains invalid embedded FOTO-NET metadata. "
                    "Re-export it with FOTONET.export(..., format='torchscript')."
                ) from exc
            metadata = validate_artifact_metadata(metadata, "torchscript")
        else:
            metadata = {}
        self.metadata = metadata
        artifact_imgsz = metadata.get("imgsz")
        self.input_imgsz = (
            (int(artifact_imgsz[0]), int(artifact_imgsz[1]))
            if isinstance(artifact_imgsz, (list, tuple))
            and len(artifact_imgsz) == 2
            and all(isinstance(value, Integral) and int(value) > 0 for value in artifact_imgsz)
            else None
        )
        self.dynamic_input = metadata.get("dynamic")
        self.input_batch = metadata.get("batch")
        self.dynamic_batch = metadata.get("batch_dynamic")
        self.head = SimpleNamespace(nc=int(metadata.get("nc", nc)), has_o2m=False)

    def _validate_input_contract(self, x):
        if not torch.is_tensor(x) or x.ndim != 4 or int(x.shape[1]) != 3:
            raise ValueError("FOTO-NET TorchScript runtime expects an NCHW tensor with shape [B,3,H,W].")
        if int(x.shape[0]) <= 0:
            raise ValueError("FOTO-NET TorchScript runtime does not accept an empty input batch.")
        if self.dynamic_input is False and self.input_imgsz is not None:
            actual_imgsz = tuple(int(value) for value in x.shape[-2:])
            if actual_imgsz != self.input_imgsz:
                raise ValueError(
                    f"This static TorchScript artifact expects HxW={self.input_imgsz}, not {actual_imgsz}."
                )

    def _forward_static_batched(self, x):
        """Safely adapt old/fixed-batch traces without claiming dynamic batch support."""
        outputs = []
        expected = int(self.input_batch)
        for start in range(0, int(x.shape[0]), expected):
            chunk = x[start:start + expected]
            actual = int(chunk.shape[0])
            if actual < expected:
                padded = torch.zeros((expected, *chunk.shape[1:]), device=chunk.device, dtype=chunk.dtype)
                padded[:actual].copy_(chunk)
                chunk = padded
            output = self.module(chunk)
            if not torch.is_tensor(output) or output.ndim != 3 or int(output.shape[0]) != expected:
                shape = tuple(output.shape) if torch.is_tensor(output) else type(output).__name__
                raise RuntimeError(
                    f"TorchScript artifact returned invalid fixed-batch output {shape}; expected [B,N,nc+4]."
                )
            outputs.append(output[:actual])
        return torch.cat(outputs, dim=0)

    def forward(self, x, **_kwargs):
        self._validate_input_contract(x)
        # New exports prove this separately with a second batch size.  Older
        # traces lack that proof, so use the safe fixed-batch adapter instead
        # of assuming a Python trace retained a symbolic batch dimension.
        if (
            self.input_batch is not None
            and self.dynamic_batch is not True
            and int(x.shape[0]) != self.input_batch
        ):
            return self._forward_static_batched(x)
        return self.module(x)


class ONNXRuntime(nn.Module):
    """ONNX Runtime adapter for raw FOTO-NET O2O exports."""

    def __init__(self, path, nc, device, strides=None):
        super().__init__()
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("Loading ONNX artifacts requires onnxruntime. Install `fotonet[export]` plus onnxruntime.") from exc
        self._ort = ort
        self.path = str(path)
        self.device = torch.device(device)
        self.session = None
        self.input_name = None
        self.output_name = None
        self._strides_from_metadata = strides is not None
        self.head = SimpleNamespace(
            nc=int(nc),
            has_o2m=False,
            strides=_normalize_strides(strides) if strides is not None else [8, 16, 32],
        )
        self._last_io_binding_used = False
        self._cuda_stream_handle = None
        self.input_batch = None
        self.dynamic_batch = None
        self._resolved_output_nc = None
        self._create_session(self.device)

    def _create_session(self, device):
        """Create a provider-matched ORT session, failing clearly when absent."""
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda:0")
        available = set(self._ort.get_available_providers())
        if device.type == "cuda":
            if "CUDAExecutionProvider" not in available:
                raise RuntimeError(
                    "device='cuda' requires an onnxruntime build with CUDAExecutionProvider; "
                    "install onnxruntime-gpu or use device='cpu'."
                )
            # Bind the EP to PyTorch's active stream.  I/O binding otherwise
            # exposes raw CUDA pointers across two independent streams, which
            # can race when callers use ``torch.cuda.stream(...)``.  ORT
            # documents this as the zero-copy integration path.
            stream_handle = int(torch.cuda.current_stream(device).cuda_stream)
            self._cuda_stream_handle = stream_handle
            providers = [
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": device.index,
                        "user_compute_stream": str(stream_handle),
                        "do_copy_in_default_stream": "1",
                    },
                ),
                "CPUExecutionProvider",
            ]
        elif device.type == "cpu":
            providers = ["CPUExecutionProvider"]
            self._cuda_stream_handle = None
        else:
            raise ValueError(f"ONNX Runtime supports device='cpu' or 'cuda', got '{device}'.")
        self.session = self._ort.InferenceSession(self.path, providers=providers)
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise RuntimeError(
                "FOTO-NET ONNX artifacts must expose exactly one input and one raw O2O output."
            )
        input_meta = inputs[0]
        self.input_name = input_meta.name
        self.output_name = outputs[0].name
        input_type = input_meta.type
        shape = tuple(input_meta.shape)
        if len(shape) != 4:
            raise RuntimeError(
                f"Invalid FOTO-NET ONNX input shape {shape}; expected NCHW [B,3,H,W]."
            )
        input_channels = shape[1]
        if isinstance(input_channels, Integral) and int(input_channels) != 3:
            raise RuntimeError(
                f"Invalid FOTO-NET ONNX input shape {shape}; expected three RGB channels."
            )
        self.input_batch = int(shape[0]) if isinstance(shape[0], Integral) and int(shape[0]) > 0 else None
        self.dynamic_batch = self.input_batch is None
        self.input_imgsz = (
            (int(shape[2]), int(shape[3]))
            if len(shape) == 4 and isinstance(shape[2], Integral) and isinstance(shape[3], Integral)
            else None
        )
        self.dynamic_input = self.input_imgsz is None
        self.input_type = input_type
        self.input_numpy_dtype = {
            "tensor(float)": np.float32,
            "tensor(float16)": np.float16,
        }.get(input_type)
        if self.input_numpy_dtype is None:
            raise RuntimeError(
                f"Unsupported ONNX input type '{input_type}'. FOTO-NET runtime expects tensor(float) or tensor(float16)."
            )
        output_meta = outputs[0]
        output_type = output_meta.type
        self.output_numpy_dtype = {
            "tensor(float)": np.float32,
            "tensor(float16)": np.float16,
        }.get(output_type)
        if self.output_numpy_dtype is None:
            raise RuntimeError(
                f"Unsupported ONNX output type '{output_type}'. FOTO-NET runtime expects tensor(float) or tensor(float16)."
            )
        self.input_torch_dtype = torch.float16 if self.input_numpy_dtype == np.float16 else torch.float32
        self.output_torch_dtype = torch.float16 if self.output_numpy_dtype == np.float16 else torch.float32
        output_shape = tuple(output_meta.shape)
        if len(output_shape) != 3:
            raise RuntimeError(
                f"Invalid FOTO-NET ONNX output shape {output_shape}; expected raw [B,N,nc+4]."
            )
        output_batch = output_shape[0]
        if (
            self.input_batch is not None
            and isinstance(output_batch, Integral)
            and int(output_batch) != self.input_batch
        ):
            raise RuntimeError(
                f"Invalid FOTO-NET ONNX batch contract: input batch={self.input_batch}, "
                f"output batch={int(output_batch)}."
            )
        last_dim = output_shape[-1] if output_shape else None
        graph_output_nc = int(last_dim) - 4 if isinstance(last_dim, Integral) and int(last_dim) > 4 else None
        if (
            graph_output_nc is not None
            and self._resolved_output_nc is not None
            and graph_output_nc != self._resolved_output_nc
        ):
            raise RuntimeError(
                f"FOTO-NET ONNX output nc={graph_output_nc} conflicts with the resolved nc="
                f"{self._resolved_output_nc}."
            )
        self.output_nc = graph_output_nc if graph_output_nc is not None else self._resolved_output_nc
        if self.output_nc is not None:
            self.head.nc = self.output_nc
            self._resolved_output_nc = self.output_nc
        self.schema_pending = self.output_nc is None
        anchor_dim = output_shape[1] if len(output_shape) > 1 else None
        if self.input_imgsz is not None and isinstance(anchor_dim, Integral):
            expected_anchors = self._anchor_count(self.input_imgsz, self.head.strides)
            if self._strides_from_metadata and expected_anchors != int(anchor_dim):
                raise RuntimeError(
                    "FOTO-NET ONNX metadata strides do not match the graph output anchor count. "
                    "Restore the matching sidecar or re-export the artifact."
                )
            self._infer_strides_from_anchors(self.input_imgsz, int(anchor_dim))
        self.device = device

    def to(self, device):
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda:0")
        if device != self.device:
            self._create_session(device)
        return self

    def _ensure_current_cuda_stream(self, device):
        """Recreate a CUDA session only when callers deliberately switch streams."""
        if device.type != "cuda":
            return
        current_handle = int(torch.cuda.current_stream(device).cuda_stream)
        if current_handle != self._cuda_stream_handle:
            self._create_session(device)

    def eval(self):
        return self

    @staticmethod
    def _anchor_count(input_imgsz, strides):
        """Return FOTO-NET's anchor count after right/bottom stride padding."""
        h, w = (int(input_imgsz[0]), int(input_imgsz[1]))
        max_stride = max(int(stride) for stride in strides)
        h = ((h + max_stride - 1) // max_stride) * max_stride
        w = ((w + max_stride - 1) // max_stride) * max_stride
        return sum((h // int(stride)) * (w // int(stride)) for stride in strides)

    def _infer_strides_from_anchors(self, input_imgsz, anchors):
        """Recover P2-vs-P3 geometry from a metadata-less FOTO-NET ONNX graph."""
        if self._strides_from_metadata:
            return
        candidates = ([8, 16, 32], [4, 8, 16, 32])
        matches = [strides for strides in candidates if self._anchor_count(input_imgsz, strides) == int(anchors)]
        if len(matches) == 1:
            self.head.strides = matches[0]

    def _infer_output_schema(self, output):
        """Recover class count when a dynamic ONNX output's last dim is symbolic."""
        shape = tuple(output.shape)
        if not shape:
            return
        output_nc = int(shape[-1]) - 4
        if output_nc <= 0:
            raise RuntimeError(
                f"Invalid FOTO-NET ONNX output shape {shape}; expected final dimension nc + 4."
            )
        self.output_nc = output_nc
        self._resolved_output_nc = output_nc
        self.head.nc = output_nc
        self.schema_pending = False

    def set_output_schema(self, nc):
        """Persist a sidecar-provided class count across CUDA session recreation."""
        nc = _positive_integer(nc, field="nc")
        if self.output_nc is not None and self.output_nc != nc:
            raise RuntimeError(
                f"FOTO-NET ONNX graph output nc={self.output_nc} conflicts with metadata nc={nc}."
            )
        self._resolved_output_nc = nc
        self.output_nc = nc
        self.head.nc = nc
        self.schema_pending = False

    def _bound_output_shape(self, x):
        """Infer FOTO-NET's raw O2O output shape after internal stride padding."""
        h, w = (int(x.shape[-2]), int(x.shape[-1]))
        max_stride = max(self.head.strides)
        h = ((h + max_stride - 1) // max_stride) * max_stride
        w = ((w + max_stride - 1) // max_stride) * max_stride
        anchors = sum((h // stride) * (w // stride) for stride in self.head.strides)
        return int(x.shape[0]), int(anchors), int(self.head.nc) + 4

    def _run_cuda_iobinding(self, x):
        """Run CUDA EP directly on torch-owned GPU buffers, with no NumPy hop."""
        if x.device.type != "cuda" or self.device.type != "cuda" or self.schema_pending:
            return None
        x = x.contiguous()
        output = torch.empty(self._bound_output_shape(x), device=x.device, dtype=self.output_torch_dtype)
        binding = self.session.io_binding()
        binding.bind_input(
            self.input_name,
            "cuda",
            int(x.device.index or 0),
            self.input_numpy_dtype,
            tuple(x.shape),
            x.data_ptr(),
        )
        binding.bind_output(
            self.output_name,
            "cuda",
            int(x.device.index or 0),
            self.output_numpy_dtype,
            tuple(output.shape),
            output.data_ptr(),
        )
        self.session.run_with_iobinding(binding)
        self._last_io_binding_used = True
        return output

    def _validate_input_contract(self, x):
        if not torch.is_tensor(x) or x.ndim != 4 or int(x.shape[1]) != 3:
            raise ValueError("FOTO-NET ONNX runtime expects an NCHW tensor with shape [B,3,H,W].")
        if int(x.shape[0]) <= 0:
            raise ValueError("FOTO-NET ONNX runtime does not accept an empty input batch.")
        if self.input_imgsz is not None and tuple(int(value) for value in x.shape[-2:]) != self.input_imgsz:
            raise ValueError(
                f"This static ONNX artifact expects HxW={self.input_imgsz}, "
                f"not {tuple(int(value) for value in x.shape[-2:])}."
            )

    def _forward_exact_batch(self, input_tensor):
        self._last_io_binding_used = False
        if input_tensor.device.type == "cuda" and self.device.type == "cuda":
            try:
                bound_output = self._run_cuda_iobinding(input_tensor)
                if bound_output is not None:
                    self._infer_output_schema(bound_output)
                    return bound_output
            except Exception as exc:
                # Portable NumPy execution remains a correctness fallback for
                # unusual provider builds.  Surface it once rather than hiding
                # a performance regression forever.
                if not getattr(self, "_warned_iobinding_fallback", False):
                    print(f"[WARN] ONNX CUDA I/O binding unavailable; using host-copy fallback: {exc}")
                    self._warned_iobinding_fallback = True
        output = self.session.run([self.output_name], {self.input_name: input_tensor.cpu().numpy()})[0]
        if output.ndim >= 2:
            self._infer_strides_from_anchors(input_tensor.shape[-2:], output.shape[1])
        self._infer_output_schema(output)
        return torch.from_numpy(output).to(input_tensor.device)

    def _forward_static_batched(self, input_tensor):
        """Run arbitrary public batch sizes through a fixed-batch ONNX graph.

        ONNX static exports encode the example batch dimension, while the
        public predictor intentionally groups decoded sources.  Chunking and
        zero-padding the final chunk keeps the API batch-agnostic without
        incorrectly declaring a static graph dynamic.
        """
        outputs = []
        all_iobound = True
        expected = int(self.input_batch)
        for start in range(0, int(input_tensor.shape[0]), expected):
            chunk = input_tensor[start:start + expected]
            actual = int(chunk.shape[0])
            if actual < expected:
                padded = torch.zeros(
                    (expected, *chunk.shape[1:]), device=chunk.device, dtype=chunk.dtype
                )
                padded[:actual].copy_(chunk)
                chunk = padded
            output = self._forward_exact_batch(chunk)
            all_iobound = all_iobound and self._last_io_binding_used
            outputs.append(output[:actual])
        self._last_io_binding_used = all_iobound
        return torch.cat(outputs, dim=0)

    def forward(self, x, **_kwargs):
        self._validate_input_contract(x)
        if x.device != self.device:
            self.to(x.device)
        else:
            self._ensure_current_cuda_stream(x.device)
        input_tensor = x.detach().to(dtype=self.input_torch_dtype)
        if self.input_batch is not None and int(input_tensor.shape[0]) != self.input_batch:
            return self._forward_static_batched(input_tensor)
        return self._forward_exact_batch(input_tensor)
