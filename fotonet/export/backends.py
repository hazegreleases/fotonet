"""Export helpers for fotonet inference graphs."""

import json
import os
import shutil
import subprocess
from contextlib import contextmanager
from numbers import Integral
from pathlib import Path

import numpy as np
import torch


def _positive_integer(value, *, field):
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"{field} must be a positive integer.")
    return int(value)


def _batch_size(batch):
    return _positive_integer(batch, field="batch")


def _imgsz_pair(imgsz):
    if isinstance(imgsz, Integral) and not isinstance(imgsz, bool):
        h = w = int(imgsz)
    elif isinstance(imgsz, (tuple, list)) and len(imgsz) == 2:
        h = _positive_integer(imgsz[0], field="imgsz height")
        w = _positive_integer(imgsz[1], field="imgsz width")
    else:
        raise TypeError("imgsz must be an integer or an (height, width) pair.")
    if h <= 0 or w <= 0:
        raise ValueError("imgsz dimensions must be positive.")
    return h, w


@contextmanager
def _export_model_state(model_api, device, half):
    """Temporarily place a model in export mode and restore it even on failure."""
    model = model_api.model
    reference = next(model.parameters(), None)
    if reference is None:
        reference = next(model.buffers(), None)
    original_device = reference.device if reference is not None else torch.device("cpu")
    original_dtype = (
        reference.dtype
        if reference is not None and (reference.is_floating_point() or reference.is_complex())
        else None
    )
    original_training = model.training
    device = torch.device(device)
    use_half = bool(half and device.type == "cuda")
    try:
        model.to(device).eval()
        if use_half:
            model.half()
        else:
            model.float()
        yield model, use_half
    finally:
        if original_dtype is None:
            model.to(device=original_device)
        else:
            model.to(device=original_device, dtype=original_dtype)
        model.train(original_training)


def _metadata(model, fmt, imgsz, batch, dynamic, half, extra=None):
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        names = {str(k): str(v) for k, v in names.items()}
    elif isinstance(names, (list, tuple)):
        names = {str(index): str(value) for index, value in enumerate(names)}
    native_model = getattr(model, "model", None)
    head = getattr(native_model, "head", None)
    nc = getattr(model, "nc", None)
    if nc is None:
        nc = getattr(head, "nc", 80)
    strides = [int(value) for value in getattr(head, "strides", [8, 16, 32])]
    if not strides or any(value <= 0 for value in strides):
        raise ValueError("model head strides must be a non-empty sequence of positive integers.")
    data = {
        "export_schema": 1,
        "format": fmt,
        "imgsz": list(_imgsz_pair(imgsz)),
        "batch": _batch_size(batch),
        "dynamic": bool(dynamic),
        "half": bool(half),
        "nc": _positive_integer(nc, field="model nc"),
        "names": names,
        "model_id": str(getattr(model, "model_id")),
        "architecture_schema": int(getattr(model, "architecture_schema")),
        "architecture_fingerprint": str(getattr(model, "architecture_fingerprint")),
        "p2": bool(getattr(model, "use_p2", getattr(native_model, "use_p2", False))),
        "reg_max": int(getattr(model, "reg_max", getattr(head, "reg_max", 16))),
        "strides": strides,
        "output": "raw_o2o_tensor[B,N,nc+4]",
        "postprocess": "sigmoid logits, max class, confidence threshold, top-k cap",
        "input_layout": "NCHW",
        "input_color": "RGB",
        "input_range": "float32/float16 [0,1]",
        "letterbox": {"pad_value": 114, "box_coordinates": "normalized xywh relative to the caller input"},
        "stride_padding": {
            "max_stride": max(strides),
            "policy": "right_bottom_pad_to_stride_then_unpad_normalized_boxes",
        },
    }
    if extra:
        data.update(extra)
    return data


def _write_metadata(path, model, fmt, imgsz, batch, dynamic, half, extra=None):
    meta_path = Path(path).with_suffix(Path(path).suffix + ".metadata.json")
    with open(meta_path, "w") as f:
        json.dump(_metadata(model, fmt, imgsz, batch, dynamic, half, extra), f, indent=2)
    return str(meta_path)


def _verify_onnx_runtime(path, samples, *, prefer_cuda=False):
    """Verify the final ONNX file against saved native reference samples."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("[WARN] onnxruntime not installed; skipped ONNX Runtime parity check.")
        return
    available = set(ort.get_available_providers())
    if prefer_cuda and "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
        if prefer_cuda:
            print("[WARN] CUDAExecutionProvider is unavailable; checking the FP16 ONNX graph on CPU Runtime.")
    session = ort.InferenceSession(str(path), providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    for label, sample, reference in samples:
        runtime_out = session.run([output_name], {input_name: sample.detach().cpu().numpy()})[0]
        if runtime_out.shape != reference.shape or not np.allclose(runtime_out, reference, rtol=2e-3, atol=2e-4):
            max_error = float(np.max(np.abs(runtime_out - reference))) if runtime_out.shape == reference.shape else float("inf")
            raise RuntimeError(f"ONNX Runtime parity check failed for {label} input (max abs error={max_error}).")


def export_onnx(model_api, path, imgsz=640, batch=1, dynamic=False, half=False,
                simplify=True, opset=17, device=None, verify=True):
    device = device or model_api.device
    batch = _batch_size(batch)
    dynamic = bool(dynamic)
    h, w = _imgsz_pair(imgsz)
    axes = {
        "images": {0: "batch", 2: "height", 3: "width"},
        "output0": {0: "batch", 1: "anchors"},
    } if dynamic else None

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    export_kwargs = {
        "input_names": ["images"],
        "output_names": ["output0"],
        "dynamic_axes": axes,
        "opset_version": int(opset),
        "do_constant_folding": True,
    }
    with _export_model_state(model_api, device, half) as (model, use_half), torch.no_grad():
        dummy = torch.zeros(batch, 3, h, w, device=device, dtype=torch.float16 if use_half else torch.float32)
        reference = model(dummy).detach().float().cpu().numpy()
        samples = [(f"{h}x{w}", dummy, reference)]
        if dynamic:
            # Exercise batch and non-stride-aligned H/W changes together.
            # This catches frozen trace-time batch, resize, grid, and
            # pad/unpad branches rather than merely testing one symbolic axis.
            dynamic_dummy = torch.zeros(
                batch + 1, 3, h + 1, w + 3, device=device, dtype=dummy.dtype
            )
            dynamic_reference = model(dynamic_dummy).detach().float().cpu().numpy()
            samples.append((f"B{batch + 1}_{h + 1}x{w + 3}", dynamic_dummy, dynamic_reference))
        try:
            torch.onnx.export(model, dummy, path, dynamo=False, **export_kwargs)
        except TypeError:
            torch.onnx.export(model, dummy, path, **export_kwargs)

    try:
        import onnx
        exported = onnx.load(path)
        onnx.checker.check_model(exported)
    except ImportError:
        print("[WARN] onnx package not installed; skipped ONNX checker.")

    if simplify:
        try:
            import onnx
            import onnxsim
            exported = onnx.load(path)
            simplified, ok = onnxsim.simplify(exported)
            if ok:
                onnx.save(simplified, path)
            else:
                print("[WARN] onnxsim reported simplification was not valid; kept original graph.")
        except ImportError:
            print("[WARN] onnxsim not installed; skipped ONNX simplification.")

    # Check the final file, not merely the pre-simplification graph.
    try:
        import onnx
        onnx.checker.check_model(onnx.load(path))
    except ImportError:
        pass
    if verify:
        _verify_onnx_runtime(path, samples, prefer_cuda=use_half)

    meta = _write_metadata(path, model_api, "onnx", imgsz, batch, dynamic, use_half, {"opset": int(opset)})
    return {"artifact": str(path), "metadata": meta}


def export_torchscript(model_api, path, imgsz=640, batch=1, half=False, device=None, dynamic=False,
                       verify=True):
    """Export TorchScript and record whether its runtime contract permits dynamic H/W."""
    device = device or model_api.device
    batch = _batch_size(batch)
    h, w = _imgsz_pair(imgsz)
    dynamic = bool(dynamic)
    with (
        _export_model_state(model_api, device, half) as (model, use_half),
        torch.no_grad(),
    ):
        dummy = torch.zeros(batch, 3, h, w, device=device, dtype=torch.float16 if use_half else torch.float32)
        traced = torch.jit.trace(model, dummy, check_trace=bool(verify))
        reference = model(dummy)
        candidate = traced(dummy)
        if verify and not torch.allclose(reference, candidate, rtol=1e-3, atol=1e-4):
            raise RuntimeError("TorchScript export parity check failed.")
        # Batch dynamism is independent of H/W dynamism.  Verify and record it
        # even for a static-H/W export, so the runtime never guesses whether a
        # Python trace froze the example batch dimension.
        batch_probe = torch.zeros(batch + 1, 3, h, w, device=device, dtype=dummy.dtype)
        try:
            batch_reference = model(batch_probe)
            batch_candidate = traced(batch_probe)
            batch_dynamic = (
                batch_candidate.shape == batch_reference.shape
                and torch.allclose(batch_reference, batch_candidate, rtol=1e-3, atol=1e-4)
            )
        except Exception:
            batch_dynamic = False
        if dynamic:
            # Exercise tensor-derived pad/resize/grid operations rather than
            # merely serializing them. A non-stride-aligned rectangle catches
            # frozen anchor-grid and unpadding errors.
            dynamic_dummy = torch.zeros(
                batch + 1, 3, h + 1, w + 3, device=device, dtype=dummy.dtype
            )
            dynamic_reference = model(dynamic_dummy)
            dynamic_candidate = traced(dynamic_dummy)
            if dynamic_candidate.shape != dynamic_reference.shape or not torch.allclose(
                dynamic_reference, dynamic_candidate, rtol=1e-3, atol=1e-4
            ):
                max_error = (
                    float((dynamic_reference - dynamic_candidate).abs().max())
                    if dynamic_candidate.shape == dynamic_reference.shape
                    else float("inf")
                )
                raise RuntimeError(
                    "Dynamic TorchScript export parity check failed for "
                    f"{h + 1}x{w + 3} input (max abs error={max_error})."
                )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    embedded_metadata = _metadata(
        model_api, "torchscript", imgsz, batch, dynamic, use_half,
        {"batch_dynamic": bool(batch_dynamic)},
    )
    # Sidecars are convenient for non-Python tooling, but the archive embeds
    # the same contract so a copied ``.torchscript`` cannot silently lose
    # class count or static-vs-dynamic preprocessing requirements.
    traced.save(str(path), _extra_files={"fotonet_metadata.json": json.dumps(embedded_metadata)})
    meta = _write_metadata(
        path, model_api, "torchscript", imgsz, batch, dynamic, use_half,
        {"batch_dynamic": bool(batch_dynamic)},
    )
    return {"artifact": str(path), "metadata": meta}


def export_tensorrt(model_api, path, imgsz=640, batch=1, dynamic=False, half=True,
                    simplify=True, opset=17, workspace=4, device=None, min_imgsz=None, max_imgsz=None,
                    min_batch=None, max_batch=None, verify=True):
    trtexec = shutil.which("trtexec")
    if trtexec is None:
        raise RuntimeError("TensorRT export requires trtexec in PATH. Install TensorRT or export ONNX first.")

    batch = _batch_size(batch)
    dynamic = bool(dynamic)
    onnx_path = str(Path(path).with_suffix(".onnx"))
    export_onnx(model_api, onnx_path, imgsz=imgsz, batch=batch, dynamic=dynamic,
                half=half, simplify=simplify, opset=opset, device=device, verify=verify)
    workspace_arg = _trtexec_workspace_arg(trtexec, workspace)
    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={path}",
        workspace_arg,
        "--skipInference",
    ]
    if half:
        cmd.append("--fp16")
    profile_metadata = {}
    if dynamic:
        opt_h, opt_w = _imgsz_pair(imgsz)
        min_h, min_w = _imgsz_pair(min_imgsz or (max(32, opt_h // 2), max(32, opt_w // 2)))
        max_h, max_w = _imgsz_pair(max_imgsz or (opt_h * 2, opt_w * 2))
        min_batch = _batch_size(1 if min_batch is None else min_batch)
        max_batch = _batch_size(batch if max_batch is None else max_batch)
        if min_h > opt_h or min_w > opt_w or max_h < opt_h or max_w < opt_w:
            raise ValueError("TensorRT dynamic shape range must satisfy min <= opt <= max.")
        if min_batch > batch or max_batch < batch:
            raise ValueError("TensorRT dynamic batch range must satisfy min_batch <= batch <= max_batch.")
        min_shape = f"images:{min_batch}x3x{min_h}x{min_w}"
        opt_shape = f"images:{batch}x3x{opt_h}x{opt_w}"
        max_shape = f"images:{max_batch}x3x{max_h}x{max_w}"
        cmd.extend([f"--minShapes={min_shape}", f"--optShapes={opt_shape}", f"--maxShapes={max_shape}"])
        profile_metadata = {
            "min_imgsz": [min_h, min_w],
            "max_imgsz": [max_h, max_w],
            "min_batch": min_batch,
            "max_batch": max_batch,
        }
    subprocess.run(cmd, check=True)
    engine_path = Path(path)
    if not engine_path.is_file() or engine_path.stat().st_size <= 0:
        raise RuntimeError("trtexec exited successfully but did not produce a non-empty TensorRT engine.")
    meta = _write_metadata(
        path,
        model_api,
        "tensorrt",
        imgsz,
        batch,
        dynamic,
        half,
        {"source_onnx": onnx_path, **profile_metadata},
    )
    return {"artifact": str(path), "metadata": meta}


def _trtexec_workspace_arg(trtexec, workspace):
    workspace = max(int(workspace), 1)
    try:
        help_text = subprocess.run(
            [trtexec, "--help"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        combined = (help_text.stdout or "") + (help_text.stderr or "")
        if "--memPoolSize" in combined:
            return f"--memPoolSize=workspace:{workspace}G"
    except Exception:
        pass
    return f"--workspace={workspace * 1024}"


def export_coreml(model_api, path, imgsz=640, batch=1, half=True, device=None, coreml_format="auto"):
    try:
        import coremltools as ct
    except ImportError as exc:
        raise RuntimeError("CoreML export requires coremltools. Install coremltools first.") from exc

    # CoreML conversion is host-side.  Tracing on CUDA is slower to transfer
    # back into the converter and is not a portable CoreML export contract.
    device = torch.device("cpu")
    batch = _batch_size(batch)
    h, w = _imgsz_pair(imgsz)
    with _export_model_state(model_api, device, half) as (model, use_half), torch.no_grad():
        dummy = torch.zeros(batch, 3, h, w, device=device, dtype=torch.float16 if use_half else torch.float32)
        traced = torch.jit.trace(model, dummy, check_trace=True)

    requested = str(coreml_format or "auto").lower()
    if requested in ("mlpackage", "mlprogram"):
        formats = ["mlprogram"]
    elif requested in ("mlmodel", "neuralnetwork", "neural_network"):
        formats = ["neuralnetwork"]
    elif requested == "auto":
        suffix = Path(path).suffix.lower()
        # An explicit suffix is a user-visible contract.  Prefer it before
        # trying the other CoreML representation in automatic mode.
        formats = ["neuralnetwork", "mlprogram"] if suffix == ".mlmodel" else ["mlprogram", "neuralnetwork"]
    else:
        raise ValueError("coreml_format must be 'auto', 'mlprogram', or 'neuralnetwork'.")

    base_path = Path(path)
    last_exc = None
    for fmt in formats:
        out_path = base_path
        if fmt == "mlprogram" and out_path.suffix != ".mlpackage":
            out_path = out_path.with_suffix(".mlpackage")
        if fmt == "neuralnetwork" and out_path.suffix != ".mlmodel":
            out_path = out_path.with_suffix(".mlmodel")

        try:
            kwargs = {"inputs": [ct.TensorType(name="images", shape=dummy.shape)]}
            if fmt == "mlprogram":
                kwargs["convert_to"] = "mlprogram"
                kwargs["compute_precision"] = ct.precision.FLOAT16 if half else ct.precision.FLOAT32
            else:
                kwargs["convert_to"] = "neuralnetwork"

            mlmodel = ct.convert(traced, **kwargs)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            mlmodel.save(str(out_path))
            meta = _write_metadata(
                out_path,
                model_api,
                "coreml",
                imgsz,
                batch,
                False,
                bool(half and fmt == "mlprogram"),
                {"coreml_format": fmt},
            )
            return {"artifact": str(out_path), "metadata": meta}
        except Exception as exc:
            last_exc = exc
            if requested != "auto" or fmt == formats[-1]:
                break

    raise RuntimeError(f"CoreML export failed for formats {formats}. Last error: {last_exc}") from last_exc


def export_onnx_int8(model_api, path, calibration_data, imgsz=640, batch=1, dynamic=False,
                     simplify=True, opset=17, device=None, verify=True):
    """Produce calibrated QDQ INT8 ONNX from preprocessed NCHW calibration batches."""
    if calibration_data is None:
        raise ValueError(
            "INT8 ONNX export requires calibration_data: an iterable of NCHW tensors/arrays in RGB [0,1]."
        )
    try:
        from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    except ImportError as exc:
        raise RuntimeError("INT8 ONNX export requires onnxruntime. Install `fotonet[export]`.") from exc

    class _Reader(CalibrationDataReader):
        def __init__(self, batches):
            self._iterator = iter(batches)
            self.count = 0

        def get_next(self):
            try:
                value = next(self._iterator)
            except StopIteration:
                return None
            if isinstance(value, dict):
                value = value.get("images", value.get("image"))
            if torch.is_tensor(value):
                value = value.detach().float().cpu().numpy()
            value = np.asarray(value, dtype=np.float32)
            if value.ndim == 3:
                value = value[None]
            if value.ndim != 4 or value.shape[1] != 3:
                raise ValueError("Every INT8 calibration batch must have shape [B,3,H,W].")
            if not np.isfinite(value).all() or value.min() < 0.0 or value.max() > 1.0:
                raise ValueError("INT8 calibration data must contain finite RGB values in [0,1].")
            self.count += 1
            return {"images": value}

    batch = _batch_size(batch)
    dynamic = bool(dynamic)
    path = Path(path)
    float_path = path.with_name(path.stem + ".fp32.onnx")
    float_metadata_path = float_path.with_suffix(float_path.suffix + ".metadata.json")
    export_onnx(
        model_api,
        float_path,
        imgsz=imgsz,
        batch=batch,
        dynamic=dynamic,
        half=False,
        simplify=simplify,
        opset=opset,
        device=device,
        verify=verify,
    )
    reader = _Reader(calibration_data)
    try:
        quantize_static(
            str(float_path),
            str(path),
            reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
        )
    finally:
        if float_path.exists():
            float_path.unlink()
        if float_metadata_path.exists():
            float_metadata_path.unlink()
    if reader.count == 0:
        raise ValueError("INT8 calibration_data was empty; no quantized model was produced.")
    try:
        import onnx
        exported = onnx.load(str(path))
        onnx.checker.check_model(exported)
    except ImportError:
        pass
    meta = _write_metadata(
        path,
        model_api,
        "onnx",
        imgsz,
        batch,
        dynamic,
        False,
        {"opset": int(opset), "int8": True, "quantization": "onnxruntime-static-qdq", "calibration_batches": reader.count},
    )
    return {"artifact": str(path), "metadata": meta}


def _validate_export_path(path, fmt):
    allowed_suffixes = {
        "onnx": {".onnx"},
        "torchscript": {".torchscript", ".ts"},
        "tensorrt": {".engine"},
        "coreml": {".mlpackage", ".mlmodel"},
    }
    suffix = Path(path).suffix.lower()
    if suffix not in allowed_suffixes[fmt]:
        expected = ", ".join(sorted(allowed_suffixes[fmt]))
        raise ValueError(
            f"format={fmt!r} requires an artifact path ending in {expected}; got {os.fspath(path)!r}."
        )
    return os.fspath(path)


def _raise_unexpected_export_options(fmt, options):
    if options:
        names = ", ".join(sorted(options))
        raise TypeError(f"Unsupported {fmt} export option(s): {names}")


def export_model(model_api, path=None, format="onnx", imgsz=640, batch=1,
                 dynamic=False, half=False, simplify=True, int8=False,
                 opset=17, device=None, **kwargs):
    if not isinstance(format, str):
        raise TypeError("format must be a string.")
    fmt = format.lower()
    if fmt == "engine":
        fmt = "tensorrt"
    if fmt == "jit":
        fmt = "torchscript"
    if path is None:
        suffix = {"onnx": ".onnx", "tensorrt": ".engine", "coreml": ".mlpackage", "torchscript": ".torchscript"}.get(fmt, f".{fmt}")
        path = f"fotonet_export{suffix}"
    if fmt not in {"onnx", "torchscript", "tensorrt", "coreml"}:
        raise ValueError(f"Unsupported export format: {format}")
    path = _validate_export_path(path, fmt)
    verify = kwargs.pop("verify", True)

    if int8:
        if fmt != "onnx":
            raise ValueError("INT8 export is currently supported for calibrated ONNX only; use format='onnx'.")
        calibration_data = kwargs.pop("calibration_data", None)
        _raise_unexpected_export_options("INT8 ONNX", kwargs)
        return export_onnx_int8(
            model_api,
            path,
            calibration_data=calibration_data,
            imgsz=imgsz,
            batch=batch,
            dynamic=dynamic,
            simplify=simplify,
            opset=opset,
            device=device,
            verify=verify,
        )
    if fmt == "onnx":
        _raise_unexpected_export_options("ONNX", kwargs)
        return export_onnx(model_api, path, imgsz, batch, dynamic, half, simplify, opset, device, verify=verify)
    if fmt == "torchscript":
        _raise_unexpected_export_options("TorchScript", kwargs)
        return export_torchscript(model_api, path, imgsz, batch, half, device, dynamic=dynamic, verify=verify)
    if fmt == "tensorrt":
        return export_tensorrt(
            model_api, path, imgsz, batch, dynamic, half, simplify, opset, device=device,
            verify=verify, **kwargs,
        )
    if fmt == "coreml":
        coreml_format = kwargs.pop("coreml_format", "auto")
        _raise_unexpected_export_options("CoreML", kwargs)
        # CoreML has no portable Linux/Windows execution API for numerical
        # parity. Its platform-specific validation remains explicitly manual.
        del verify
        return export_coreml(model_api, path, imgsz, batch, half, device, coreml_format=coreml_format)
