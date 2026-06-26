"""Export helpers for FOTO-NET inference graphs."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import torch


def _metadata(model, fmt, imgsz, batch, dynamic, half, extra=None):
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        names = {str(k): v for k, v in names.items()}
    data = {
        "format": fmt,
        "imgsz": int(imgsz),
        "batch": int(batch),
        "dynamic": bool(dynamic),
        "half": bool(half),
        "nc": int(getattr(model, "nc", getattr(model.model.head, "nc", 80))),
        "names": names,
        "use_p2": bool(getattr(model, "use_p2", getattr(model.model, "use_p2", False))),
        "reg_max": int(getattr(model, "reg_max", getattr(model.model.head, "reg_max", 16))),
        "p3_extra_blocks": int(getattr(model, "p3_extra_blocks", getattr(model.model, "p3_extra_blocks", 0))),
        "p4_extra_blocks": int(getattr(model, "p4_extra_blocks", getattr(model.model, "p4_extra_blocks", 0))),
        "p5_extra_blocks": int(getattr(model, "p5_extra_blocks", getattr(model.model, "p5_extra_blocks", 0))),
        "p5_gate_blocks": int(getattr(model, "p5_gate_blocks", getattr(model.model, "p5_gate_blocks", 0))),
        "strides": list(getattr(model.model.head, "strides", [8, 16, 32])),
        "output": "raw_o2o_tensor[B,N,nc+4]",
        "nms_free": True,
        "postprocess": "sigmoid logits, max class, threshold, top-k cap; NMS optional only for legacy comparison",
    }
    if extra:
        data.update(extra)
    return data


def _write_metadata(path, model, fmt, imgsz, batch, dynamic, half, extra=None):
    meta_path = Path(path).with_suffix(Path(path).suffix + ".metadata.json")
    with open(meta_path, "w") as f:
        json.dump(_metadata(model, fmt, imgsz, batch, dynamic, half, extra), f, indent=2)
    return str(meta_path)


def export_onnx(model_api, path, imgsz=640, batch=1, dynamic=False, half=False,
                simplify=True, opset=17, device=None):
    device = device or model_api.device
    model = model_api.model.to(device).eval()
    old_dtype = next(model.parameters()).dtype
    use_half = bool(half and device.type == "cuda")
    if use_half:
        model.half()

    dummy = torch.zeros(batch, 3, imgsz, imgsz, device=device, dtype=torch.float16 if use_half else torch.float32)
    axes = None
    if dynamic:
        axes = {
            "images": {0: "batch"},
            "output0": {0: "batch"},
        }

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    export_kwargs = {
        "input_names": ["images"],
        "output_names": ["output0"],
        "dynamic_axes": axes,
        "opset_version": int(opset),
        "do_constant_folding": True,
    }
    with torch.no_grad():
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

    if old_dtype == torch.float32:
        model.float()

    meta = _write_metadata(path, model_api, "onnx", imgsz, batch, dynamic, use_half, {"opset": int(opset)})
    return {"artifact": str(path), "metadata": meta}


def export_torchscript(model_api, path, imgsz=640, batch=1, half=False, device=None):
    device = device or model_api.device
    model = model_api.model.to(device).eval()
    old_dtype = next(model.parameters()).dtype
    use_half = bool(half and device.type == "cuda")
    if use_half:
        model.half()
    dummy = torch.zeros(batch, 3, imgsz, imgsz, device=device, dtype=torch.float16 if use_half else torch.float32)
    with torch.no_grad(), _export_reference_graph(model_api):
        traced = torch.jit.trace(model, dummy)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    traced.save(path)
    if old_dtype == torch.float32:
        model.float()
    meta = _write_metadata(path, model_api, "torchscript", imgsz, batch, False, use_half)
    return {"artifact": str(path), "metadata": meta}


def export_tensorrt(model_api, path, imgsz=640, batch=1, dynamic=False, half=True,
                    simplify=True, opset=17, workspace=4, device=None):
    trtexec = shutil.which("trtexec")
    if trtexec is None:
        raise RuntimeError("TensorRT export requires trtexec in PATH. Install TensorRT or export ONNX first.")

    onnx_path = str(Path(path).with_suffix(".onnx"))
    export_onnx(model_api, onnx_path, imgsz=imgsz, batch=batch, dynamic=dynamic,
                half=half, simplify=simplify, opset=opset, device=device)
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
    if dynamic:
        shape = f"images:{batch}x3x{imgsz}x{imgsz}"
        cmd.extend([f"--minShapes={shape}", f"--optShapes={shape}", f"--maxShapes={shape}"])
    subprocess.run(cmd, check=True)
    meta = _write_metadata(path, model_api, "tensorrt", imgsz, batch, dynamic, half, {"source_onnx": onnx_path})
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

    device = device or torch.device("cpu")
    model = model_api.model.to(device).eval()
    dummy = torch.zeros(batch, 3, imgsz, imgsz, device=device)
    with torch.no_grad(), _export_reference_graph(model_api):
        traced = torch.jit.trace(model, dummy)

    requested = str(coreml_format or "auto").lower()
    if requested in ("mlpackage", "mlprogram"):
        formats = ["mlprogram"]
    elif requested in ("mlmodel", "neuralnetwork", "neural_network"):
        formats = ["neuralnetwork"]
    elif requested == "auto":
        formats = ["mlprogram", "neuralnetwork"]
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


def export_model(model_api, path=None, format="onnx", imgsz=640, batch=1,
                 dynamic=False, half=False, simplify=True, int8=False,
                 opset=17, device=None, **kwargs):
    fmt = format.lower()
    if fmt == "engine":
        fmt = "tensorrt"
    if path is None:
        suffix = {"onnx": ".onnx", "tensorrt": ".engine", "coreml": ".mlpackage", "torchscript": ".torchscript"}.get(fmt, f".{fmt}")
        path = f"fotonet_export{suffix}"

    if int8:
        raise NotImplementedError("INT8 export needs calibration data; planned but not enabled yet.")
    if fmt == "onnx":
        return export_onnx(model_api, path, imgsz, batch, dynamic, half, simplify, opset, device)
    if fmt in ("torchscript", "jit"):
        return export_torchscript(model_api, path, imgsz, batch, half, device)
    if fmt == "tensorrt":
        return export_tensorrt(model_api, path, imgsz, batch, dynamic, half, simplify, opset, device=device, **kwargs)
    if fmt == "coreml":
        return export_coreml(model_api, path, imgsz, batch, half, device, coreml_format=kwargs.get("coreml_format", "auto"))
    raise ValueError(f"Unsupported export format: {format}")
