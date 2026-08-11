__all__ = [
    "AnnotationAudit",
    "AnnotationError",
    "CocoDetectionDataset",
    "DetectionDataset",
    "build_detection_dataset",
    "expand_image_sources",
]


def __getattr__(name):
    # Keep package initialization acyclic: dataset imports augmentation modules,
    # while the optional COCO adapter subclasses the dataset.
    if name in {"AnnotationAudit", "AnnotationError"}:
        from fotonet.data.audit import AnnotationAudit, AnnotationError

        return {"AnnotationAudit": AnnotationAudit, "AnnotationError": AnnotationError}[name]
    if name == "CocoDetectionDataset":
        from fotonet.data.coco import CocoDetectionDataset

        return CocoDetectionDataset
    if name in {"DetectionDataset", "build_detection_dataset", "expand_image_sources"}:
        from fotonet.data.dataset import DetectionDataset, build_detection_dataset, expand_image_sources

        return {
            "DetectionDataset": DetectionDataset,
            "build_detection_dataset": build_detection_dataset,
            "expand_image_sources": expand_image_sources,
        }[name]
    raise AttributeError(name)
