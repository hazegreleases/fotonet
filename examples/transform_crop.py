import numpy as np

from fotonet import AnchorPoint, BoxTransform


def main():
    image = np.zeros((400, 800, 3), dtype=np.uint8)
    box = BoxTransform((0.5, 0.5, 0.25, 0.5), image_size=(800, 400))

    print("original xyxy:", tuple(round(v, 4) for v in box.xyxy))
    print("original pixel size:", tuple(round(v, 1) for v in box.pixelSize))

    box.setAnchor(AnchorPoint.CENTER).pixelExpand(40).pixelMove((10, -20)).clamp()

    crop = box.crop(image)
    print("expanded/moved xyxy:", tuple(round(v, 4) for v in box.xyxy))
    print("expanded/moved pixel position:", tuple(round(v, 1) for v in box.pixelPosition))
    print("expanded/moved pixel size:", tuple(round(v, 1) for v in box.pixelSize))
    print("crop shape:", crop.shape)
    print("contains moved center:", box.pixelContains((410, 180)))
    print("contains far outside point:", box.pixelContains((50, 50)))

    assert tuple(round(v, 1) for v in box.pixelSize) == (280.0, 280.0)
    assert crop.shape == (280, 280, 3)
    assert bool(box.pixelContains((410, 180))) is True
    assert bool(box.pixelContains((50, 50))) is False


if __name__ == "__main__":
    main()
