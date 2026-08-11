"""Active-checkpoint identity adapter for the strict training launcher."""

from dataclasses import asdict, dataclass
import hashlib
import json

from fotonet.models.v1.spec import get_architecture_spec


@dataclass(frozen=True)
class FoundationIdentity:
    profile: str
    backbone_channels: tuple[int, int, int, int, int]
    backbone_depths: tuple[int, int, int, int]
    downsample_modes: tuple[str, str, str, str, str]
    neck_channels: tuple[int, int, int, int]
    foundation_version: int = 2

    @property
    def fingerprint(self):
        payload = json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
        return hashlib.sha256(payload).hexdigest()

    @property
    def backbone_out_channels(self):
        return self.backbone_channels[1:]


def get_foundation_spec(profile):
    spec = get_architecture_spec(profile)
    return FoundationIdentity(
        profile=spec.profile,
        backbone_channels=spec.backbone_channels,
        backbone_depths=spec.backbone_depths,
        downsample_modes=spec.downsample_modes,
        neck_channels=spec.neck_channels,
    )
