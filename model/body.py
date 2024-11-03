from dataclasses import dataclass
from pathlib import Path

from voxel_core_model.file_utils import Buffer, FileBuffer
from voxel_core_model.model.material import Material
from voxel_core_model.model.model import Model


@dataclass(slots=True, frozen=True)
class Body:
    models: list[Model]
    materials: list[Material]

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        material_count, model_count = buffer.read_fmt("2H")
        materials = [Material.from_buffer(buffer) for _ in range(material_count)]
        models = [Model.from_buffer(buffer) for _ in range(model_count)]
        return cls(models, materials)

    def to_buffer(self, buffer: Buffer):
        buffer.write_fmt("2H", len(self.materials), len(self.models))
        for material in self.materials:
            material.to_buffer(buffer)
        for model in self.models:
            model.to_buffer(buffer)


def load_model_from_buffer(buffer: Buffer) -> Body:
    ident = buffer.read(8)
    if ident != b"\x00\x00VEC3\x00\x00":
        raise ValueError(f"Invalid header. Invalid identifier, expected b\"\x00\x00VEC3\x00\x00\", but got {ident}.")
    version, _ = buffer.read_fmt("2H")
    if version != 1:
        raise ValueError(f"Invalid header. Unsupported version, expected 1, but got {version}.")
    return Body.from_buffer(buffer)


def load_model_from_path(path: Path) -> Body:
    with FileBuffer(path, "rb") as f:
        return load_model_from_buffer(f)


def write_model_to_buffer(buffer: Buffer, model: Body) -> Buffer:
    buffer.write(b"\x00\x00VEC3\x00\x00")
    buffer.write_uint32(1)
    model.to_buffer(buffer)
    return buffer
