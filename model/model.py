from dataclasses import dataclass

from voxel_core_model.file_utils import Buffer
from .mesh import Mesh


@dataclass(slots=True, frozen=True)
class Model:
    name: str
    origin: tuple[float, float, float]
    meshes: list[Mesh]

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        name_size = buffer.read_uint16()
        origin = buffer.read_fmt("3f")
        mesh_count = buffer.read_uint32()
        meshes = [Mesh.from_buffer(buffer) for _ in range(mesh_count)]
        name = buffer.read_ascii_string(name_size)
        return cls(name, origin, meshes)

    def to_buffer(self, buffer: Buffer):
        buffer.write_uint16(len(self.name))
        buffer.write_fmt("3f", *self.origin)
        buffer.write_uint32(len(self.meshes))
        for mesh in self.meshes:
            mesh.to_buffer(buffer)
        buffer.write_ascii_string(self.name)
        return buffer
