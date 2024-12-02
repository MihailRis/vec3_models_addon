import gzip
from dataclasses import dataclass
from enum import IntFlag

import numpy as np

from voxel_core_model.file_utils import Buffer
from voxel_core_model.model.vertex_attribute import VertexAttributeType, VertexAttribute


class MeshFlags(IntFlag):
    NONE = 0
    GZIP = 1
    USHORT_INDICES = 2


@dataclass(slots=True, frozen=True)
class Mesh:
    material_id: int
    flags: MeshFlags
    attributes: list[VertexAttribute]
    indices: np.ndarray

    @classmethod
    def from_buffer(cls, buffer: Buffer) -> 'Mesh':
        triangle_count, material_id, flags, attribute_count = buffer.read_fmt("I3H")
        attributes = [VertexAttribute.from_buffer(buffer) for _ in range(attribute_count)]
        expected_buffer_size = triangle_count * 3 * attribute_count
        if flags & MeshFlags.USHORT_INDICES:
            expected_buffer_size *= 2
        if flags & MeshFlags.GZIP:
            compressed_size = buffer.read_uint32()
            data = gzip.decompress(buffer.read(compressed_size))
            if len(data) != expected_buffer_size:
                raise ValueError(
                    "Decompressed data size does not match: {}!={}".format(len(data), expected_buffer_size))
        else:
            data = buffer.read(expected_buffer_size)
        if flags & MeshFlags.USHORT_INDICES:
            indices = np.frombuffer(data, dtype=np.uint16).reshape(-1, 3, attribute_count)
        else:
            indices = np.frombuffer(data, dtype=np.uint8).reshape(-1, 3, attribute_count)
        return cls(material_id, MeshFlags(flags), attributes, indices)

    def to_buffer(self, buffer: Buffer) -> Buffer:
        buffer.write_fmt("I3H", self.indices.shape[0], self.material_id,
                         self.flags, len(self.attributes))
        for attribute in self.attributes:
            attribute.to_buffer(buffer)

        if self.flags & MeshFlags.GZIP:
            data = gzip.compress(self.indices.tobytes())
            buffer.write_uint32(len(data))
            buffer.write(data)
        else:
            buffer.write(self.indices.tobytes())

        return buffer

    def find_attribute(self, attribute_type: VertexAttributeType) -> tuple[np.ndarray | None, int | None]:
        for i, attribute in enumerate(self.attributes):
            if attribute.type == attribute_type:
                return attribute.data, i
        return None, None

    def has_attribute(self, attribute_type: VertexAttributeType) -> bool:
        for i, attribute in enumerate(self.attributes):
            if attribute.type == attribute_type:
                return True
        return False
