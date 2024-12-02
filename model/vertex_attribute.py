import gzip
from dataclasses import dataclass
from enum import IntFlag, IntEnum

import numpy as np

from voxel_core_model.file_utils import Buffer


class VertexAttributeFlags(IntFlag):
    NONE = 0
    GZIP = 1


class VertexAttributeType(IntEnum):
    POSITION = 0
    UV = 1
    NORMAL = 2
    COLOR = 4

    def data_type(self) -> tuple[np.number, int]:
        """Returns a tuple of component numpy type and component count"""
        if self == VertexAttributeType.POSITION:
            return np.float32, 3
        elif self == VertexAttributeType.NORMAL:
            return np.float32, 3
        elif self == VertexAttributeType.UV:
            return np.float32, 2
        elif self == VertexAttributeType.COLOR:
            return np.float32, 4
        raise ValueError(f"Unknown attribute type: {self}")


@dataclass(slots=True, frozen=True)
class VertexAttribute:
    type: VertexAttributeType
    flags: VertexAttributeFlags
    data: np.ndarray

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        v_type = VertexAttributeType(buffer.read_uint8())
        flags = VertexAttributeFlags(buffer.read_uint8())
        size = buffer.read_uint32()

        if flags & VertexAttributeFlags.GZIP:
            decompressed_size = buffer.read_uint32()
            data = gzip.decompress(buffer.read(size - 4))
            if len(data) != decompressed_size:
                raise ValueError("Decompressed data size does not match: {}!={}".format(len(data), decompressed_size))
        else:
            data = buffer.read(size)
        return cls(v_type, flags, np.frombuffer(data, np.float32).reshape(-1, v_type.data_type()[1]))

    def to_buffer(self, buffer: Buffer):
        buffer.write_fmt("2B", self.type, self.flags)

        if self.flags & VertexAttributeFlags.GZIP:
            data = gzip.compress(self.data.tobytes())
            buffer.write_uint32(len(data) + 4)
            buffer.write_uint32(self.data.nbytes)
            buffer.write(data)
        else:
            buffer.write_uint32(self.data.nbytes)
            buffer.write(self.data.data)

        return buffer
