from dataclasses import dataclass
from enum import IntFlag

from voxel_core_model.file_utils import Buffer


class MaterialFlags(IntFlag):
    NONE = 0
    SHADELESS = 1


@dataclass(slots=True, frozen=True)
class Material:
    name: str
    flags: int

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        flags = MaterialFlags(buffer.read_uint16())
        name = buffer.read_ascii_string(buffer.read_uint16())
        return cls(name, flags)

    def to_buffer(self, buffer: Buffer):
        buffer.write_uint16(self.flags)
        buffer.write_uint16(len(self.name))
        buffer.write_ascii_string(self.name)
        return buffer
