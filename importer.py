import bpy
import numpy as np

from voxel_core_model.file_utils import Buffer
from voxel_core_model.mesh_utils import add_uv_layer, add_custom_normals, add_vertex_color_layer, \
    get_or_create_material, add_material
from voxel_core_model.model.body import load_model_from_buffer
from voxel_core_model.model.vertex_attribute import VertexAttributeType

DIRECTION_SWAP = np.asarray([1, -1, 1], np.float32)
AXIS_SWAP = [0, 2, 1]


def import_vec3(buffer: Buffer):
    model = load_model_from_buffer(buffer)

    model_materials = model.materials
    for sub_model in model.models:
        mesh0 = sub_model.meshes[0]
        attrs_comp0 = [atr.type for atr in mesh0.attributes]
        for mesh in sub_model.meshes:
            attrs_comp = [atr.type for atr in mesh.attributes]
            if attrs_comp != attrs_comp0:
                raise NotImplementedError("All meshes in submodel must have same number and order of attributes")

        attributes: dict[int, np.ndarray] = {}
        for attribute in mesh0.attributes:
            attributes[attribute.type] = np.zeros((0, *attribute.data.shape[1:]), np.float32)
        total_indices = np.empty((0, 3, len(mesh0.attributes)), np.uint16)

        for mesh in sub_model.meshes:
            mesh_indices = mesh.indices.copy().astype(np.uint16)
            if total_indices.size != 0:
                mesh_indices += (total_indices.reshape(-1, len(attributes)).max(axis=0) + 1)
            total_indices = np.vstack((total_indices, mesh_indices))
            for i, attribute in enumerate(mesh.attributes):
                attributes[i] = np.vstack((attributes[i], attribute.data))

        mesh_data = bpy.data.meshes.new(f"{sub_model.name}_MESH")
        mesh_obj = bpy.data.objects.new(f"{sub_model.name}", mesh_data)
        _, position_index = mesh0.find_attribute(VertexAttributeType.POSITION)

        if not mesh0.has_attribute(VertexAttributeType.POSITION):
            raise ValueError("Position attribute not found!")

        vertex_indices = total_indices[:, :, position_index]
        mesh_data.from_pydata(attributes[position_index][:, AXIS_SWAP] * DIRECTION_SWAP, [],
                              vertex_indices.reshape(-1, 3))
        mesh_data.update(calc_edges=True, calc_edges_loose=True)

        material_indices = np.zeros(len(mesh_data.polygons), np.uint32)
        poly_offset = 0
        for mesh in sub_model.meshes:
            material = model_materials[mesh.material_id]
            mat = get_or_create_material(material.name)
            mat_index = add_material(mat, mesh_obj)
            material_indices[poly_offset:poly_offset + mesh.indices.shape[0]] = mat_index
            poly_offset += mesh.indices.shape[0]

        mesh_data.polygons.foreach_set('material_index', material_indices)

        if mesh0.has_attribute(VertexAttributeType.UV):
            _, uv_index = mesh0.find_attribute(VertexAttributeType.UV)
            uv_indices = total_indices[:, :, uv_index].ravel()
            add_uv_layer("UV", attributes[uv_index], mesh_data, uv_indices, flip_uv=False)
        if mesh0.has_attribute(VertexAttributeType.NORMAL):
            _, normal_index = mesh0.find_attribute(VertexAttributeType.NORMAL)
            normals_indices = total_indices[:, :, normal_index].ravel()
            add_custom_normals(attributes[normal_index][normals_indices][:, AXIS_SWAP] * DIRECTION_SWAP, mesh_data)
        if mesh0.has_attribute(VertexAttributeType.COLOR):
            _, color_index = mesh0.find_attribute(VertexAttributeType.COLOR)
            colors_indices = total_indices[:, :, color_index].ravel()
            add_vertex_color_layer("COLOR", attributes[color_index], mesh_data, colors_indices)

        mesh_obj.location = (sub_model.origin[0], sub_model.origin[2], -sub_model.origin[1])
        bpy.context.scene.collection.objects.link(mesh_obj)
