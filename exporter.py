from dataclasses import dataclass, field

import bpy
import numpy as np
from bpy.types import Depsgraph

from voxel_core_model.model.body import Body
from voxel_core_model.model.material import Material, MaterialFlags
from voxel_core_model.model.mesh import Mesh, MeshFlags
from voxel_core_model.model.model import Model
from voxel_core_model.model.vertex_attribute import VertexAttribute, VertexAttributeType, VertexAttributeFlags

DIRECTION_SWAP = np.asarray([1, -1, 1], np.float32)
AXIS_SWAP = [0, 2, 1]


@dataclass(slots=True)
class IntermediateMesh:
    positions: np.ndarray = field(default_factory=lambda: np.empty((0, 3), np.float32))
    normals: np.ndarray = field(default_factory=lambda: np.empty((0, 3), np.float32))
    uvs: np.ndarray = field(default_factory=lambda: np.empty((0, 2), np.float32))
    polygons: np.ndarray = field(default_factory=lambda: np.empty((0, 3), np.uint32))
    material_ids: np.ndarray = field(default_factory=lambda: np.empty((0,), np.uint32))
    materials: list[str] = field(default_factory=list)


def convert_to_vec3_meshes(mesh_data: IntermediateMesh, materials: list[Material], compress=False) -> list[Mesh]:
    meshes: dict[int, Mesh] = {}

    positions = (np.asarray(mesh_data.positions, np.float32) * DIRECTION_SWAP)[:, AXIS_SWAP]
    normals = (np.asarray(mesh_data.normals, np.float32) * DIRECTION_SWAP)[:, AXIS_SWAP]
    uvs = np.asarray(mesh_data.uvs, np.float32)
    polygons = np.asarray(mesh_data.polygons, np.uint32)
    material_ids = np.asarray(mesh_data.material_ids, np.uint32)

    unique_material_ids = np.unique(material_ids)

    for material_id in unique_material_ids:
        poly_mask = (material_ids == material_id)
        material_polygons = polygons[poly_mask]

        unique_vertex_indices, inverse_indices = np.unique(material_polygons, return_inverse=True)

        material_positions = positions[unique_vertex_indices]
        material_normals = normals[unique_vertex_indices]
        material_uvs = uvs[unique_vertex_indices]

        unique_positions, positions_inverse = np.unique(material_positions, axis=0, return_inverse=True)
        unique_normals, normals_inverse = np.unique(material_normals, axis=0, return_inverse=True)
        unique_uvs, uvs_inverse = np.unique(material_uvs, axis=0, return_inverse=True)

        remapped_polygons = inverse_indices.reshape(material_polygons.shape)
        attribute_flags = VertexAttributeFlags.NONE
        if compress:
            attribute_flags |= VertexAttributeFlags.GZIP

        attributes = [
            VertexAttribute(VertexAttributeType.POSITION, attribute_flags, unique_positions),
            VertexAttribute(VertexAttributeType.UV, attribute_flags, unique_uvs),
            VertexAttribute(VertexAttributeType.NORMAL, attribute_flags, unique_normals),
        ]

        use_short_indices = polygons.max() >= 255

        indices = np.zeros((material_polygons.shape[0], len(attributes), 3),
                           np.uint16 if use_short_indices else np.uint8)

        indices[:, :, VertexAttributeType.POSITION] = positions_inverse.ravel()[remapped_polygons]
        indices[:, :, VertexAttributeType.UV] = uvs_inverse.ravel()[remapped_polygons]
        indices[:, :, VertexAttributeType.NORMAL] = normals_inverse.ravel()[remapped_polygons]

        for g_material_id, material in enumerate(materials):
            if material.name == mesh_data.materials[material_id]:
                break
        else:
            g_material_id = 0

        mesh_flags = MeshFlags.NONE
        if compress:
            mesh_flags |= MeshFlags.GZIP
        if use_short_indices:
            mesh_flags |= MeshFlags.USHORT_INDICES

        meshes[material_id] = Mesh(g_material_id, mesh_flags, attributes,
                                   indices)
    return list(meshes.values())


def collect_meshes_data(obj: bpy.types.Object, depsgraph: Depsgraph, materials: list[Material]):
    obj_eval = obj.evaluated_get(depsgraph)
    mesh: bpy.types.Mesh = obj_eval.to_mesh()
    uv_layer = mesh.uv_layers.active
    if uv_layer is None:
        print(f"No UV layer found on mesh: {obj.name}")
        obj_eval.to_mesh_clear()
        return mesh

    mesh.calc_tangents(uvmap=uv_layer.name)
    mesh.calc_loop_triangles()
    material_remap = np.zeros(len(obj.material_slots), np.uint32)
    material_names = []

    for mat_id, mat_slot in enumerate(obj.material_slots):
        slot_material = mat_slot.material
        mat_name = slot_material.name if slot_material else "NoMaterial"
        material_names.append(mat_name)

        emat_id = next((i for i, material in enumerate(materials) if material.name == mat_name), None)
        if emat_id is not None:
            material_remap[mat_id] = emat_id
        else:
            material_remap[mat_id] = len(materials)
            if slot_material.shadeless:
                material_flags = MaterialFlags.SHADELESS
            else:
                material_flags = MaterialFlags.NONE
            materials.append(Material(mat_name, material_flags))

    if not obj.material_slots:
        mat_name = "NoMaterial"
        emat_id = next((i for i, material in enumerate(materials) if material.name == mat_name), None)
        if emat_id is not None:
            material_remap[0] = emat_id
        else:
            material_remap[0] = len(materials)
            materials.append(Material(mat_name, MaterialFlags.NONE))

    uv_data = mesh.uv_layers.active.data

    vertices = np.zeros((len(mesh.vertices), 3), np.float32)
    normals = np.zeros((len(mesh.loops), 3), np.float32)
    uvs = np.zeros((len(mesh.loop_triangles) * 3, 2), np.float32)
    vertex_indices = np.zeros(len(mesh.loops), np.uint32)

    mesh.vertices.foreach_get("co", vertices.ravel())
    mesh.loops.foreach_get("normal", normals.ravel())
    uv_data.foreach_get("uv", uvs.ravel())
    mesh.loops.foreach_get("vertex_index", vertex_indices.ravel())

    n_tris = len(mesh.loop_triangles)
    n_loops = n_tris * 3

    loops = np.empty((n_loops,), dtype=np.int32)
    for t, tri in enumerate(mesh.loop_triangles):
        loops[3 * t: 3 * t + 3] = tri.loops

    pos = vertices[vertex_indices[loops]]
    norm = normals[loops]
    uv = uvs[loops]

    data = np.hstack((pos, norm, uv))

    dt = np.dtype([('row', data.dtype, data.shape[1])])
    recs = data.view(dt).ravel()

    uniq_recs, inv = np.unique(recs, return_inverse=True)
    uniq = uniq_recs['row']

    output_positions = uniq[:, :3]
    output_normals = uniq[:, 3:6]
    output_uvs = uniq[:, 6:8]

    polygons = inv.reshape(n_tris, 3).astype(np.uint32)
    material_ids = np.array(
        [tri.material_index for tri in mesh.loop_triangles],
        dtype=np.uint32
    )

    obj_eval.to_mesh_clear()
    return IntermediateMesh(
        output_positions,
        output_normals,
        output_uvs,
        polygons,
        material_ids,
        material_names
    )

def export_vec3(context: bpy.context, compress=False):
    depsgraph: Depsgraph = context.evaluated_depsgraph_get()
    submodels: list[Model] = []
    materials: list[Material] = []
    for obj in context.selected_objects:
        if obj.type == 'MESH':
            print(f"Processing {obj.name}")
            mesh_data = collect_meshes_data(obj, depsgraph, materials)
            meshes = convert_to_vec3_meshes(mesh_data, materials, compress)
            loc = obj.location
            submodels.append(Model(obj.name, (loc.x, -loc.z, loc.y), meshes))
    return Body(submodels, materials)
