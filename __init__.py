import sys
from pathlib import Path

import bpy
from bpy.props import StringProperty, CollectionProperty, BoolProperty
from bpy_extras.io_utils import ExportHelper, ImportHelper

if "voxel_core_model" not in sys.modules:
    sys.modules['voxel_core_model'] = sys.modules[Path(__file__).parent.stem]

from voxel_core_model.exporter import export_vec3
from voxel_core_model.importer import import_vec3
from voxel_core_model.file_utils import FileBuffer
from voxel_core_model.mesh_utils import is_blender_4_1
from voxel_core_model.model.body import write_model_to_buffer

bl_info = {
    "name": "VoxelEngine model tools",
    "author": "RED_EYE",
    "version": (0, 0, 1),
    "blender": (3, 6, 0),
    "location": "File > Import > VC vec3",
    "description": "VoxelEngine import/export tools",
    "category": "Import-Export"
}


class OperatorHelper(bpy.types.Operator):
    if is_blender_4_1():
        directory: StringProperty(subtype='FILE_PATH', options={'SKIP_SAVE', 'HIDDEN'})
    filepath: StringProperty(subtype='FILE_PATH', default="model.vec3")
    files: CollectionProperty(name='File paths', type=bpy.types.OperatorFileListElement)

    def get_directory(self):
        if is_blender_4_1():
            return Path(self.directory)
        else:
            filepath = Path(self.filepath)
            print(filepath)
            if filepath.is_file():
                return filepath.parent.absolute()
            else:
                return filepath.absolute()


class ImportOperatorHelper(OperatorHelper):
    need_popup = True

    def invoke_popup(self, context, confirm_text=""):
        if self.properties.is_property_set("filepath"):
            title = self.filepath
            if len(self.files) > 1:
                title = f"Import {len(self.files)} files"

            if not confirm_text:
                confirm_text = self.bl_label
            return context.window_manager.invoke_props_dialog(self, confirm_text=confirm_text, title=title,
                                                              translate=False)

        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        if is_blender_4_1() and self.directory and self.files:
            if self.need_popup:
                return self.invoke_popup(context)
            else:
                return self.execute(context)
        wm = context.window_manager
        wm.fileselect_add(self)
        return {'RUNNING_MODAL'}


class ExportOperatorHelper(OperatorHelper):
    def invoke(self, context, event):
        wm = context.window_manager
        wm.fileselect_add(self)
        return {'RUNNING_MODAL'}


class VOXELCORE_OT_VEC3Import(ImportOperatorHelper, ImportHelper):
    """Load VC vec3 models"""
    bl_idname = "voxelcore.import_vec3"
    bl_label = "Import Voxel Core vec3 model"
    bl_options = {'UNDO'}

    filter_glob: StringProperty(default="*.vec3", options={'HIDDEN'})

    def execute(self, context):
        directory = self.get_directory()

        for file in self.files:
            filepath = directory / file.name
            with FileBuffer(filepath) as f:
                import_vec3(f)
        return {'FINISHED'}


class VOXELCORE_OT_VEC3Export(ExportOperatorHelper, ExportHelper):
    """Save VOXELCORE vec3 models"""
    bl_idname = "voxelcore.export_vec3"
    bl_label = "Export Voxel Core vec3 model"
    bl_options = {'UNDO', 'PRESET'}

    # ExportHelper mixin class uses this
    filename_ext = ".vec3"

    filter_glob: StringProperty(default="*.vec3", options={'HIDDEN'})

    compress: BoolProperty(default=False, name="Compress", description="Compress mesh data with GZIP")

    def invoke(self, context, event):
        # Set a default filepath
        self.filepath = bpy.path.ensure_ext(bpy.data.filepath or "model", ".vec3")
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        if not self.filepath:
            raise Exception("No filename provided")
        with FileBuffer(self.filepath, 'wb') as f:
            body = export_vec3(context, self.compress)
            write_model_to_buffer(f, body)
        return {'FINISHED'}


class MATERIAL_PT_VoxelEngineProperties(bpy.types.Panel):
    bl_label = "Voxel Engine Material Properties"
    bl_idname = "voxelcore.material_properties"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = 'material'

    @classmethod
    def poll(cls, context):
        return context.material is not None

    def draw(self, context):
        layout = self.layout
        material = context.material
        layout.prop(material, "shadeless")


classes = [VOXELCORE_OT_VEC3Import, VOXELCORE_OT_VEC3Export, MATERIAL_PT_VoxelEngineProperties]

register_, unregister_ = bpy.utils.register_classes_factory(classes)


def menu_import(self, context):
    self.layout.operator(VOXELCORE_OT_VEC3Import.bl_idname)


def menu_export(self, context):
    self.layout.operator(VOXELCORE_OT_VEC3Export.bl_idname)


def register():
    register_()
    bpy.types.Material.shadeless = bpy.props.BoolProperty(
        name="Shadeless",
        default=False
    )
    bpy.types.TOPBAR_MT_file_import.append(menu_import)
    bpy.types.TOPBAR_MT_file_export.append(menu_export)


def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_import)
    bpy.types.TOPBAR_MT_file_export.remove(menu_export)
    del bpy.types.Material.shadeless
    unregister_()
