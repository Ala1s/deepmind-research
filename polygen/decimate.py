import bpy
from glob import glob
import os
import random

##Cleans all decimate modifiers
def cleanAllDecimateModifiers(obj):
	for m in obj.modifiers:
	    if(m.type=="DECIMATE"):
	        obj.modifiers.remove(modifier=m)


objs = bpy.data.objects
objs.remove(objs["Cube"], do_unlink=True)

chair_paths = glob("/home/deria/.objaverse/hf-objaverse-v1/glbs/*/*.glb")

for path in chair_paths:
	print(path)
	imported_object = bpy.ops.import_scene.gltf(filepath=path)
    print('imported name: ', imported_object)
    print('selected objects: ', bpy.context.selected_objects)
	obj_object = bpy.context.selected_objects[0] ####<--Fix
	bpy.context.view_layer.objects.active = obj_object

	# for edge in obj_object.data.edges:
	#     edge.use_edge_sharp = False

	# # print(bpy.context.active_object)
	# bpy.ops.object.mode_set(mode='EDIT')
	# bpy.ops.mesh.dissolve_degenerate(threshold=0.0001)
	# bpy.ops.mesh.select_all(action='SELECT')
	# bpy.ops.mesh.normals_make_consistent(inside=False)
	# bpy.ops.object.mode_set(mode='OBJECT')


	
	modifierName='DecimateMod'

	cleanAllDecimateModifiers(obj_object)
	modifier=obj_object.modifiers.new(modifierName,'DECIMATE')
	modifier.decimate_type="DISSOLVE"
	modifier.angle_limit = random.uniform(0.0174533, 0.349066)
	bpy.ops.wm.obj_export(filepath=path.replace("glbs","objs"))
	bpy.ops.object.delete() 
