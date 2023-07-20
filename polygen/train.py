import objaverse
import multiprocessing
import random
import trimesh
import numpy as np


import os
import numpy as np
import tensorflow.compat.v1 as tf
tf.logging.set_verbosity(tf.logging.ERROR)  # Hide TF deprecation messages
import matplotlib.pyplot as plt

import modules
import data_utils

processes = multiprocessing.cpu_count()
uids = objaverse.load_uids()
annotations = objaverse.load_annotations()

#OBJECT LOAD PARAMETERS

max_classes = 5
max_objects_per_class = 5
max_faceCount = 3000
min_faceCount = 500
min_vertexCount = 200


class_dict = {}
for uid, annotation in annotations.items():
    if (annotation["faceCount"] <= max_faceCount and annotation["faceCount"] >= min_faceCount and annotation["vertexCount"] >= min_vertexCount):
        for tag in annotation["tags"]:
            newList = class_dict[tag['name']] if tag['name'] in class_dict else []
            newList.append(uid)
            class_dict[tag['name']] = newList


tags_to_remove = ['lowpoly', 'substancepainter', 'substance', 'low-poly', 'blender', '3d', '3dmodel', '3dmodeling', 'poly', 'video-game', 'game', 'low', 'maya', 'gameasset', '3dsmax', 'google', 'asset', 'medievalfantasyassets', 'voxel', 'unity', 'blender3d', 'medieval', 'gameready', 'stylized', 'pbr', 'design', 'low-poly-model', 'model', 'sketchup', 'handpainted', 'downloadable', 'wooden', 'unity3d', 'metal', 'props', 'game-ready', 'lowpolymodel', 'art', 'texture', '3dsmaxpublisher','gameart','simple','pointcloud','game-asset','freemodel', 'low-poly-blender', 'animation', 'obj', 'zbrush', 'blender-3d', 'fbx','scifi','substance-painter','sample','photogrammetry','unreal','free3dmodel','lidar','games', 'video-games', 'free', 'old', 'download', 'home','staircon', 'assets', 'videogame', 'gameassets', 'gameasset','cute', 'textured','everypoint', 'animated','sweethome3d','cinema4d','pixelart','photoshop','retro','red', 'vr', 'realistic', '3dscan', 'isometric', 'rigged', 'gamedev', 'pixel', 'gaming', 'pixel-art', 'b3d', 'blue','stairs','stair','staircases','staircase', 'pamir']

for tag in tags_to_remove:
    if tag in class_dict:
        del class_dict[tag]

random.seed(42)
classes = []
object_uids = []
for i in range(max_classes):
    max_class = max(class_dict, key=lambda k: len(class_dict[k]))
    print(max_class, len(class_dict[max_class]))
    classes.append(max_class)
    object_uids.append(class_dict[max_class][:max_objects_per_class])
    del class_dict[max_class]

objects = []
for i, class_name in enumerate(classes):
    print(class_name)
    objects.append(objaverse.load_objects(
        uids=object_uids[i],
        download_processes=processes))
    
ex_list = []
max_verts  = 0
max_faces = 0
for i, class_name in enumerate(classes):
  for k, (uid,path) in enumerate(objects[i].items()):
    mesh = trimesh.load(path)
    vertices, faces = data_utils.read_obj_file(mesh.export(file_type="OBJ").splitlines())
    mesh_dict = data_utils.process_mesh(vertices, faces, quantization_bits=8)
    if(mesh_dict['vertices'].shape[0] > max_verts):
      max_verts = mesh_dict['vertices'].shape[0]
    if(mesh_dict['faces'].shape[0] > max_faces):
      max_faces = mesh_dict['faces'].shape[0]
    mesh_dict['class_label'] = i
    ex_list.append(mesh_dict)
synthetic_dataset = tf.data.Dataset.from_generator(
    lambda: ex_list, 
    output_types={
        'vertices': tf.int32, 'faces': tf.int32, 'class_label': tf.int32},
    output_shapes={
        'vertices': tf.TensorShape([None, 3]), 'faces': tf.TensorShape([None]), 
        'class_label': tf.TensorShape(())}
    )
ex = synthetic_dataset.make_one_shot_iterator().get_next()

# Inspect the first mesh
with tf.Session() as sess:
  ex_np = sess.run(ex)
print(ex_np)

vertex_module_config=dict(
decoder_config=dict(
        hidden_size=512,
        fc_size=2048,
        num_heads=8,
        layer_norm=True,
        num_layers=4,
        dropout_rate=0.4,
        re_zero=True,
        memory_efficient=True
        ),
class_conditional=True,
num_classes=max_classes,
max_num_input_verts=2000,
use_discrete_embeddings=True,
quantization_bits=8,)

face_module_config=dict(
encoder_config=dict(
        hidden_size=512,
        fc_size=2048,
        num_heads=8,
        layer_norm=True,
        num_layers=4,
        dropout_rate=0.2,
        re_zero=True,
        memory_efficient=True,
        ),
    decoder_config=dict(
        hidden_size=512,
        fc_size=2048,
        num_heads=8,
        layer_norm=True,
        num_layers=6,
        dropout_rate=0.2,
        re_zero=True,
        memory_efficient=True,
        ),
class_conditional=False,
max_seq_length=10000, # number of faces in the input mesh, if this is lower than the number of vertices in the mesh, there will be errors during training
decoder_cross_attention=True,
use_discrete_vertex_embeddings=True,
)

# Prepare the dataset for vertex model training
vertex_model_dataset = data_utils.make_vertex_model_dataset(
    synthetic_dataset, apply_random_shift=False)
vertex_model_dataset = vertex_model_dataset.repeat()
vertex_model_dataset = vertex_model_dataset.padded_batch(
    max_classes, padded_shapes=vertex_model_dataset.output_shapes)
vertex_model_dataset = vertex_model_dataset.prefetch(1)
vertex_model_batch = vertex_model_dataset.make_one_shot_iterator().get_next()

# Create vertex model
vertex_model = modules.VertexModel( **vertex_module_config) 

vertex_model_pred_dist = vertex_model(vertex_model_batch)
vertex_model_loss = -tf.reduce_sum(
    vertex_model_pred_dist.log_prob(vertex_model_batch['vertices_flat']) * 
    vertex_model_batch['vertices_flat_mask'])
vertex_samples = vertex_model.sample(
    max_classes, context=vertex_model_batch, max_sample_length=1500, top_p=0.95,
    recenter_verts=False, only_return_complete=False)

print(vertex_model_batch)
print(vertex_model_pred_dist)
print(vertex_samples)

face_model_dataset = data_utils.make_face_model_dataset(
    synthetic_dataset, apply_random_shift=False)
face_model_dataset = face_model_dataset.repeat()
face_model_dataset = face_model_dataset.padded_batch(
    max_classes, padded_shapes=face_model_dataset.output_shapes)
face_model_dataset = face_model_dataset.prefetch(1)
face_model_batch = face_model_dataset.make_one_shot_iterator().get_next()

# Create face model
face_model = modules.FaceModel( **face_module_config )
face_model_pred_dist = face_model(face_model_batch)
face_model_loss = -tf.reduce_sum(
    face_model_pred_dist.log_prob(face_model_batch['faces']) * 
    face_model_batch['faces_mask'])
face_samples = face_model.sample(
    context=vertex_samples, max_sample_length=9000, top_p=0.95,
    only_return_complete=False)
print(face_model_batch)
print(face_model_pred_dist)
print(face_samples)

save_dir = 'saved_models/objaverse/'


# Optimization settings
learning_rate = 5e-4
training_steps = 1000
check_step = 10
# Create an optimizer an minimize the summed log probability of the mesh 
# sequences
optimizer = tf.train.AdamOptimizer(learning_rate)
vertex_model_optim_op = optimizer.minimize(vertex_model_loss)
face_model_optim_op = optimizer.minimize(face_model_loss)
vertex_model_saver = tf.train.Saver(var_list=vertex_model.variables)
face_model_saver = tf.train.Saver(var_list=face_model.variables)

# Training loop
config = tf.ConfigProto(device_count = {'GPU': 0})
with tf.Session(config=config) as sess:
  sess.run(tf.global_variables_initializer())
  for n in range(training_steps):
    if n % check_step == 0:
      v_loss, f_loss = sess.run((vertex_model_loss, face_model_loss))
      print('Step {}'.format(n))
      print('Loss (vertices) {}'.format(v_loss))
      print('Loss (faces) {}'.format(f_loss))
      v_samples_np, f_samples_np, b_np = sess.run(
        (vertex_samples, face_samples, vertex_model_batch))
      mesh_list = []
      for n in range(max_classes):
        mesh_list.append(
            {
                'vertices': v_samples_np['vertices'][n][:v_samples_np['num_vertices'][n]],
                'faces': data_utils.unflatten_faces(
                    f_samples_np['faces'][n][:f_samples_np['num_face_indices'][n]])
            }
        )
      data_utils.plot_meshes(mesh_list, ax_lims=0.5)
    vertex_model_saver.save(sess, save_dir+'vertex_model/model')
    face_model_saver.save(sess, save_dir+'face_model/model')
    sess.run((vertex_model_optim_op, face_model_optim_op))