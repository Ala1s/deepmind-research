ex_list = []
max_verts  = 0
max_faces = 0
for k, (uid,path) in enumerate(objects.items()):
  print("current object = ", k)
  mesh_dict = data_utils.process_mesh_trimesh(list(trimesh.load(path).geometry.values())[0])
  if(mesh_dict['vertices'].shape[0] > max_verts):
    max_verts = mesh_dict['vertices'].shape[0]
  if(mesh_dict['faces'].shape[0] > max_faces):
    max_faces = mesh_dict['faces'].shape[0]
  mesh_dict['class_label'] = k
  ex_list.append(mesh_dict)
synthetic_dataset = tf.data.Dataset.from_generator(
    lambda: ex_list, 
    output_types={
        'vertices': tf.int32, 'faces': tf.int32, 'class_label': tf.int32},
    output_shapes={
        'vertices': tf.TensorShape([None, 3]), 'faces': tf.TensorShape([None]), 
        'class_label': tf.TensorShape(())}
    )
ex = tf.compat.v1.data.make_one_shot_iterator(synthetic_dataset).get_next()

# Inspect the first mesh
with tf.compat.v1.Session() as sess:
  ex_np = sess.run(ex)
print(ex_np)

# Plot the meshes
mesh_list = []
with tf.compat.v1.Session() as sess:
  for i in range(4):
    ex_np = sess.run(ex)
    mesh_list.append(
        {'vertices': data_utils.dequantize_verts(ex_np['vertices']), 
         'faces': data_utils.unflatten_faces(ex_np['faces'])})
data_utils.plot_meshes(mesh_list, ax_lims=0.4)