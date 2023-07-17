#!/usr/bin/env python
# coding: utf-8

# In[1]:


from glob import glob
from tqdm import tqdm
import pandas as pd
import tensorflow.compat.v1 as tf
tf.logging.set_verbosity(tf.logging.ERROR)  # Hide TF deprecation messages
import matplotlib.pyplot as plt
import trimesh
import random
import numpy as np
import pickle
import modules
import data_utils


# In[2]:


import datetime
from tensorflow import summary as s
log_dir = "logs/text/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# summary_writer = s.FileWriter(log_dir)


# In[3]:


BATCH_SIZE=2


# In[4]:


chair_meshes_paths = list(glob("chairs_ngon/*.obj"))
# chair_meshes_paths = [["Chair", path] for path in chair_meshes_paths]
tables_meshes_paths = list(glob("tables_ngon/*.obj"))
# tables_meshes_paths = [["Table", path] for path in tables_meshes_paths]
# chair_meshes_paths.extend(tables_meshes_paths)
# paths = chair_meshes_paths.copy()
# random.shuffle(paths)


# In[5]:


with open("chairs_split_dict.pickle", 'rb') as f:
    chairs_split_dict = pickle.load(f)


# In[6]:


with open("tables_split_dict.pickle", 'rb') as f:
    tables_split_dict = pickle.load(f)


# In[7]:


chairs_train = []
chairs_val = []
chairs_test = []
for c in chair_meshes_paths: 
    try:
        split = chairs_split_dict[c.split("/")[-1].replace(".obj", "")]
    except KeyError:
        print(c.split("/")[-1].replace(".obj", ""))
        continue
    if split =='train':
        chairs_train.append(c)
    elif split =='val':
        chairs_val.append(c)
    else:
        chairs_test.append(c)
print(len(chairs_train))
print(len(chairs_val))
print(len(chairs_test))


# In[8]:


tables_train = []
tables_val = []
tables_test = []
for c in tables_meshes_paths: 
    try:
        split = tables_split_dict[c.split("/")[-1].replace(".obj", "")]
    except KeyError:
        print(c.split("/")[-1].replace(".obj", ""))
        continue
    if split =='train':
        tables_train.append(c)
    elif split =='val':
        tables_val.append(c)
    else:
        tables_test.append(c)
print(len(tables_train))
print(len(tables_val))
print(len(tables_test))


# In[9]:


chairs_train.extend(tables_train)
train_paths = chairs_train.copy()
random.shuffle(train_paths)
# train_paths = train_paths[:10]


# In[10]:


TRAIN_SIZE = len(train_paths)


# In[11]:


chairs_val.extend(tables_val)
val_paths = chairs_val.copy()
random.shuffle(val_paths)


# In[12]:


VAL_SIZE = len(val_paths)


# In[13]:


captions = pd.read_csv("captions_tablechair.csv").dropna()


# In[14]:


captions


# In[15]:


max_length = 0
for c in captions['description'].values:
    cur = len(c.split(" "))
    if cur>max_length:
        max_length =cur
max_length


# In[16]:


train_captions=[]
for index, row in captions.iterrows():
    try:
        if row["category"]=="Table":
            if tables_split_dict[row["modelId"]]=='train':
                train_captions.append(row['description'])
        if row["category"]=="Chair":
            if chairs_split_dict[row["modelId"]]=='train':
                train_captions.append(row['description'])
    except KeyError:
        continue
print(len(train_captions))


# In[17]:


from tensorflow.keras.preprocessing.text import Tokenizer
tk = Tokenizer()
tk.fit_on_texts(train_captions)


# In[18]:


def text2shape(paths, captions, tokenizer):
    for path in paths:
        # with open(path, 'rb') as obj_file:
        mesh_dict = data_utils.load_process_mesh(path)
#         mesh_dict['class_label'] = 18 if cls=="Chair" else 49
        if len(mesh_dict['faces'])>2600:
            continue
        try:
            text = captions[captions["modelId"]==path.split("/")[-1].replace(".obj", "")].sample(n=1)["description"].values[0]
        except:
            continue
        text = text.lower().replace("the", '').replace("a", '').replace("of", '').replace("for", '').replace("and", '').replace("to", '').replace("in", '')
        text = tokenizer.texts_to_sequences([text])[0]
        mesh_dict['text_feature'] = np.pad(text, (0,max_length-len(text)))
        yield mesh_dict


# In[19]:


Text2ShapeDataset = tf.data.Dataset.from_generator(
        lambda:text2shape(train_paths, captions, tk),
        output_types={
            'vertices': tf.int32, 'faces': tf.int32,
#             'class_label': tf.int32,
            'text_feature': tf.int32},
        output_shapes={
            'vertices': tf.TensorShape([None, 3]), 'faces': tf.TensorShape([None]),
#             'class_label': tf.TensorShape(()),
            'text_feature':tf.TensorShape(140)})
ex = Text2ShapeDataset.make_one_shot_iterator().get_next()
ex


# In[20]:


vertex_model_dataset = data_utils.make_vertex_model_dataset(
    Text2ShapeDataset, apply_random_shift=False)
vertex_model_dataset = vertex_model_dataset.repeat()
vertex_model_dataset = vertex_model_dataset.padded_batch(
    BATCH_SIZE, padded_shapes=vertex_model_dataset.output_shapes)
vertex_model_dataset = vertex_model_dataset.prefetch(1)
it = vertex_model_dataset.make_initializable_iterator()
vertex_model_batch = it.get_next()
iterator_init_op_train = it.initializer


# In[21]:


Text2ShapeDatasetVal = tf.data.Dataset.from_generator(
        lambda:text2shape(val_paths, captions, tk),
        output_types={
            'vertices': tf.int32, 'faces': tf.int32,
#             'class_label': tf.int32,
            'text_feature': tf.int32},
        output_shapes={
            'vertices': tf.TensorShape([None, 3]), 'faces': tf.TensorShape([None]),
#             'class_label': tf.TensorShape(()),
            'text_feature':tf.TensorShape(140)})
vertex_model_dataset_val = data_utils.make_vertex_model_dataset(
    Text2ShapeDatasetVal, apply_random_shift=False)
vertex_model_dataset_val = vertex_model_dataset_val.repeat()
vertex_model_dataset_val = vertex_model_dataset_val.padded_batch(
    BATCH_SIZE, padded_shapes=vertex_model_dataset_val.output_shapes)
vertex_model_dataset_val = vertex_model_dataset_val.prefetch(1)
itv = vertex_model_dataset_val.make_initializable_iterator()
vertex_model_batch_val = itv.get_next()
iterator_init_op_val = itv.initializer


# In[22]:


# with tf.Session() as sess:
#     ex_np = sess.run(ex)
# print(ex_np)

# # Plot the meshes
# mesh_list = []
# with tf.Session() as sess:
#     for i in range(8):
#         ex_np = sess.run(ex)
#         mesh_list.append(
#         {'vertices': data_utils.dequantize_verts(ex_np['vertices']),
#          'faces': data_utils.unflatten_faces(ex_np['faces'])})
# data_utils.plot_meshes(mesh_list, ax_lims=0.4)


# In[23]:


vertex_model = modules.TextToVertexModel(
    decoder_config=dict(
      hidden_size=512,
      fc_size=2048,
      num_heads=8,
      layer_norm=True,
      num_layers=24,
      dropout_rate=0.4,
      re_zero=True,
      memory_efficient=True
      ),
    path_to_embeddings="glove.42B.300d.txt",
    embedding_dims = 300,
    quantization_bits=8,
    tokenizer=tk,
    max_num_input_verts=5000,  # number of vertices in the input mesh, if this is lower than the number of vertices in the mesh, there will be errors during training
    use_discrete_embeddings=True
)


# In[24]:


vertex_model_pred_dist = vertex_model(vertex_model_batch)
vertex_model_loss = -tf.reduce_sum(
    vertex_model_pred_dist.log_prob(vertex_model_batch['vertices_flat']) *
    vertex_model_batch['vertices_flat_mask'])
vertex_samples = vertex_model.sample(
    BATCH_SIZE, context=vertex_model_batch, max_sample_length=1000, top_p=0.95,
    recenter_verts=False, only_return_complete=False)

print(vertex_model_batch)
print(vertex_model_pred_dist)


# In[25]:


vertex_model_pred_dist_val = vertex_model(vertex_model_batch_val)
vertex_model_loss_val = -tf.reduce_sum(
    vertex_model_pred_dist_val.log_prob(vertex_model_batch_val['vertices_flat']) *
    vertex_model_batch_val['vertices_flat_mask'])
vertex_samples_val = vertex_model.sample(
    BATCH_SIZE, context=vertex_model_batch_val, max_sample_length=1000, top_p=0.95,
    recenter_verts=False, only_return_complete=False)


# In[26]:


# vertex_module_config=dict(
#   decoder_config=dict(
#       hidden_size=512,
#       fc_size=2048,
#       num_heads=8,
#       layer_norm=True,
#       num_layers=24,
#       dropout_rate=0.4,
#       re_zero=True,
#       memory_efficient=True
#       ),
#   quantization_bits=8,
#   class_conditional=True,
#   max_num_input_verts=5000,
#   use_discrete_embeddings=True,
#   )
# vertex_model1 = modules.VertexModel(**vertex_module_config)


# In[27]:


# vertex_model_pred_dist = vertex_model(vertex_model_batch)
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     text_vars = vertex_model.variables
# print(text_vars)


# In[28]:


# vertex_model1_pred_dist = vertex_model1(vertex_model_batch)
# with tf.Session() as sess:
#     orig_vars = vertex_model1.variables
# print(orig_vars)


# In[29]:


# text_vars_names = []
# for v in text_vars:
# #     shape = tuple()
# #     for dim in v._variable._shape_val._dims:
# #         shape = shape+(dim._value,)
# #     text_vars_names.append(('/'.join((v._variable._name).split('/')[1:]), shape))
#     text_vars_names.append('/'.join((v._variable._name).split('/')[1:]))
# print(text_vars_names)

#'coord_embeddings/embeddings:0', 'coord_embeddings_1/embeddings:0', 'embed/embeddings:0', 'embed_zero:0', 'linear/b:0', 'linear/w:0', 'linear_1/b:0', 'linear_1/w:0'


# In[30]:


# orig_vars_names = []
# for v in orig_vars:
# #     shape = tuple()
# #     for dim in v._variable._shape_val._dims:
# #         shape = shape+(dim._value,)
# #     orig_vars_names.append(('/'.join((v._variable._name).split('/')[1:]), shape))
#     orig_vars_names.append('/'.join((v._variable._name).split('/')[1:]))
# print(orig_vars_names)


# In[31]:


# common_vars = set(text_vars_names) & set(orig_vars_names)
# print(common_vars)
# only_text_vars = set(text_vars_names) - common_vars
# print(only_text_vars)


# In[32]:


# only_text_vars = set(text_vars_names) - common_vars
# print(only_text_vars)


# In[33]:


# set(orig_vars_names) - common_vars


# In[34]:


# import pickle
# with open("only_text_vars.pickle", 'wb') as out:
#     pickle.dump(only_text_vars, out)


# In[35]:


with open("only_text_vars.pickle", 'rb') as f:
    only_text_vars = pickle.load(f)

text_vars = []
for var in vertex_model.variables:
    if '/'.join((var._variable._name).split('/')[1:]) in only_text_vars:
        text_vars.append(var)
text_vars=tuple(text_vars)
# print(text_vars)


# In[36]:


import pickle
with open("pretrained_vars.pickle", 'rb') as f:
    common_vars = pickle.load(f)

pretrained_vars = []
for var in vertex_model.variables:
    if '/'.join((var._variable._name).split('/')[1:]) in common_vars:
        pretrained_vars.append(var)
pretrained_vars=tuple(pretrained_vars)
# print(pretrained_vars)


# In[37]:


vertex_model_saver = tf.train.Saver(var_list=pretrained_vars)


# In[38]:


# with tf.Session() as sess:
#   vertex_model_saver.restore(sess, "D:\\PyCharmProjects\\vertex_model\\model")


# In[39]:


face_model_dataset = data_utils.make_face_model_dataset(
    Text2ShapeDataset, apply_random_shift=False)
face_model_dataset = face_model_dataset.repeat()
face_model_dataset = face_model_dataset.padded_batch(
    BATCH_SIZE, padded_shapes=face_model_dataset.output_shapes)
face_model_dataset = face_model_dataset.prefetch(1)
face_model_batch = face_model_dataset.make_one_shot_iterator().get_next()

# Create face model
face_model = modules.FaceModel(
      encoder_config=dict(
      hidden_size=512,
      fc_size=2048,
      num_heads=8,
      layer_norm=True,
      num_layers=10,
      dropout_rate=0.2,
      re_zero=True,
      memory_efficient=True,
      ),
  decoder_config=dict(
      hidden_size=512,
      fc_size=2048,
      num_heads=8,
      layer_norm=True,
      num_layers=14,
      dropout_rate=0.2,
      re_zero=True,
      memory_efficient=True,
      ),
    class_conditional=False,
    max_seq_length=8000, # number of faces in the input mesh, if this is lower than the number of vertices in the mesh, there will be errors during training
    quantization_bits=8,
    decoder_cross_attention=True,
    use_discrete_vertex_embeddings=True,
)
face_model_pred_dist = face_model(face_model_batch)
face_model_loss = -tf.reduce_sum(
    face_model_pred_dist.log_prob(face_model_batch['faces']) *
    face_model_batch['faces_mask'])
face_samples = face_model.sample(
    context=vertex_samples, max_sample_length=1000, top_p=0.95,
    only_return_complete=False)
print(face_model_batch)
print(face_model_pred_dist)
print(face_samples)


# In[40]:


face_model_dataset_val = data_utils.make_face_model_dataset(
    Text2ShapeDatasetVal, apply_random_shift=False)
face_model_dataset_val  = face_model_dataset_val .repeat()
face_model_dataset_val  = face_model_dataset_val .padded_batch(
    BATCH_SIZE, padded_shapes=face_model_dataset_val.output_shapes)
face_model_dataset_val = face_model_dataset_val.prefetch(1)
face_model_batch_val = face_model_dataset_val.make_one_shot_iterator().get_next()

face_samples_val = face_model.sample(
    context=vertex_samples_val, max_sample_length=1000, top_p=0.95,
    only_return_complete=False)


# In[41]:


face_model_saver = tf.train.Saver(var_list=face_model.variables)


# In[42]:


text_model_saver = tf.train.Saver(var_list=vertex_model.variables, keep_checkpoint_every_n_hours=1, max_to_keep=10)


# In[ ]:





# In[ ]:


from tqdm import trange
import os

with tf.variable_scope("metrics"):
    metrics = {
            'loss': tf.metrics.mean(vertex_model_loss),
            }
update_metrics_op = tf.group(*[op for _, op in metrics.values()])
metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
metrics_init_op = tf.variables_initializer(metric_variables)

tf.summary.scalar('loss', vertex_model_loss)
summary_op = tf.summary.merge_all()



# text_model_saver = tf.train.Saver(var_list=vertex_model.variables, keep_checkpoint_every_n_hours=1, max_to_keep=10)
last_saver = tf.train.Saver(var_list=vertex_model.variables) # will keep last 5 epochs
best_saver = tf.train.Saver(var_list=vertex_model.variables, max_to_keep=2)  # only keep 1 best checkpoint (best on eval)

#get_ipython().run_line_magic('matplotlib', 'inline')
learning_rate = 5e-4
training_steps = 500
check_step_metrics = 10
check_step_samples = 100
EPOCHS = 20
optimizer = tf.train.AdamOptimizer(learning_rate)
vertex_model_optim_op = optimizer.minimize(vertex_model_loss, var_list=text_vars)
best_v_loss = float('inf')
# Training loop
global_step = tf.train.create_global_step()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    vertex_model_saver.restore(sess, "vertex_model/model")
    face_model_saver.restore(sess, 'face_model/model')
    
    train_writer = s.FileWriter(os.path.join(log_dir, 'train_summaries'), sess.graph)
    eval_writer = s.FileWriter(os.path.join(log_dir, 'eval_summaries'), sess.graph)
    
    for e in range(EPOCHS):
        print("Epoch {}/{}".format(e + 1, EPOCHS))
        num_steps = (TRAIN_SIZE + BATCH_SIZE - 1) // BATCH_SIZE
        global_step = tf.train.get_global_step()
        sess.run(iterator_init_op_train)
        sess.run(metrics_init_op)
        t = trange(num_steps)
        for i in t:
            
            #fhjghvkerl
            if i % check_step_metrics == 0:
                _, _, loss_val, summ, global_step_val = sess.run([vertex_model_optim_op, update_metrics_op, vertex_model_loss,
                                                              summary_op, global_step])
                #_, _, loss_val, global_step_val = sess.run([vertex_model_optim_op, update_metrics_op, vertex_model_loss, global_step])
#                 sess.run(vertex_model_optim_op)
#                 sess.run(update_metrics_op)
#                 loss_val = sess.run(vertex_model_loss)
#                 global_step_val = sess.run(global_step)
                train_writer.add_summary(summ, global_step_val)
            else:
                _, _, loss_val = sess.run([vertex_model_optim_op, update_metrics_op, vertex_model_loss])
            # Log the loss in the tqdm progress bar
            t.set_postfix(loss='{:05.3f}'.format(loss_val))
        metrics_values = {k: v[0] for k, v in metrics.items()}
        metrics_val = sess.run(metrics_values)
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_val.items())
        print("- Train metrics: " + metrics_string)
        
        last_save_path = os.path.join("text_generaization", 'last')
        last_saver.save(sess, last_save_path, global_step=e + 1)
    
        num_steps = (VAL_SIZE + BATCH_SIZE - 1) // BATCH_SIZE
        eval_metrics = metrics
        sess.run(iterator_init_op_val)
        sess.run(metrics_init_op)
        for _ in range(num_steps):
            sess.run(update_metrics_op)
        metrics_values = {k: v[0] for k, v in eval_metrics.items()}
        metrics_val = sess.run(metrics_values)
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_val.items())
        print("- Eval metrics: " + metrics_string)
        global_step_val = sess.run(global_step)
        for tag, val in metrics_val.items():
            summ = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=val)])
            eval_writer.add_summary(summ, global_step_val)
        
        if metrics_val['loss']<=best_v_loss:
            best_v_loss = metrics_val['loss']
            best_save_path = os.path.join('text_generaization', 'best')
            best_save_path = best_saver.save(sess, best_save_path, global_step=e + 1)
            print("- Found new best model, saving in {}".format(best_save_path))
                
#         #SAmples
#         if e>2:
#             sess.run(iterator_init_op_val)
#             v_samples_np, f_samples_np, b_np = sess.run((vertex_samples_val, face_samples_val, vertex_model_batch_val))
#             mesh_list = []
#             for n in range(BATCH_SIZE):
#                 mesh_list.append({
#                     'vertices': v_samples_np['vertices'][n][:v_samples_np['num_vertices'][n]],
#                     'faces': data_utils.unflatten_faces(
#                         f_samples_np['faces'][n][:f_samples_np['num_face_indices'][n]])
#                     })
#             data_utils.plot_meshes(mesh_list, ax_lims=0.5)
                


# In[ ]:


# for k, v in metrics.items():
#     print(v)


# In[ ]:


# from tqdm import tqdm
# %matplotlib inline 
# learning_rate = 5e-4
# training_steps = 1500
# check_step_metrics = 10
# check_step_samples = 300

# # Create an optimizer an minimize the summed log probability of the mesh
# # sequences
# optimizer = tf.train.AdamOptimizer(learning_rate)
# vertex_model_optim_op = optimizer.minimize(vertex_model_loss, var_list=text_vars)
# best_v_loss = float('inf')
# # Training loop
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     vertex_model_saver.restore(sess, "vertex_model/model")
#     face_model_saver.restore(sess, 'face_model/model')
#     for i in range(training_steps):
#         if i % check_step_metrics == 0:
#             v_loss_train = sess.run(vertex_model_loss)
# #             with summary_writer:
# #                 tf.summary.scalar('vertex_loss_train', v_loss_train)
#             v_loss_val = sess.run(vertex_model_loss_val)
# #             with summary_writer:
# #                 tf.summary.scalar('vertex_loss_val', v_loss_val)
#             if v_loss_val<best_v_loss:
#                 text_model_saver.save(sess, 'text_model/', global_step=i)
#                 best_v_loss = v_loss_val
#                 print("saved best model")
        
#             print('Step {}'.format(i))
#             print('Loss vertices train {}'.format(v_loss_train))
#             print('Loss vertices val {}'.format(v_loss_val))
#         if i % check_step_samples == 0:
#             v_samples_np, f_samples_np, b_np = sess.run((vertex_samples_val, face_samples_val, vertex_model_batch_val))
#             mesh_list = []
#             for n in range(BATCH_SIZE):
#                 mesh_list.append({
#                 'vertices': v_samples_np['vertices'][n][:v_samples_np['num_vertices'][n]],
#                 'faces': data_utils.unflatten_faces(
#                     f_samples_np['faces'][n][:f_samples_np['num_face_indices'][n]])
#                 })
#             data_utils.plot_meshes(mesh_list, ax_lims=0.5)
#         sess.run(vertex_model_optim_op)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




