
#%%
vertex_model_pred_dist = vertex_model(vertex_model_batch)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    text_vars = vertex_model.variables
print(text_vars)
vertex_model1_pred_dist = vertex_model1(vertex_model_batch)
with tf.Session() as sess:
    orig_vars = vertex_model1.variables
print(orig_vars)
text_vars_names = []
for v in text_vars:
    # shape = tuple()
    # for dim in v._variable._shape_val._dims:
    #     shape = shape+(dim._value,)
    # text_vars_names.append(('/'.join((v._variable._name).split('/')[1:]), shape))
    text_vars_names.append('/'.join((v._variable._name).split('/')[1:]))
print(text_vars_names)
orig_vars_names = []
for v in orig_vars:
    # shape = tuple()
    # for dim in v._variable._shape_val._dims:
    #     shape = shape+(dim._value,)
    # orig_vars_names.append(('/'.join((v._variable._name).split('/')[1:]), shape))
    orig_vars_names.append('/'.join((v._variable._name).split('/')[1:]))
print(orig_vars_names)
common_vars = set(text_vars_names) & set(orig_vars_names)
import pickle
with open("pretrained_vars.pickle", 'wb') as out:
    pickle.dump(common_vars, out)