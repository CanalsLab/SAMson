"""
RBM U-net (zero-padding of slices narrower than 128 voxels added)
RBM U-net (Hsu et al. 2020) inference with the functions of 09_SAMson_rodentUnet.ipynb (in-plane resampling only).
JSON configuration: model_path and jobs (input, output, voxsize).
"""
import os, sys, json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from keras import backend as K
from keras.models import load_model
import SimpleITK as sitk  # imported after Keras/TensorFlow: the reverse order crashes Python in the ben environment


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def min_max_normalization(img):
    new_img = img.copy()
    new_img = new_img.astype(np.float32)
    min_val = np.min(new_img)
    max_val = np.max(new_img)
    new_img = (np.asarray(new_img).astype(np.float32) - min_val) / (max_val - min_val)
    return new_img

def dim_2_categorical(label, num_class):
    dims = label.ndim
    if dims == 2:
        col, row = label.shape
        ex_label = np.zeros((num_class, col, row))
        for i in range(0, num_class):
            ex_label[i, ...] = np.asarray(label == i).astype(np.uint8)
    elif dims == 3:
        leng, col, row = label.shape
        ex_label = np.zeros((num_class, leng, col, row))
        for i in range(0, num_class):
            ex_label[i, ...] = np.asarray(label == i).astype(np.uint8)
    else:
        raise Exception
    return ex_label

def resample_img(imgobj, new_spacing, interpolator, new_size=None):
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(interpolator)
    resample.SetOutputDirection(imgobj.GetDirection())
    resample.SetOutputOrigin(imgobj.GetOrigin())
    resample.SetOutputSpacing(new_spacing)
    if new_size is None:
        orig_size = np.array(imgobj.GetSize(), dtype=int)
        orig_spacing = np.array(imgobj.GetSpacing())
        new_size = orig_size * (orig_spacing / new_spacing)
        new_size = np.ceil(new_size).astype(int)  # Image dimensions are in integers
        new_size = [int(s) for s in new_size]
    resample.SetSize(new_size)
    return resample.Execute(imgobj)

def out_LabelHot_map_2D(img, seg_net, pre_paras, keras_paras):
    patch_dims = pre_paras.patch_dims
    label_dims = pre_paras.patch_label_dims
    strides = pre_paras.patch_strides
    n_class = pre_paras.n_class
    length, col, row = img.shape
    categorical_map = np.zeros((n_class, length, col, row), dtype=np.uint8)
    likelihood_map = np.zeros((length, col, row), dtype=np.float32)
    counter_map = np.zeros((length, col, row), dtype=np.float32)
    length_step = int(patch_dims[0] / 2)
    # predict the whole image from two directions, small to large and large to small
    for i in range(0, length - patch_dims[0] + 1, strides[0]):
        for j in range(0, col - patch_dims[1] + 1, strides[1]):
            for k in range(0, row - patch_dims[2] + 1, strides[2]):
                cur_patch = img[i:i + patch_dims[0], j:j + patch_dims[1], k:k + patch_dims[2]][:].reshape([1, patch_dims[0], patch_dims[1], patch_dims[2]])
                if keras_paras.img_format == 'channels_last':
                    cur_patch = np.transpose(cur_patch, (0, 2, 3, 1))
                cur_patch_output = seg_net.predict(cur_patch, batch_size=1, verbose=0)
                if isinstance(cur_patch_output, list):
                    cur_patch_output = cur_patch_output[keras_paras.outID]
                cur_patch_output = np.squeeze(cur_patch_output)
                cur_patch_out_label = cur_patch_output.copy()
                cur_patch_out_label[cur_patch_out_label >= keras_paras.thd] = 1
                cur_patch_out_label[cur_patch_out_label < keras_paras.thd] = 0
                middle = i + length_step
                cur_patch_out_label = dim_2_categorical(cur_patch_out_label, n_class)
                categorical_map[:, middle, j:j + label_dims[1], k:k + label_dims[2]] = categorical_map[:, middle, j:j + label_dims[1], k:k + label_dims[2]] + cur_patch_out_label
                likelihood_map[middle, j:j + label_dims[1], k:k + label_dims[2]] = likelihood_map[middle, j:j + label_dims[1], k:k + label_dims[2]] + cur_patch_output
                counter_map[middle, j:j + label_dims[1], k:k + label_dims[2]] += 1
    for i in range(length, patch_dims[0] - 1, -strides[0]):
        for j in range(col, patch_dims[1] - 1, -strides[1]):
            for k in range(row, patch_dims[2] - 1, -strides[2]):
                cur_patch = img[i - patch_dims[0]:i, j - patch_dims[1]:j, k - patch_dims[2]:k][:].reshape([1, patch_dims[0], patch_dims[1], patch_dims[2]])
                if keras_paras.img_format == 'channels_last':
                    cur_patch = np.transpose(cur_patch, (0, 2, 3, 1))
                cur_patch_output = seg_net.predict(cur_patch, batch_size=1, verbose=0)
                if isinstance(cur_patch_output, list):
                    cur_patch_output = cur_patch_output[keras_paras.outID]
                cur_patch_output = np.squeeze(cur_patch_output)
                cur_patch_out_label = cur_patch_output.copy()
                cur_patch_out_label[cur_patch_out_label >= keras_paras.thd] = 1
                cur_patch_out_label[cur_patch_out_label < keras_paras.thd] = 0
                middle = i - patch_dims[0] + length_step
                cur_patch_out_label = dim_2_categorical(cur_patch_out_label, n_class)
                categorical_map[:, middle, j - label_dims[1]:j, k - label_dims[2]:k] = categorical_map[:, middle, j - label_dims[1]:j, k - label_dims[2]:k] + cur_patch_out_label
                likelihood_map[middle, j - label_dims[1]:j, k - label_dims[2]:k] = likelihood_map[middle, j - label_dims[1]:j, k - label_dims[2]:k] + cur_patch_output
                counter_map[middle, j - label_dims[1]:j, k - label_dims[2]:k] += 1
    label_map = np.zeros([length, col, row], dtype=np.uint8)
    for idx in range(0, length):
        cur_slice_label = np.squeeze(categorical_map[:, idx, ].argmax(axis=0))
        label_map[idx, ] = cur_slice_label
    counter_map = np.maximum(counter_map, 10e-10)
    likelihood_map = np.divide(likelihood_map, counter_map)
    return label_map, likelihood_map

class KerasParas:
    def __init__(self):
        self.model_path = None
        self.outID = 0                          # indicating which output we need when dealing with multi-output network
        self.thd = 0.5
        self.img_format = 'channels_first'
        self.loss = None

class PreParas:
    def __init__(self):
        self.patch_dims = []
        self.patch_label_dims = []
        self.patch_strides = []
        self.n_class = ''

_models = {}
def brain_seg_prediction(input_path, output_path, voxsize, pre_paras, keras_paras):
    if keras_paras.model_path not in _models:  # loaded once (the original loaded it again for every subject)
        _models[keras_paras.model_path] = load_model(keras_paras.model_path, custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
    seg_net = _models[keras_paras.model_path]
    imgobj = sitk.ReadImage(input_path)
    original_spacing = imgobj.GetSpacing()  # Original spacing of the input image
    # Set new_spacing to a 3-element list, preserving the slice thickness
    new_spacing = [voxsize, voxsize, original_spacing[2]]
    resampled_imgobj = resample_img(imgobj, new_spacing=new_spacing, interpolator=sitk.sitkLinear)
    img_array = sitk.GetArrayFromImage(resampled_imgobj)
    normed_array = min_max_normalization(img_array)
    # Zero-padding of slices narrower than one patch (array axes: slice, row, column); cropped back after the prediction
    rows, cols = normed_array.shape[1:]
    pad = [(0, 0), (0, max(0, pre_paras.patch_dims[1] - rows)), (0, max(0, pre_paras.patch_dims[2] - cols))]
    out_label_map, out_likelihood_map = out_LabelHot_map_2D(np.pad(normed_array, pad, mode="constant"), seg_net, pre_paras, keras_paras)
    out_label_map = out_label_map[:, :rows, :cols]
    out_label_img = sitk.GetImageFromArray(out_label_map.astype(np.uint8))
    out_label_img.CopyInformation(resampled_imgobj)
    resampled_label_map = resample_img(out_label_img, new_spacing=imgobj.GetSpacing(), new_size=imgobj.GetSize(), interpolator=sitk.sitkNearestNeighbor)
    sitk.WriteImage(resampled_label_map, output_path)


if __name__ == "__main__":
    cfg = json.load(open(sys.argv[1]))
    # Default Parameters Preparation
    pre_paras = PreParas()
    pre_paras.patch_dims = [1, 128, 128]
    pre_paras.patch_label_dims = [1, 128, 128]
    pre_paras.patch_strides = [1, 32, 32]
    pre_paras.n_class = 2
    # Parameters for Keras model
    keras_paras = KerasParas()
    keras_paras.outID = 0
    keras_paras.thd = 0.5
    keras_paras.loss = 'dice_coef_loss'
    keras_paras.img_format = 'channels_last'
    keras_paras.model_path = cfg["model_path"]
    for job in cfg["jobs"]:
        brain_seg_prediction(job["input"], job["output"], job["voxsize"], pre_paras, keras_paras)
        print(os.path.basename(job["output"]), flush=True)
