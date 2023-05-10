# SegMerlin
Image semantic segmentation using LightGBM

SegMerlinWLGBMS, is a machine learning model for image segmentation. It has various methods for fitting and predicting segmentation masks on images. The model uses feature extractor backbone and LightGBM models for segmentation.

The constructor of the class initializes the model, and takes three arguments:

feature_extractor: a feature extractor backbone for the model, if one is used
init_model: an initialized model object, if one is being loaded from a file
input_shape: the shape of the input images
The add_predictor() method is used to add a new LightGBM model to the ensemble. It takes several arguments:

Name: the name of the predictor
tile_shape: the shape of the tiles to be extracted from images for segmentation
tile_thresh: a threshold for determining if a tile is foreground or background
lgbm_params: parameters for the LightGBM model
pre_process_func: a function for preprocessing images before segmentation
pos_process_func: a function for postprocessing segmentation masks
The fit_predictor() method is used to fit a predictor to a set of training images and masks. It takes several arguments:

predictor_names: a list of names of the predictors to fit
images: a list of images for training
masks: a list of masks for training
fit_type: the type of fitting to perform ('refit', 'update', or 'train')
nb_rounds_for_trainfit: the number of rounds to use for fitting the model
The predict() method is used to generate segmentation masks for a set of input images. It takes several arguments:

image_list: a list of images to segment
predictor_names: a list of names of the predictors to use for segmentation
filter_strength: a strength for filtering the segmentation masks
predictor_weights: a list of weights for each predictor, if different weights are desired
voting_type: the type of voting to use for combining the masks

# SegMerlinWLGBMS

`SegMerlinWLGBMS` is a class that provides a framework for training and using a segmentation model based on LightGBM. The model uses a sliding window approach to segment large images and can be trained with both images and corresponding masks. 

## Usage

To use the `SegMerlinWLGBMS` class, import it into your Python script and create an instance:

from SegMerlinWLGBMS import SegMerlinWLGBMS

model = SegMerlinWLGBMS(feature_extractor=my_feature_extractor, init_model=my_init_model, input_shape=(720, 1280))

### Initialization Parameters

- `feature_extractor`: (optional) A callable object that takes a list of images and returns a list of feature vectors. The default is `None`.
- `init_model`: (optional) The file path to a pre-trained `SegMerlinWLGBMS` model. The default is `None`.
- `input_shape`: The input shape of the images. The default is `(720, 1280)`.

### Methods

- `add_predictor(Name, tile_shape, tile_thresh=0.8, lgbm_params=LGBM_DEFAULTS, pre_process_func=None, pos_process_func=None)`: Adds a new predictor to the model.
- `fit_predictor(predictor_names, images, masks, fit_type='refit', nb_rounds_for_trainfit=None, verbose=0)`: Fits the specified predictors on the given images and masks.
- `predict(image_list, predictor_names, filter_strength=255/1.5, predictor_weights=None, voting_type='majority')`: Segments the given images using the specified predictors.

## Example

Here's an example of how to use `SegMerlinWLGBMS` to segment an image:

from SegMerlinWLGBMS import SegMerlinWLGBMS

Create a model instance
model = SegMerlinWLGBMS(feature_extractor=my_feature_extractor, input_shape=(720, 1280))

Add a predictor
model.add_predictor(Name='my_predictor', tile_shape=(256, 256), tile_thresh=0.8, lgbm_params=LGBM_DEFAULTS)

Fit the predictor
model.fit_predictor(predictor_names=['my_predictor'], images=[my_image], masks=[my_mask], fit_type='train', nb_rounds_for_trainfit=100)

Segment the image
segmentation = model.predict(image_list=[my_image], predictor_names=['my_predictor'], filter_strength=255/1.5)
