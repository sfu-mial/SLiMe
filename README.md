# SLiMe: Segment Like Me
**[Paper](https://arxiv.org/abs/2309.03179)**

PyTorch implementation of SLiMe: Segment Like Me, a 1-shot image segmentation method based on Stable Diffusion. <br><br>
[Aliasghar Khani<sup>1, 2</sup>](https://aliasgharkhani.github.io/), [Saeid Asgari Taghanaki<sup>2</sup>](https://asgsaeid.github.io/), [Aditya Sanghi<sup>2</sup>](https://www.research.autodesk.com/people/aditya-sanghi/), [Ali Mahdavi Amiri<sup>1</sup>](https://www.sfu.ca/~amahdavi/), [Ghassan Hamarneh<sup>1</sup>](https://www.medicalimageanalysis.com/)

<sup><sup>1</sup> Simon Fraser University  <sup>2</sup> Autodesk Research</sup>

# Setup
To start using SLiMe, first you need to create a virtual environment and install the dependencies using these commands:
```
python -m venv slime_venv
source slime_venv/bin/activate
pip install -r requirements.txt
```

*** ***For every image and mask pair that is used for training, validation, or testing SLiMe, their names should be the same. Moreover the images should be in `PNG` format and the masks should be in `numpy` format.*** ***

# SLiMe training
First you need to create a new folder (e.g., `slime/data/train`), and put the train images together with their masks in that folder (`slime/data/train`). Then pass the address of the folder you created (`slime/data/train`) to `--train_data_dir`. If you have validation data, which will be only used for checkpoint selection, do the same for validation data (e.g., put the images and masks in `slime/data/val`) and pass the folders address to `--val_data_dir` . But if you don't have validation data, pass the address of the train data folder to `--val_data_dir`. Afterwards, put the test images in another folder (e.g., `slime/data/test`) and pass the folder's path to `--test_data_dir`. You should also pass a name for the segmented parts in the train images to `--parts_to_return`, including background. For example if you have segmented body and head of a dog, you should pass `"background body head"` to this argument. Finally, run this command in `slime` folder (the main folder that you have after cloning):
```
python -m src.main --dataset sample \
                   --part_names {PARTNAMES} \
                   --train_data_dir {TRAIN_DATA_DIR} \
                   --val_data_dir {TRAIN_DATA_DIR} \
                   --test_data_dir {TEST_DATA_DIR} \
                   --train \
```
If you have provided test images together with their masks, this command will print the mIoU of each of the segmented parts on the test data. Additionally, the trained text embeddings and the log files will be saved in `slime/outputs/checkpoints` and `slime/outputs/lightning_logs` folders in `slime`, respectively.

# Testing with the trained text embeddings
To use trained text embeddings for testing, run this command:
```
python -m src.main --dataset sample \
                   --checkpoint_dir {CHECKPOINT_DIR} \
                   --test_data_dir {TEST_DATA_DIR} \
```
where you should pass the address of the folder where trained text embeddings exist to `--checkpoint_dir`. Please note that there should not be other text embeddings than those you want to be used, because the code will load all the existing text embeddings in the passed directory. Moreover, like the previous section, you need to put the test images (and there masks, if exist, for calculating mIoU) in a new folder and pass that folder's path to `--target_image_paths`. 

## Patchifying the Image
For running any of the commands above, you can pass different values to `--patch_size` and `--num_patchs_per_side`, which will be used for the validation and testing steps. By passing `--patch_size` and `--num_patchs_per_side`, you want the method to patchify the image into `"num_patchs_per_side"` pathces of size `"patch_size"`, get there final attention maps (**WAS-attention maps**) separately, aggregate them, and generate the segmentation mask prediction.

# 1-sample and 10-sample training on datasets
## PASCAL-Part Car
To train and test with the 1-sample setting of SLiMe on the car class of PASCAL-Part, first download the data from [here]() and extract it. Then run the following command in `slime` folder:
```
DATADIR={}
python3 -m src.main --dataset_name pascal \
                    --part_names background body light plate wheel window \
                    --train_data_dir $DATADIR/car/train_1 \
                    --val_data_dir $DATADIR/car/train_1 \
                    --test_data_dir $DATADIR/car/test \
                    --min_crop_ratio 0.6 \
                    --num_patchs_per_side 1 \
                    --patch_size 512 \
                    --train \
```
where you should pass the path of the folder which, you extracted and contains `car/train_1` and `car/test`, to `DATADIR`, without a `\` at the end. For the 10-sample setting, you need to pass `car/train_10` instead of `car/train_1` to `--train_data_dir` and pass `car/val` to `--val_data_dir`.

## PASCAL-Part Horse
To train and test with the 1-sample setting of SLiMe on the horse class of PASCAL-Part, first download the data from [here]() and extract it. Then run the following command in `slime` folder:
```
DATADIR={}
python3 -m src.main --dataset_name pascal \
                    --part_names background head neck+torso leg tail \
                    --train_data_dir $DATADIR/horse/train_1 \
                    --val_data_dir $DATADIR/horse/train_1 \
                    --test_data_dir $DATADIR/horse/test \
                    --min_crop_ratio 0.8 \
                    --num_patchs_per_side 1 \
                    --patch_size 512 \
                    --train \
```
where you should pass the path of the folder, which you extracted and contains `horse/train_1` and `horse/test`, to `DATADIR`, without a `\` at the end. For the 10-sample setting, you need to pass `horse/train_10` instead of `horse/train_1` to `--train_data_dir` and pass `horse/val` to `--val_data_dir`.

## CelebAMask-HQ
To train and test with the 1-sample setting of SLiMe on CelebAMask-HQ, first download the data from [here]() and extract it. Then run the following command in `slime` folder:
```
DATADIR={}
python3 -m src.main --dataset_name celeba \
                    --part_names background skin eye mouth nose brow ear neck cloth hair \
                    --train_data_dir $DATADIR/celeba/train_1 \
                    --val_data_dir $DATADIR/celeba/train_1 \
                    --test_data_dir $DATADIR/celeba/test \
                    --min_crop_ratio 0.6 \
                    --train \
```
where you should pass the path of the folder, which you extracted and contains `celeba/train_1` and `celeba/test`, to `DATADIR`, without a `\` at the end. For the 10-sample setting, you need to pass `celeba/train_10` instead of `celeba/train_1` to `--train_data_dir` and pass `celeba/val` to `--val_data_dir`.

# Trained text embeddings
In this [link]() we are uploading the text embeddings that we have trained, inlcluding the text embeddings we trained for the paper. You can download these text embeddings and use them for testing on your data using [this command](https://github.com/aliasgharkhani/one_shot_segmentation/tree/master#testing-with-the-trained-text-embeddings).
