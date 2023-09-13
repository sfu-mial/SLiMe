import pytorch_lightning as pl

from src.co_part_segmentation_trainer import (
    CoSegmenterTrainer,
)
from src.datasets.pascal_voc_part_dataset import PascalVOCPartDataModule
from src.datasets.sample_dataset import SampleDataModule
from src.datasets.celeba_hq_dataset import CelebaHQDataModule
from src.datasets.paper_test_sample_dataset import PaperTestSampleDataModule
from src.datasets.ade20k_dataset import ADE20KDataModule
from src.datasets.cat15_dataset import CAT15DataModule
from src.arguments import init_args


def main():
    # torch.manual_seed(42)
    config = init_args()
    # config = Config()
    # wandb_logger = WandbLogger()

    if config.dataset == "sample":
        dm = SampleDataModule(
            src_image_dirs=config.src_image_paths,
            target_image_dir=config.target_image_path,
            src_mask_dirs=config.src_mask_paths,
            batch_size=config.batch_size,
            train_mask_size=config.train_mask_size,
            test_mask_size=config.test_mask_size,
            num_parts=len(config.parts_to_return) - 1,
            min_crop_ratio=config.min_crop_ratio,
        )
    elif config.dataset == "pascal":
        dm = PascalVOCPartDataModule(
            ann_file_base_dir=config.ann_file_base_dir,
            images_base_dir=config.images_base_dir,
            car_test_data_dir=config.car_test_data_dir,
            train_data_file_ids_file=config.train_data_file_ids_file,
            val_data_file_ids_file=config.val_data_file_ids_file,
            object_name=config.object_name,
            parts_to_return=config.parts_to_return,
            batch_size=config.batch_size,
            train_data_ids=config.train_data_ids,
            val_data_ids=config.val_data_ids,
            train_mask_size=config.train_mask_size,
            test_mask_size=config.test_mask_size,
            blur_background=config.blur_background,
            fill_background_with_black=config.fill_background_with_black,
            remove_overlapping_objects=config.remove_overlapping_objects,
            object_overlapping_threshold=config.object_overlapping_threshold,
            min_crop_size=config.min_crop_size,
            single_object=config.single_object,
            adjust_bounding_box=config.adjust_bounding_box,
            keep_aspect_ratio=config.keep_aspect_ratio,
            min_crop_ratio=config.min_crop_ratio,
        )
    elif config.dataset == "celeba-hq":
        dm = CelebaHQDataModule(
            images_dir=config.images_dir,
            masks_dir=config.masks_dir,
            idx_mapping_file=config.idx_mapping_file,
            test_file_names_file_path=config.test_file_names_file_path,
            train_file_names_file_path=config.train_file_names_file_path,
            val_file_names_file_path=config.val_file_names_file_path,
            parts_to_return=config.parts_to_return,
            batch_size=config.batch_size,
            train_mask_size=config.train_mask_size,
            test_mask_size=config.test_mask_size,
            train_data_ids=config.train_data_ids,
            val_data_ids=config.val_data_ids,
            min_crop_ratio=config.min_crop_ratio,
            version=config.human_version,
        )
    elif config.dataset == "paper_test":
        dm = PaperTestSampleDataModule(
            test_images_dir=config.test_images_dir,
            test_masks_dir=config.test_masks_dir,
            train_mask_size=config.train_mask_size,
            test_mask_size=config.test_mask_size,
        )
    elif config.dataset == "ade20k":
        dm = ADE20KDataModule(
            train_data_dir=config.train_data_dir,
            test_data_dir=config.test_data_dir,
            object_names=config.object_name,
            mask_size=config.mask_size,
            min_crop_ratio=config.min_crop_ratio,
        )
    elif config.dataset == "cat15":
        dm = CAT15DataModule(
            train_data_dir=config.train_data_dir,
            test_data_dir=config.test_data_dir,
            mask_size=config.mask_size,
            min_crop_ratio=config.min_crop_ratio,
        )
    model = CoSegmenterTrainer(config=config)
    if isinstance(config.gpu_id, int):
        gpu_id = [config.gpu_id]
    else:
        gpu_id = config.gpu_id
    trainer = pl.Trainer(
        accelerator="gpu",
        # strategy="dp",
        default_root_dir=config.base_dir,
        max_epochs=config.epochs,
        devices=gpu_id,
        accumulate_grad_batches=config.accumulate_grad_batches,
        # precision=16,
        # logger=wandb_logger,
        log_every_n_steps=1,
        # accumulate_grad_batches=config.train_num_crops // config.batch_size,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
    )
    if config.train:
        trainer.fit(model=model, datamodule=dm)
        if not config.dataset == "sample":
            trainer.test(model=model, datamodule=dm)
    else:
        trainer.test(model=model, datamodule=dm)


if __name__ == "__main__":
    main()
