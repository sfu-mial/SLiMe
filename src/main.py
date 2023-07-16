import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from src.co_part_segmentation_trainer import (
    CoSegmenterTrainer,
)
from src.config import Config
from src.datasets.pascal_voc_part_dataset import PascalVOCPartDataModule
from src.datasets.sample_dataset import SampleDataModule
from src.datasets.celeba_hq_dataset import CelebaHQDataModule
from src.datasets.paper_test_sample_dataset import PaperTestSampleDataModule
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
            mask_size=config.mask_size,
        )
    elif config.dataset == "pascal":
        dm = PascalVOCPartDataModule(
            train_data_file_ids_file=config.train_data_file_ids_file,
            val_data_file_ids_file=config.val_data_file_ids_file,
            object_name=config.object_name,
            part_names=config.part_names,
            batch_size=config.batch_size,
            train_data_ids=config.train_data_ids,
            val_data_ids=config.val_data_ids,
            mask_size=config.mask_size,
            blur_background=config.blur_background,
            fill_background_with_black=config.fill_background_with_black,
            remove_overlapping_objects=config.remove_overlapping_objects,
            object_overlapping_threshold=config.object_overlapping_threshold,
            final_min_crop_size=config.final_min_crop_size,
            single_object=config.single_object,
            adjust_bounding_box=config.adjust_bounding_box,
            zero_pad_test_output=config.zero_pad_test_output,
        )
    elif config.dataset == "celeba-hq":
        dm = CelebaHQDataModule(
            images_dir=config.images_dir,
            masks_dir=config.masks_dir,
            idx_mapping_file=config.idx_mapping_file,
            test_file_names_file_path=config.test_file_names_file_path,
            train_file_names_file_path=config.train_file_names_file_path,
            val_file_names_file_path=config.val_file_names_file_path,
            part_names=config.part_names,
            batch_size=config.batch_size,
            mask_size=config.mask_size,
            train_data_ids=config.train_data_ids,
            val_data_ids=config.val_data_ids,
        )
    elif config.dataset == "paper_test":
        dm = PaperTestSampleDataModule(
            test_images_dir=config.test_images_dir,
            test_masks_dir=config.test_masks_dir,
            mask_size=config.mask_size,
            zero_pad_test_output=config.zero_pad_test_output,
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
        # precision=16,
        # logger=wandb_logger,
        log_every_n_steps=1,
        # accumulate_grad_batches=config.train_num_crops // config.batch_size,
        enable_checkpointing=False,
        num_sanity_val_steps=0
    )
    if config.train:
        trainer.fit(model=model, datamodule=dm)
        trainer.test(model=model, datamodule=dm)
    else:
        trainer.test(model=model, datamodule=dm)


if __name__ == "__main__":
    main()
