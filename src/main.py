import pytorch_lightning as pl

from src.config import Config
from src.datasets.sample_dataset import SampleDataModule
from src.datasets.pascal_voc_part_dataset import PascalVOCPartDataModule
from src.co_part_segmentation_trainer import (
    CoSegmenterTrainer,
)


def main():
    # torch.manual_seed(42)
    config = Config()
    if config.dataset == "sample":
        dm = SampleDataModule(
            src_image_dir=config.src_image_path,
            target_image_dir=config.target_image_path,
            src_segmentation_dir=config.src_segmentation_path,
            location1=config.point_location1,
            location2=config.point_location2,
            train_num_crops=config.train_num_crops,
            test_num_crops=config.test_num_crops,
            batch_size=config.batch_size,
            train_crop_ratio=config.train_crop_ratio,
            test_crop_ratio=config.test_crop_ratio,
            mask_size=config.mask_size,
        )
    elif config.dataset == "pascal":
        dm = PascalVOCPartDataModule(
            annotations_files_dir=config.annotations_files_dir,
            object_name=config.object_name,
            part_name=config.part_name,
            train_num_crops=config.train_num_crops,
            test_num_crops=config.test_num_crops,
            batch_size=config.batch_size,
            train_crop_ratio=config.train_crop_ratio,
            test_crop_ratio=config.test_crop_ratio,
            train_data_id=config.train_data_id,
            mask_size=config.mask_size,
        )
    model = CoSegmenterTrainer(config=config)
    trainer = pl.Trainer(
        accelerator="gpu",
        # strategy="dp",
        default_root_dir=config.base_dir,
        max_epochs=config.epochs,
        devices=config.gpu_id,
        # precision=16,
        log_every_n_steps=1,
        accumulate_grad_batches=config.train_num_crops//config.batch_size,
        enable_checkpointing=False
    )
    if config.train:
        trainer.fit(model=model, datamodule=dm)
        trainer.test(model=model, datamodule=dm)
    else:
        trainer.test(model=model, datamodule=dm)


if __name__ == "__main__":
    main()
