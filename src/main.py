import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor

from src.config import Config
from src.dataset import SampleDataModule
from src.co_part_segmentation_trainer import (
    CoSegmenterTrainer,
)


def main():
    # torch.manual_seed(42)
    config = Config()

    dm = SampleDataModule(
        src_image_dir=config.src_image_path,
        target_image_dir=config.target_image_path,
        src_segmentation_dir=config.src_segmentation_path,
        location1=config.point_location1,
        location2=config.point_location2,
        num_crops=config.num_crops,
        batch_size=config.batch_size,
        crop_ratio=config.crop_ratio
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
        accumulate_grad_batches=config.num_crops//config.batch_size,
        enable_checkpointing=False
    )
    if config.train:
        trainer.fit(model=model, datamodule=dm)
        trainer.test(model=model, datamodule=dm)
    else:
        trainer.test(model=model, datamodule=dm)


if __name__ == "__main__":
    main()
