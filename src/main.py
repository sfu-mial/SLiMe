import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from src.config import Config
from src.dataset import SampleDataModule
from src.co_part_segmentation_trainer import (
    CoSegmenterTrainer,
)


def main():
    # torch.manual_seed(42)
    config = Config()
    if os.path.exists(os.path.join(config.base_dir, "lightning_logs/")):
        runs = os.listdir(os.path.join(config.base_dir, "lightning_logs/"))
        current_version = len(runs)
    else:
        current_version = 0

    dm = SampleDataModule(
        src_image_dir=config.src_image_path,
        target_image_dir=config.target_image_path,
        location=config.point_location,
        num_crops=config.num_crops,
        batch_size=config.batch_size,
        crop_ratio=config.crop_ratio
    )
    model = CoSegmenterTrainer(config=config)
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(
            config.base_dir, f"checkpoints/version_{current_version}"
        ),
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_monitor]
    trainer = pl.Trainer(
        accelerator="gpu",
        # strategy="dp",
        callbacks=callbacks,
        default_root_dir=config.base_dir,
        max_epochs=config.epochs,
        devices=config.gpu_id,
        # precision=16,
        log_every_n_steps=1,
        accumulate_grad_batches=config.num_crops//config.batch_size
    )
    if config.train:
        trainer.fit(model=model, datamodule=dm)
        trainer.test(model=model, datamodule=dm)
    else:
        trainer.test(model=model, datamodule=dm)


if __name__ == "__main__":
    main()
