import lightning as L
import torch


class CFGLightningModule(L.LightningModule):
    """
    PyTorch Lightning module for ChemFM Classifier-Free Guidance training.
    """

    def __init__(
        self,
        model,
        optimizer_partial,
        scheduler_partials=None,
    ):
        super().__init__()

        self.model = model
        self.optimizer_partial = optimizer_partial
        self.scheduler_partials = scheduler_partials or []

    def forward(self, **batch):
        return self.model(**batch)

    def training_step(self, batch, batch_idx):

        outputs = self.model(**batch)

        loss = outputs.loss

        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            sync_dist=True,
            on_step=True,
            on_epoch=True,
            batch_size=batch["input_ids"].size(0),
        )

        return loss

    def validation_step(self, batch, batch_idx):

        outputs = self.model(**batch)

        loss = outputs.loss

        self.log(
            "val/loss",
            loss,
            prog_bar=True,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch["input_ids"].size(0),
        )

        return loss

    def configure_optimizers(self):

        optimizer = self.optimizer_partial(
            self.parameters()
        )

        # No scheduler
        if len(self.scheduler_partials) == 0:
            return optimizer

        # Build first scheduler
        scheduler = self.scheduler_partials[0](optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }