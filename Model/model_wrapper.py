from Pipeline.dataset import BasePytorchModelDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
import wandb
from contextlib import nullcontext

from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR


class SklearnModelWrapper:
    def __init__(self, model):
        self.model = model

    def fit(self, train_X, train_y, test_X, test_y):
        self.model.fit(train_X, train_y)
        return {}

    def predict(self, X):
        predict = self.model.predict(X)
        return predict

    def reset(self,):
        print('Reset The model')
        if isinstance(self.model, MultiOutputRegressor):
            self.model = self.model.__class__(SVR(kernel="rbf"))
        else:
            self.model = self.model.__class__()


class PytorchModelWrapper:
    def __init__(self, model, train_config):
        self.model = model
        self.train_config = train_config
        self.logging = train_config['log_experiments']
        if self.logging :
            wandb.init(project="circuit_training", config=train_config)
    
    def reset(self,):
        print('Reset The model')
        for layers in self.model.children():
            if isinstance(layers, nn.Sequential):
                for layer in layers:
                    if hasattr(layer, 'reset_parameters'):
                        layer.reset_parameters()
            else:
                if hasattr(layers, 'reset_parameters'):
                    layers.reset_parameters()

    def fit(self, train_X, train_y, test_X, test_y):
        train_dataset = BasePytorchModelDataset(train_X, train_y)
        test_dataset = BasePytorchModelDataset(test_X, test_y)
        batch_size = self.train_config.get("batch_size", 100)
        num_workers = self.train_config.get("num_workers", 0)
        pin_memory = self.train_config.get("pin_memory", False)

        train_loader_kwargs = {
            "dataset": train_dataset,
            "batch_size": batch_size,
            "shuffle": True,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        }
        test_loader_kwargs = {
            "dataset": test_dataset,
            "batch_size": batch_size,
            "shuffle": False,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        }

        if num_workers > 0:
            persistent_workers = self.train_config.get("persistent_workers", True)
            prefetch_factor = self.train_config.get("prefetch_factor", 2)
            train_loader_kwargs["persistent_workers"] = persistent_workers
            test_loader_kwargs["persistent_workers"] = persistent_workers
            train_loader_kwargs["prefetch_factor"] = prefetch_factor
            test_loader_kwargs["prefetch_factor"] = prefetch_factor

        train_dataloader = DataLoader(**train_loader_kwargs)
        test_dataloader = DataLoader(**test_loader_kwargs)
        train_result = self.model_train(train_dataloader, test_dataloader)
        return train_result

    def predict(self, X):
        self.model.eval()
        return self.model(torch.Tensor(X).to(self.train_config["device"])).to('cpu').detach().numpy()

    def model_train(self, train_dataloader, test_dataloader):
        device = torch.device(self.train_config["device"])
        self.model.to(device)

        use_smooth_l1 = self.train_config.get("use_smooth_l1", False)
        if use_smooth_l1:
            train_loss = nn.SmoothL1Loss(beta=self.train_config.get("smooth_l1_beta", 0.5))
        else:
            train_loss = nn.L1Loss()

        lr = self.train_config.get("learning_rate", 1e-3)
        weight_decay = self.train_config.get("weight_decay", 0.0)
        optimizer_name = self.train_config.get("optimizer", "adamw").lower()
        betas = tuple(self.train_config.get("adam_betas", (0.9, 0.999)))
        eps = self.train_config.get("adam_eps", 1e-8)

        use_param_groups = optimizer_name == "adamw" and self.train_config.get("adamw_param_groups", True)
        if use_param_groups:
            decay_params, no_decay_params = [], []
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                if param.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower():
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
            parameter_groups = [
                {"params": decay_params, "weight_decay": weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ]
        else:
            parameter_groups = self.model.parameters()

        if optimizer_name == "adam":
            optimizer = optim.Adam(parameter_groups, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
        else:
            optimizer = optim.AdamW(parameter_groups, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)

        # Optional: ReduceLROnPlateau scheduler
        use_scheduler = self.train_config.get("use_scheduler", False)
        scheduler_type = self.train_config.get("scheduler", "plateau").lower()
        onecycle_scheduler = None
        scheduler = (
            optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.train_config.get("scheduler_factor", 0.5),
                patience=self.train_config.get("scheduler_patience", 8),
            )
            if use_scheduler else None
        )

        if use_scheduler and scheduler_type == "onecycle":
            onecycle_scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.train_config.get("max_learning_rate", lr * 4),
                epochs=self.train_config["epochs"],
                steps_per_epoch=max(1, len(train_dataloader)),
                pct_start=self.train_config.get("onecycle_pct_start", 0.15),
                div_factor=self.train_config.get("onecycle_div_factor", 10.0),
                final_div_factor=self.train_config.get("onecycle_final_div_factor", 100.0),
            )
            scheduler = None

        # Optional: early stopping
        use_early_stop = self.train_config.get("use_early_stop", False)
        patience = self.train_config.get("early_stop_patience", 20)
        min_delta = self.train_config.get("early_stop_min_delta", 1e-6)
        best_val_loss = float("inf")
        no_improve = 0
        best_model_state = None

        # Optional: gradient clipping
        clip_grad = self.train_config.get("clip_grad", False)
        clip_value = self.train_config.get("clip_grad_norm", 1.0)
        pin_memory = self.train_config.get("pin_memory", False)

        use_amp = self.train_config.get("use_amp", True) and device.type == "cuda"
        scaler = torch.amp.GradScaler(enabled=use_amp)
        autocast_ctx = torch.cuda.amp.autocast if use_amp else nullcontext

        losses = []
        val_losses = []

        for epoch in range(self.train_config["epochs"]):
            self.model.train()
            avg_loss = 0
            val_avg_loss = 0
            for t, (x, y) in enumerate(train_dataloader):
                optimizer.zero_grad(set_to_none=True)
                x_var = x.to(device=device, dtype=torch.float32, non_blocking=pin_memory)
                y_var = y.to(device=device, dtype=torch.float32, non_blocking=pin_memory)

                with autocast_ctx():
                    scores = self.model(x_var)
                    loss = train_loss(scores.float(), y_var.float())

                avg_loss += (loss.item() - avg_loss) / (t + 1)
                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if clip_grad:
                    if use_amp:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)

                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                if onecycle_scheduler is not None:
                    onecycle_scheduler.step()

            with torch.no_grad():
                self.model.eval()
                for t, (x, y) in enumerate(test_dataloader):
                    x_var = x.to(device=device, dtype=torch.float32, non_blocking=pin_memory)
                    y_var = y.to(device=device, dtype=torch.float32, non_blocking=pin_memory)
                    with autocast_ctx():
                        scores = self.model(x_var)
                        loss = train_loss(scores.float(), y_var.float())

                    val_avg_loss += (loss.item() - val_avg_loss) / (t + 1)

            losses.append(avg_loss)
            val_losses.append(val_avg_loss)

            if scheduler is not None:
                scheduler.step(val_avg_loss)

            if self.train_config["loss_per_epoch"]:
                current_lr = optimizer.param_groups[0]["lr"]
                print(f'epoch: {"{:<4}".format(epoch)} train loss: {"{:1.4f}".format(avg_loss, 4)}, validation loss: {"{:1.4f}".format(val_avg_loss, 4)}, lr: {current_lr:.6g}')
            else:
                print(f'epoch: {"{:<4}".format(epoch)} ')

            if self.logging:
                wandb.log({'train_loss': avg_loss, 'val_loss': val_avg_loss, 'epoch': epoch, 'lr': optimizer.param_groups[0]['lr']})

            # Early stopping
            # if use_early_stop:
            #     if val_avg_loss + min_delta < best_val_loss:
            #         best_val_loss = val_avg_loss
            #         no_improve = 0
            #         best_model_state = {
            #             k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
            #         }
            #     else:
            #         no_improve += 1
            #         if no_improve >= patience:
            #             print(f'Early stopping at epoch {epoch} (no improvement for {patience} epochs)')
            #             break

        if use_early_stop and best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            self.model.to(device)

        result_dict = dict()

        result_dict["train_loss"] = losses
        result_dict["validation_loss"] = val_losses

        return result_dict