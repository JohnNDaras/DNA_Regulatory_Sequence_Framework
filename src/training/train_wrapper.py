import torch
import torch.nn as nn

def train_model_boosted_iter(model, train_loader, val_loader, epochs=20, steps_per_epoch=300,
                             patience=6, base_lr=3e-3, verbose=True, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    torch.backends.cudnn.benchmark = True

    approx_batches = 50
    pos, tot = 0, 0
    for i, (_, yb) in enumerate(train_loader):
        if i >= approx_batches:
            break
        pos += (yb > 0.5).sum().item()
        tot += yb.numel()
    pos = max(pos, 1)
    neg = max(tot - pos, 1)
    pos_weight = torch.tensor([neg / pos], device=device, dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=base_lr, epochs=epochs, steps_per_epoch=steps_per_epoch,
        pct_start=0.1, anneal_strategy="cos", div_factor=10.0, final_div_factor=10.0
    )

    ema_decay = 0.996
    with torch.no_grad():
        ema = {k: v.detach().clone() for k, v in model.state_dict().items()
               if torch.is_tensor(v) and torch.is_floating_point(v)}

    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    train_accs, val_accs = [], []
    best_val_acc, best_state = 0.0, None
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        correct, total, step = 0, 0, 0
        it = iter(train_loader)

        while step < steps_per_epoch:
            try:
                xb, yb = next(it)
            except StopIteration:
                it = iter(train_loader)
                continue

            step += 1
            xb = xb.to(device, non_blocking=True).float()
            yb = yb.to(device, non_blocking=True).float()
            yb_smooth = yb.mul(1 - 0.05) + 0.5 * 0.05

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(xb).squeeze()
                loss = loss_fn(logits, yb_smooth)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()

            preds = torch.sigmoid(logits) > 0.5
            correct += (preds == yb.bool()).sum().item()
            total += xb.size(0)

            with torch.no_grad():
                ms = model.state_dict()
                for k in ema.keys():
                    ema[k].mul_(ema_decay).add_(ms[k] * (1 - ema_decay))

        train_acc = correct / max(total, 1)
        train_accs.append(train_acc)

        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        ema_full = backup.copy()
        ema_full.update(ema)
        model.load_state_dict(ema_full, strict=False)

        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True).float()
                yb = yb.to(device, non_blocking=True).float()
                logits = model(xb).squeeze()
                preds = torch.sigmoid(logits) > 0.5
                v_correct += (preds == yb.bool()).sum().item()
                v_total += xb.size(0)

        val_acc = v_correct / max(v_total, 1)
        val_accs.append(val_acc)
        model.load_state_dict(backup, strict=False)

        if verbose:
            print(f"Epoch {epoch:02d}/{epochs} | steps {steps_per_epoch} | "
                  f"train {train_acc:.4f} | val {val_acc:.4f} | lr {sched.get_last_lr()[0]:.2e}")

        if val_acc > best_val_acc + 1e-4:
            best_val_acc = val_acc
            best_state = {k: v.detach().clone() for k, v in ema.items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                if verbose:
                    print(f"Early stopping. Best val acc={best_val_acc:.4f}")
                break

    if best_state is not None:
        final_state = model.state_dict()
        final_state.update(best_state)
        model.load_state_dict(final_state, strict=False)

    return model, train_accs, val_accs
