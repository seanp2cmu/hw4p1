import torch
from typing import Dict, Any, Optional
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import numpy as np
import copy

def _epochs_to_steps(epochs: int, train_loader: torch.utils.data.DataLoader, gradient_accumulation_steps: int = 1) -> int:
    """Convert epochs to total steps based on the train loader length."""
    import math
    return epochs * math.ceil(len(train_loader) / gradient_accumulation_steps)

def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_config: Dict[str, Any],
    train_loader: torch.utils.data.DataLoader,
    gradient_accumulation_steps: int = 1,
    num_epochs: int = 100
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create learning rate scheduler based on config settings.
    All schedulers except ReduceLROnPlateau are configured to be step-based.
    """
    scheduler_name = scheduler_config['name'].lower()

    print("\n📈 Configuring Learning Rate Scheduler:")
    print(f"├── Type: {scheduler_name.upper()}")

    # Create base scheduler
    if scheduler_name == 'reduce_lr':
        reduce_config = scheduler_config['reduce_lr']
        patience_epochs = reduce_config.get('patience', 10)
        cooldown_epochs = reduce_config.get('cooldown', 0)
        
        print("├── ReduceLROnPlateau Settings:")
        print(f"│   ├── Mode: {reduce_config.get('mode', 'min')}")
        print(f"│   ├── Factor: {reduce_config.get('factor', 0.1)}")
        print(f"│   ├── Patience: {patience_epochs} epochs")
        print(f"│   ├── Threshold: {reduce_config.get('threshold', 0.0001)}")
        print(f"│   ├── Threshold Mode: {reduce_config.get('threshold_mode', 'rel')}")
        print(f"│   ├── Cooldown: {cooldown_epochs} epochs")
        print(f"│   └── Min LR: {reduce_config.get('min_lr', 0.00001)}")

        base_scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=reduce_config.get('mode', 'min'),
            factor=reduce_config.get('factor', 0.1),
            patience=patience_epochs,  # Keep as epochs
            threshold=reduce_config.get('threshold', 0.0001),
            threshold_mode=reduce_config.get('threshold_mode', 'rel'),
            cooldown=cooldown_epochs,  # Keep as epochs
            min_lr=reduce_config.get('min_lr', 0.00001),
            eps=reduce_config.get('eps', 1e-8)
        )
        return base_scheduler

    elif scheduler_name == 'cosine':
        cosine_config = scheduler_config['cosine']
        T_max_epochs = cosine_config.get('T_max', 60)
        T_max_steps = _epochs_to_steps(T_max_epochs, train_loader, gradient_accumulation_steps)
        
        print("├── Cosine Annealing Settings:")
        print(f"│   ├── T_max: {T_max_epochs} epochs ({T_max_steps} steps)")
        print(f"│   └── Min LR: {cosine_config.get('eta_min', 0.0001)}")

        base_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max_steps,  # Convert to steps
            eta_min=cosine_config.get('eta_min', 0.0001),
            last_epoch=cosine_config.get('last_epoch', -1)
        )

    elif scheduler_name == 'cosine_warm':
        warm_config = scheduler_config['cosine_warm']
        T_0_epochs = warm_config.get('T_0', 10)
        T_0_steps = _epochs_to_steps(T_0_epochs, train_loader, gradient_accumulation_steps)
        
        print("├── Cosine Annealing Warm Restarts Settings:")
        print(f"│   ├── T_0: {T_0_epochs} epochs ({T_0_steps} steps)")
        print(f"│   ├── T_mult: {warm_config.get('T_mult', 2)}")
        print(f"│   └── Min LR: {warm_config.get('eta_min', 0.0001)}")

        base_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=T_0_steps,  # Convert to steps
            T_mult=warm_config.get('T_mult', 2),
            eta_min=warm_config.get('eta_min', 0.0001),
            last_epoch=warm_config.get('last_epoch', -1)
        )

    elif scheduler_name == 'onecycle':
        onecycle_config = scheduler_config['onecycle']
        total_steps = _epochs_to_steps(num_epochs, train_loader, gradient_accumulation_steps)
        max_lr = [group['lr'] for group in optimizer.param_groups]

        print("├── OneCycleLR Settings:")
        print(f"│   ├── Total Epochs: {num_epochs} ({total_steps} steps)")
        print(f"│   ├── Max LR (first group): {max_lr[0]}")
        print(f"│   ├── Pct Start: {onecycle_config.get('pct_start', 0.1)}")
        print(f"│   ├── Div Factor: {onecycle_config.get('div_factor', 10)}")
        print(f"│   └── Final Div Factor: {onecycle_config.get('final_div_factor', 1000)}")

        return lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=onecycle_config.get('pct_start', 0.1),
            anneal_strategy=onecycle_config.get('anneal_strategy', 'cos'),
            div_factor=onecycle_config.get('div_factor', 10),
            final_div_factor=onecycle_config.get('final_div_factor', 1000),
            three_phase=onecycle_config.get('three_phase', False),
        )

    else:
        raise ValueError(
            f"Unsupported scheduler: {scheduler_name}. "
            f"Supported: ['reduce_lr', 'cosine', 'cosine_warm', 'onecycle']"
        )

    return base_scheduler


def plot_lr_schedule(
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    num_epochs: int,
    train_loader: torch.utils.data.DataLoader,
    gradient_accumulation_steps: int = 1,
    max_groups: int = 5  # Maximum number of groups to plot
) -> None:
    """
    Plot the learning rate schedule over epochs.
    
    Args:
        scheduler: The learning rate scheduler
        num_epochs: Total number of epochs to plot
        train_loader: Training data loader to determine steps per epoch
        gradient_accumulation_steps: Number of gradient accumulation steps
        max_groups: Maximum number of parameter groups to plot
    """
    # Save initial states
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler_state = copy.deepcopy(scheduler.__dict__)
    else:
        scheduler_state = copy.deepcopy(scheduler.state_dict())
    
    optimizer_state = copy.deepcopy(scheduler.optimizer.state_dict())
    
    # Store initial learning rates
    initial_lr = [group['lr'] for group in scheduler.optimizer.param_groups]
    num_groups = len(initial_lr)
    
    # If there are too many groups, only plot a subset
    groups_to_plot = min(num_groups, max_groups)
    if num_groups > max_groups:
        print(f"Warning: Only showing {max_groups} out of {num_groups} parameter groups for clarity")
    
    lrs = [[] for _ in range(groups_to_plot)]
    
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        # For ReduceLROnPlateau, simulate epoch-wise updates
        for epoch in range(num_epochs):
            # Record current learning rates
            for idx, group in enumerate(scheduler.optimizer.param_groups[:groups_to_plot]):
                lrs[idx].extend([group['lr']] * len(train_loader))
            
            # Step the scheduler with a dummy metric that triggers LR reduction
            # every patience+1 epochs to show the behavior
            scheduler.step(1.0 if epoch % (scheduler.patience + 1) != 0 else 0.0)
        
        x = np.linspace(0, num_epochs, num_epochs * len(train_loader))
    else:
        # For step-based schedulers
        total_steps = _epochs_to_steps(num_epochs, train_loader, gradient_accumulation_steps)
        
        # Simulate training loop
        for step in range(total_steps):
            # Record current learning rates
            for idx, group in enumerate(scheduler.optimizer.param_groups[:groups_to_plot]):
                lrs[idx].append(group['lr'])
            
            # Step the scheduler
            scheduler.optimizer.step()
            scheduler.step()
        
        x = np.linspace(0, num_epochs, total_steps)
    
    # Restore initial states
    scheduler.optimizer.load_state_dict(optimizer_state)
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.__dict__.update(scheduler_state)
    else:
        scheduler.load_state_dict(scheduler_state)
    
    # Plot the learning rates with better styling
    plt.figure(figsize=(12, 4))
    
    # Define a set of distinct colors and line styles
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    line_styles = ['-', '--', '-.', ':', '-']
    
    for idx, lr_list in enumerate(lrs[:groups_to_plot]):
        color = colors[idx % len(colors)]
        style = line_styles[idx % len(line_styles)]
        plt.plot(x, lr_list, 
                label=f'Group {idx}', 
                color=color, 
                linestyle=style,
                linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title('Learning Rate Schedule', fontsize=14, pad=20)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.yscale('log')
    
    # Create a second x-axis for steps
    ax2 = plt.gca().twiny()
    ax2.set_xlim(0, len(x))
    ax2.set_xlabel('Steps', fontsize=12)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.show()