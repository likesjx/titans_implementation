import torch
import torch.nn as nn
import torch.nn.functional as F


class LongTermMemoryMLP(nn.Module):
    """
    Long-term memory implemented as a multi-layer perceptron (MLP).
    
    This is a key component of the Titans architecture that:
    1. Learns to memorize and forget information at test time
    2. Uses a "surprise" metric to determine what to memorize
    3. Incorporates momentum and forgetting mechanisms
    """
    
    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        learning_rate=0.01,
        momentum_factor=0.9,
        forget_factor=0.1,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.momentum_factor = momentum_factor
        self.forget_factor = forget_factor
        
        # Build the MLP layers
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
        
        # Initialize gradients and velocity for momentum
        self.grads = None
        self.velocity = None
        
    def initialize(self, batch_size, device):
        """
        Initialize the memory state for a new batch.
        
        Args:
            batch_size: Number of sequences in the batch
            device: Device to initialize tensors on
            
        Returns:
            Memory state dictionary
        """
        # Initialize model parameters for each example in the batch
        params = []
        for i, param in enumerate(self.model.parameters()):
            # Clone and expand parameters for each example in batch
            batch_param = param.unsqueeze(0).expand(batch_size, *param.shape).clone()
            batch_param.requires_grad_(True)
            params.append(batch_param)
        
        # Initialize velocity tensors for momentum-based updates
        velocity = []
        for param in params:
            velocity.append(torch.zeros_like(param))
        
        memory_state = {
            "params": params,
            "velocity": velocity
        }
        
        return memory_state
    
    def forward(self, memory_state, query):
        """
        Retrieve values from memory using the provided query.
        
        Args:
            memory_state: Current memory state dictionary
            query: Query tensor [batch_size, seq_len, input_dim]
            
        Returns:
            Retrieved values [batch_size, seq_len, output_dim]
        """
        batch_size, seq_len, _ = query.shape
        params = memory_state["params"]
        
        # For each sequence position, apply the MLP
        outputs = []
        
        for i in range(seq_len):
            q = query[:, i, :]  # [batch_size, input_dim]
            
            # For each example in the batch, use its own memory parameters
            batch_outputs = []
            for b in range(batch_size):
                x = q[b]  # [input_dim]
                
                # Manual forward pass through the MLP using batch-specific parameters
                for j in range(0, len(params), 2):
                    # Apply linear layer
                    if j + 1 < len(params):
                        # Linear + ReLU
                        w, b = params[j][b], params[j+1][b]
                        x = F.linear(x, w, b)
                        x = F.relu(x)
                    else:
                        # Final linear layer
                        w, b = params[j][b], params[j+1][b]
                        x = F.linear(x, w, b)
                
                batch_outputs.append(x)
            
            # Stack batch outputs
            batch_output = torch.stack(batch_outputs)  # [batch_size, output_dim]
            outputs.append(batch_output)
        
        # Stack outputs for all sequence positions
        outputs = torch.stack(outputs, dim=1)  # [batch_size, seq_len, output_dim]
        
        return outputs
    
    def update(self, memory_state, keys, values, values_pred=None):
        """
        Update memory based on the surprise between predicted and actual values.
        
        Args:
            memory_state: Current memory state dictionary
            keys: Input keys [batch_size, seq_len, input_dim]
            values: Actual values [batch_size, seq_len, output_dim]
            values_pred: Predicted values (if None, will be computed)
            
        Returns:
            Updated memory state dictionary
        """
        batch_size, seq_len, _ = keys.shape
        params = memory_state["params"]
        velocity = memory_state["velocity"]
        
        # If predictions not provided, compute them
        if values_pred is None:
            values_pred = self.forward(memory_state, keys)
        
        # Compute loss (surprise metric)
        loss = F.mse_loss(values_pred, values, reduction='none')
        loss = loss.mean(dim=-1)  # [batch_size, seq_len]
        
        # Compute gradients for each parameter
        grads = []
        for param in params:
            grad = torch.zeros_like(param)
            grads.append(grad)
        
        # Process each sequence position
        for i in range(seq_len):
            # For each example in batch
            for b in range(batch_size):
                # Skip updates with very small surprise
                if loss[b, i].item() < 1e-6:
                    continue
                
                # Compute gradients using autograd
                q = keys[b, i].unsqueeze(0)  # [1, input_dim]
                target = values[b, i].unsqueeze(0)  # [1, output_dim]
                
                # Forward pass with current parameters
                x = q
                activations = []
                
                for j in range(0, len(params), 2):
                    # Store input for backward pass
                    activations.append(x)
                    
                    # Apply linear layer
                    if j + 1 < len(params):
                        # Linear + ReLU
                        w, b = params[j][b], params[j+1][b]
                        x = F.linear(x, w, b)
                        x = F.relu(x)
                    else:
                        # Final linear layer
                        w, b = params[j][b], params[j+1][b]
                        x = F.linear(x, w, b)
                
                # Compute loss and do backward pass manually
                pred = x
                mse = F.mse_loss(pred, target)
                
                # Manual backward pass for output layer
                d_loss = 2 * (pred - target)  # [1, output_dim]
                
                # Backpropagate through layers
                for j in range(len(params)-2, -1, -2):
                    # Gradient for bias
                    grads[j+1][b] += d_loss.squeeze(0) * self.learning_rate
                    
                    # Gradient for weights
                    x = activations[j//2]
                    grads[j][b] += torch.matmul(d_loss.t(), x) * self.learning_rate
                    
                    # Backpropagate through ReLU and linear layer
                    if j > 0:
                        w = params[j][b]
                        d_loss = torch.matmul(d_loss, w)
                        
                        # ReLU gradient
                        prev_output = F.linear(activations[j//2-1], params[j-2][b], params[j-1][b])
                        d_loss = d_loss * (prev_output > 0).float()
        
        # Apply gradients with momentum and forgetting
        for i, (param, grad, vel) in enumerate(zip(params, grads, velocity)):
            # Update velocity with momentum
            vel.mul_(self.momentum_factor).add_(grad)
            
            # Incorporate forgetting mechanism
            forget_mask = torch.rand_like(param) > self.forget_factor
            forget_mask = forget_mask.float()
            
            # Update parameters
            param.add_(-vel * forget_mask)
        
        # Update memory state
        memory_state = {
            "params": params,
            "velocity": velocity
        }
        
        return memory_state


class ParallelLongTermMemory(nn.Module):
    """
    Parallel implementation of long-term memory for efficient batch processing.
    
    This implements the optimized version described in the Titans paper
    for parallel training of the memory module.
    """
    
    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        learning_rate=0.01,
        momentum_factor=0.9,
        forget_factor=0.1,
        batch_size=8,
    ):
        super().__init__()
        
        self.memory_mlp = LongTermMemoryMLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            learning_rate=learning_rate,
            momentum_factor=momentum_factor,
            forget_factor=forget_factor,
        )
        
        self.batch_size = batch_size
        
    def process_batch(self, keys, values, memory_state=None):
        """
        Process a batch of sequences in parallel.
        
        Args:
            keys: Input keys [batch_size, seq_len, input_dim]
            values: Actual values [batch_size, seq_len, output_dim]
            memory_state: Initial memory state (if None, will be initialized)
            
        Returns:
            outputs: Model outputs
            updated_memory: Updated memory state
        """
        batch_size, seq_len, _ = keys.shape
        device = keys.device
        
        # Initialize memory if not provided
        if memory_state is None:
            memory_state = self.memory_mlp.initialize(batch_size, device)
        
        # Process the batch
        retrieved_values = self.memory_mlp.forward(memory_state, keys)
        
        # Update memory
        updated_memory = self.memory_mlp.update(memory_state, keys, values, retrieved_values)
        
        return retrieved_values, updated_memory 