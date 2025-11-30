import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalTypingLoss(nn.Module):
    """
    A hierarchical loss function for training a sequence recommendation model.
    
    Inspiration:
    - PinnerFormer (Pinterest): Using long-term user interaction sequences.
    - DIEN (Alibaba): Deep Interest Evolution Network for capturing evolving user interests.
    
    Hierarchy:
    1. Speed (Raw WPM): The foundational metric. Can the user type fast?
    2. Accuracy: Given the speed, are they accurate?
    3. Flow (Speed * Accuracy): The combined objective.
    4. Cross-Network: Predicting the next best snippet embedding.
    
    This loss assumes the model outputs multiple heads:
    - speed_pred: Predicted WPM
    - acc_pred: Predicted Accuracy
    - next_embedding: Predicted embedding of the optimal next snippet
    """
    def __init__(self, w_speed=0.2, w_acc=0.3, w_flow=0.3, w_contrastive=0.2):
        super().__init__()
        self.w_speed = w_speed
        self.w_acc = w_acc
        self.w_flow = w_flow
        self.w_contrastive = w_contrastive
        
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss() # Or MSE for continuous accuracy
    
    def forward(self, predictions, targets):
        """
        predictions: dict containing:
            - pred_wpm: (batch, 1)
            - pred_acc: (batch, 1)
            - pred_next_embedding: (batch, dim)
            
        targets: dict containing:
            - true_wpm: (batch, 1)
            - true_acc: (batch, 1)
            - positive_snippet_embedding: (batch, dim)
            - negative_snippet_embeddings: (batch, k, dim) [Optional for contrastive]
        """
        
        # 1. Speed Loss (Regression)
        loss_speed = self.mse(predictions['pred_wpm'], targets['true_wpm'])
        
        # 2. Accuracy Loss (Regression)
        loss_acc = self.mse(predictions['pred_acc'], targets['true_acc'])
        
        # 3. Flow Loss (Combined Metric)
        # Flow = WPM * Accuracy. We want the model to learn this interaction explicitly.
        true_flow = targets['true_wpm'] * targets['true_acc']
        pred_flow = predictions['pred_wpm'] * predictions['pred_acc']
        loss_flow = self.mse(pred_flow, true_flow)
        
        # 4. Cross-Network / Contrastive Loss (Next Item Prediction)
        # Predicting the embedding of the snippet that maximizes the user's improvement (or engagement)
        # Here we use a simple InfoNCE-style loss or Triplet Loss
        
        anchor = predictions['pred_next_embedding']
        positive = targets['positive_snippet_embedding']
        
        # Cosine Embedding Loss (Maximize similarity between predicted and actual 'good' next snippet)
        # target=1 means they should be similar
        loss_contrastive = F.cosine_embedding_loss(anchor, positive, torch.ones(anchor.size(0)).to(anchor.device))
        
        total_loss = (
            self.w_speed * loss_speed +
            self.w_acc * loss_acc +
            self.w_flow * loss_flow +
            self.w_contrastive * loss_contrastive
        )
        
        return total_loss, {
            "loss_speed": loss_speed.item(),
            "loss_acc": loss_acc.item(),
            "loss_flow": loss_flow.item(),
            "loss_contrastive": loss_contrastive.item()
        }

class FlowTypeModel(nn.Module):
    """
    Conceptual architecture for the future FlowType model.
    """
    def __init__(self, input_dim, embedding_dim):
        super().__init__()
        # Evolution Layer (GRU/LSTM) to track user state over time (DIEN style)
        self.interest_evolution = nn.GRU(input_dim, embedding_dim, batch_first=True)
        
        # Multi-Task Heads
        self.speed_head = nn.Sequential(nn.Linear(embedding_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.acc_head = nn.Sequential(nn.Linear(embedding_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
        self.embedding_head = nn.Sequential(nn.Linear(embedding_dim, embedding_dim)) # Projects to snippet space
        
    def forward(self, user_interaction_sequence):
        # user_interaction_sequence: (batch, seq_len, input_dim)
        
        # Process sequence
        output, hidden = self.interest_evolution(user_interaction_sequence)
        
        # Use final state for predictions
        final_state = output[:, -1, :]
        
        pred_wpm = self.speed_head(final_state)
        pred_acc = self.acc_head(final_state)
        pred_next_embedding = self.embedding_head(final_state)
        
        return {
            "pred_wpm": pred_wpm,
            "pred_acc": pred_acc,
            "pred_next_embedding": pred_next_embedding
        }
