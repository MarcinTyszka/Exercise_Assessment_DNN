import torch
import torch.nn as nn


class PoseBaseMLP(nn.Module):
    # Base Multi-Layer Perceptron without regularization for Phase 2 baseline evaluation
    def __init__(self, input_dim=138, hidden_dim=128, num_exercises=5):
        super().__init__()
        self.shared_backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        # Head 1: Multi-class exercise recognition (5 classes)
        self.exercise_head = nn.Linear(hidden_dim // 2, num_exercises)
        # Head 2: Binary form assessment (1 = Correct Form, 0 = Improper Form)
        self.form_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x):
        features = self.shared_backbone(x)
        exercise_logits = self.exercise_head(features)
        form_logit = self.form_head(features).squeeze(-1)
        return exercise_logits, form_logit


class PoseAssessmentLoss(nn.Module):
    # Combines CrossEntropyLoss for exercise classification and BCEWithLogitsLoss for form error detection
    def __init__(self):
        super().__init__()
        self.exercise_criterion = nn.CrossEntropyLoss()
        self.form_criterion = nn.BCEWithLogitsLoss()

    def forward(self, exercise_logits, form_logit, target_exercise, target_form):
        loss_ex = self.exercise_criterion(exercise_logits, target_exercise)
        loss_form = self.form_criterion(form_logit, target_form)
        total_loss = loss_ex + loss_form
        return total_loss, loss_ex, loss_form