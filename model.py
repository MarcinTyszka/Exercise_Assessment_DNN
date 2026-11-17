import torch
import torch.nn as nn
from feature_extraction import NUM_FEATURES, NUM_BIOMECH_FEATURES


class PoseBaseMLP(nn.Module):
    # Multi-task Base MLP without regularization (Phase 2 specification)
    def __init__(self, input_dim=NUM_FEATURES, hidden_dim=128, num_exercises=5):
        super().__init__()
        self.num_biomech = NUM_BIOMECH_FEATURES

        # Shared pose representation backbone
        self.shared_backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

        # Head 1: Exercise classification (5 classes)
        self.exercise_head = nn.Linear(hidden_dim // 2, num_exercises)

        # Head 2: Dedicated biomechanical form assessment branch
        # Concatenates shared pose context with the 12 explicit biomechanical features
        self.form_branch = nn.Sequential(
            nn.Linear((hidden_dim // 2) + self.num_biomech, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        shared_features = self.shared_backbone(x)
        exercise_logits = self.exercise_head(shared_features)

        biomech_features = x[:, -self.num_biomech:]
        form_input = torch.cat([shared_features, biomech_features], dim=1)
        form_logit = self.form_branch(form_input).squeeze(-1)

        return exercise_logits, form_logit


class PoseAssessmentLoss(nn.Module):
    # Multi-task loss combining CrossEntropyLoss and class-weighted BCEWithLogitsLoss
    def __init__(self, pos_weight=None):
        super().__init__()
        self.exercise_criterion = nn.CrossEntropyLoss()
        self.form_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, exercise_logits, form_logit, target_exercise, target_form):
        loss_ex = self.exercise_criterion(exercise_logits, target_exercise)
        loss_form = self.form_criterion(form_logit, target_form)
        total_loss = loss_ex + 1.5 * loss_form
        return total_loss, loss_ex, loss_form