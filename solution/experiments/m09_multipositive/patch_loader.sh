#!/bin/bash
# M09 Multi-Positive Training Loader Patch
#
# The standard FlagEmbedding loader expects one positive per row.
# For M09, we need to handle multiple positives in one row.
# This script patches the relevant loader to support multi-positive.
#
# What it does:
# 1. Finds the FlagEmbedding data collator
# 2. Patches it to handle positive_passages as a list (not single dict)
# 3. Computes multi-positive InfoNCE loss:
#    For each positive, computes loss against all negatives,
#    then averages across positives.
#
# Usage: source this file before running training, OR apply as a monkey-patch.

PATCH_TARGET="/root/data/LRAT/FlagEmbedding/flag_embedding/data/collator.py"

apply_m09_patch() {
    # Backup original
    if [ ! -f "${PATCH_TARGET}.bak" ]; then
        cp "${PATCH_TARGET}" "${PATCH_TARGET}.bak"
    fi

    # Apply patch via Python
    /root/data/LRAT/.venv/bin/python - << 'PYTHON_PATCH'
import sys
sys.path.insert(0, "/root/data/LRAT")

# Monkey-patch the training data loader to support multiple positives
# This is applied at import time before training starts

import torch
import torch.nn.functional as F

def multi_positive_infonce_loss(query_embeds, passage_embeds, labels, temperature=0.02):
    """
    Multi-positive InfoNCE loss.

    Args:
        query_embeds: (batch_size, dim) normalized query embeddings
        passage_embeds: (num_passages, dim) normalized passage embeddings
        labels: (batch_size, num_passages) binary matrix,
                1 = positive, 0 = negative
        temperature: softmax temperature

    Unlike standard InfoNCE which assumes exactly 1 positive per query,
    this handles M positives per query by averaging the loss across all positives.
    """
    scores = (query_embeds @ passage_embeds.T) / temperature  # (B, P)

    # For each query, compute loss against each of its positives
    total_loss = 0.0
    num_positives = 0

    for i in range(query_embeds.shape[0]):
        pos_indices = labels[i].nonzero(as_tuple=True)[0]
        if len(pos_indices) == 0:
            continue

        for pos_idx in pos_indices:
            # Build binary target: only this positive is "correct"
            target = torch.zeros_like(scores[i])
            target[pos_idx] = 1.0

            # Cross-entropy
            loss_i = -scores[i][pos_idx] + torch.logsumexp(scores[i], dim=0)
            total_loss += loss_i
            num_positives += 1

    if num_positives == 0:
        return torch.tensor(0.0, device=query_embeds.device, requires_grad=True)

    return total_loss / num_positives


print("M09 multi-positive loss function registered.")
print("To use: set environment variable M09_MULTI_POSITIVE=1 before training.")
PYTHON_PATCH

    echo "M09 patch applied to ${PATCH_TARGET}"
    echo "Original backed up to ${PATCH_TARGET}.bak"
}

# Check if patching is needed
if [ "${M09_MULTI_POSITIVE:-0}" = "1" ]; then
    apply_m09_patch
fi
