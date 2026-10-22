"""Visualisation of the tracking state.

The overlay is the operator's view of the system. It shows what was observed
and what was only predicted, and it never mixes the two: a predicted position
is drawn differently from an observed one and is labelled as a prediction.
"""

from __future__ import annotations

from .renderer import DashboardStats, Renderer

__all__ = ["DashboardStats", "Renderer"]
