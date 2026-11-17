"""Backbone model definitions for visual feature extraction."""

from app.features.models.osnet import OSNet, build_osnet

__all__ = ["OSNet", "build_osnet"]
