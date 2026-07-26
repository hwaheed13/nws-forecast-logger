"""nwslogger — the renovated package (strangler pattern, 2026-07).

Modules migrate here from the legacy monoliths (prediction_writer.py,
train_models.py) phase by phase. Old entrypoints keep working via import
shims until each phase's switch commit. See docs/RENOVATION_BLUEPRINT.md.
"""
