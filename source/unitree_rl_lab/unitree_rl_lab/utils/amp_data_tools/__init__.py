"""AMP data preparation tools.

Design goals:
- Pure-Python inspection/splitting tools should not require Isaac Sim.
- Isaac-dependent conversion/replay tools should import Isaac only inside their own modules.
- Joint/body ordering should stay aligned with the previous Unitree 23-DoF pipeline.
"""
