"""Training learned brains for ABMA animals.

``fnt.abma.core`` can *run* a brain with nothing but numpy. This package is
what *makes* one, and is the only part of ABMA that needs PyTorch:

    python -m fnt.abma.learn --name vole_forager --minutes 20

See :mod:`.nursery` for the worlds a brain is raised in and :mod:`.ppo` for the
learner.
"""
