---
name: project-pyeki
description: pyEKI is a library, not a research repo; scope boundaries and current state
metadata:
  type: project
---

pyEKI implements Ensemble Kalman Inversion as a reusable library. It was
extracted from a research repository in August 2026 so that the EKI machinery
could be shared across projects.

**In scope:** structured linear operators (`pyeki.linalg`), joint Gaussian
conditioning, localization, and the EKI algorithms themselves.

**Out of scope, permanently:** forward models, priors, Gaussian process
kernels, and anything domain-specific. The forward model is any callable from
parameters to predicted observations. Domain knowledge must not appear in this
package, including in docstrings and examples.

**Why:** the calling repositories are research code with their own concerns;
pyEKI stays a dependable library that colleagues can build on.

**How to apply:** read `CLAUDE.md` for conventions and `HANDOFF.md` for current
state and next steps. If a docstring wants to mention a specific application,
that content belongs in the calling repository instead.
