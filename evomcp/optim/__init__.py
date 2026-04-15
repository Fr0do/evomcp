"""Optimizer entry points.

- gepa_runner.run(config)   : evolve text_genome (prompts, rubrics)
- evox_runner.run(config)   : evolve prog_genome (hyperparams, patches)
- hybrid_runner.run(config) : outer EvoX proposes prog candidates,
                              inner GEPA refines text for each promising one
"""
