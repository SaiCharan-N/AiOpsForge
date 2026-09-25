"""The orchestration layer: LangGraph wiring (graph.py), node functions
(nodes.py), and the task-type extension point (task_handlers/).

Nothing outside this package should need to know how a task type's
generation actually works — app.main only ever calls
app.pipeline.graph.run_pipeline().
"""
