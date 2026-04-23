This folder contains logic for handling task dependencies within the pipelines. 

`task_dependencies.py` contains the global dependency graph across all current (heat map, momentum graph) and future workflows. When defining new tasks, add it to that file. Use `TaskGraph` to manage these dependencies. 

To create views such as the momentum graph workflow, use `TaskGraphView` and pass in the global `TaskGraph` to it.

There used to be some code for visualising it, but it's deemed as not useful. Refer to the git history to retrieve the files should they be of interest.