import optuna

study = optuna.load_study(
    study_name="augmentation_search",
    storage="sqlite:///augmentation_search.db",
)

print(f"Completed trials: {len(study.trials)}")
print(f"Best val accuracy: {study.best_value}")
print("Best params:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")

# Full trial history
df = study.trials_dataframe()
print(df.sort_values("value", ascending=False).head(10))
from optuna.trial import TrialState

states = [t.state for t in study.trials]
print(f"Complete: {states.count(TrialState.COMPLETE)}")
print(f"Pruned:   {states.count(TrialState.PRUNED)}")
print(f"Failed:   {states.count(TrialState.FAIL)}")
print(f"Running:  {states.count(TrialState.RUNNING)}")
# for t in study.trials:
#     if t.state == TrialState.RUNNING:
#         study.tell(t.number, state=TrialState.FAIL)
