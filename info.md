# To point to the tuner logs.

Specify the path to the existing tuner_logs directory

existing_tuner_logs_path = '/path/to/existing/tuner_logs'

## Initialize tuner with existing directory

```
tuner = RandomSearch(
build_model,
objective='val_accuracy',
max_trials=2,
directory=existing_tuner_logs_path,
project_name='efficientnet_tuning'
)
```

## Retrieve the best model

best_model = tuner.get_best_models(num_models=1)[0]
