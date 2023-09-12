run_preprocess:
	python -c 'from flood_prediction.interface.main import preprocess; preprocess()'

run_train:
	python -c 'from flood_prediction.interface.main import train; train()'

run_pred:
	python -c 'from flood_prediction.interface.main import pred; pred()'
