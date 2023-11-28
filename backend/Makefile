#################### PACKAGE ACTIONS ###################
reinstall_package:
	@pip uninstall -y fast_api || :
	@pip install -e .


##### Prediction API - - - - - - - - - - - - - - - - - - - - - - - - -

run_api:
	uvicorn fast_api.api:app --reload
