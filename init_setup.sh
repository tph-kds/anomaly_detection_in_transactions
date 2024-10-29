
PYTHON_VERSION = 3.9

echo [$(date)]: "START INIT SETUP"


echo [$(date)]: "creating env file with python ${PYTHON_VERSION} version"


conda create -p ./anod_env python==${PYTHON_VERSION} -y


source avtivate ./anod_env


echo [$(date)]: "installing the requirements"


pip install -r requirements.txt


echo [$(date)]: "END INIT SETUP"