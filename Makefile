.PHONY: hello
hello:
	@echo "Hello Everyone."

.PHONY: help
help:
	@echo "Usage: make [target]"
	@echo "  hello: Print \"Hello Everyone!\"."

.PHONY: env
env:
	set PYTHONPATH=${ROOT}



# DEVELOPMENT
define anod_env
conda create -p anod_env python==3.9 -y
endef

## # MONITORING # ##

.PHONY: up
up: 
	docker compose -f docker-compose.yaml up  -d --build

.PHONY: down
down: 
	docker compose down

# logs_name ?= grafana
.PHONY: logs
logs: 
	docker compose logs otel-collector loki grafana prometheus alertmanager
# ifeq (${logs_name},true) 
# 	echo "Getting logs... Please wait..."
# 	git commit -m "${message}" 
# 	git push origin ${branch}

# else ifeq (${status}, false)
# 	echo "Withdrawing commit message..."
# 	git reset
		


## # TESTING # ##

.PHONY: test
test: 
	python tests/integration/main.py

.PHONY: qs
qs: 
	python tests/integration/quick_start.py

.PHONY: install
install:
	@$(PIP) install -r requirements.txt

message ?= "fix: test" 
status ?= true
branch ?= main

# args = $(foreach a,$($(subst -,_,$1)_args),$(if $(value $a),$a="$($a)"))

# rule1_args = message
# rule2_args = status
# rule3_args = branch

.PHONY: git_add
git_add:
	git add . && git status 

.PHONY: git_commit
git_commit:
ifeq (${status},true) 
	echo "Commit message:"
	git commit -m "${message}"
	git push origin ${branch}

else ifeq (${status}, false)
	echo "Withdrawing commit message..."
	git reset

else 
	echo "Error: Invalid status value. Only 'true' or 'false' are allowed."
	exit 1 

endif 


.PHONY: custom_task
custom_task:
	@echo "Custom message: ${message}"
	@echo "Branch: ${branch}"
	@echo "Status: ${status}"
	


.PHONY: clean
clean:
	python -rf __pycache__

