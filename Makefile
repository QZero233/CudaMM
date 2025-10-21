.PHONY: all
all:
	@mkdir -p build
	@cd build; cmake ..
	@cd build; make cuda_perf