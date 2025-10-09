.PHONY: all build clean

CMAKE := cmake

BUILD_DIR := build
PROFILE_DIR := profile
all: build

build:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && $(CMAKE) -DCMAKE_BUILD_TYPE=Release ..
	@$(MAKE) -C $(BUILD_DIR)

profile: build
	@mkdir -p $(PROFILE_DIR)
	@ncu --set full --export $(PROFILE_DIR)/kernel_$(KERNEL) --force-overwrite $(BUILD_DIR)/sgemm $(KERNEL)

clean:
	@rm -rf $(BUILD_DIR)
