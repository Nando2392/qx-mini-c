CC ?= cc
CFLAGS ?= -std=c17 -O2 -Wall -Wextra -Wpedantic -D_CRT_SECURE_NO_WARNINGS -Iinclude
LDFLAGS ?=

BUILD := build
FIT_SRC := src/qxfit.c src/qx_main.c
QXF_SRC := src/qx_format.c src/qx_gguf.c src/qx_tokenizer.c src/qx_qxf_main.c
FIT_BIN := $(BUILD)/qxfit
QXF_BIN := $(BUILD)/qxqxf

.PHONY: all clean smoke

all: $(FIT_BIN) $(QXF_BIN)

$(BUILD):
	mkdir -p $(BUILD)

$(FIT_BIN): $(FIT_SRC) include/qxfit.h | $(BUILD)
	$(CC) $(CFLAGS) $(FIT_SRC) -o $(FIT_BIN) $(LDFLAGS)

$(QXF_BIN): $(QXF_SRC) include/qx_format.h include/qx_gguf.h include/qx_tokenizer.h | $(BUILD)
	$(CC) $(CFLAGS) $(QXF_SRC) -o $(QXF_BIN) $(LDFLAGS)

smoke: $(FIT_BIN) $(QXF_BIN)
	./$(FIT_BIN) --model qwen3-8b --weight-gib 3.3 --ctx 4096 --kv int8
	./$(FIT_BIN) --model qwen3-30b-a3b --weight-gib 12.38 --ctx 4096 --kv int8 || true
	./$(QXF_BIN) create --model qwen3-8b --quant q3q4mix --out models/qwen3-8b-meta.qxf
	./$(QXF_BIN) inspect --in models/qwen3-8b-meta.qxf

clean:
	rm -rf $(BUILD)
