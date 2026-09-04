CC ?= cc
CFLAGS ?= -std=c11 -O2 -Wall -Wextra -Werror -I../main
TESTS := test_mfcc test_stream test_wav test_prompts test_vad test_wakefront test_assist_gate \
         test_wake_smoke test_command_smoke

# Where the generated inference C and its vector headers live. Defaults to
# the committed firmware/main/gen; tests/test_codegen_parity.py's whole-model
# tests override this on the `make` command line to a scratch tmp_path
# instead, so regenerating for a real-clip parity run never mutates the
# tracked gen/ files. The two *_smoke targets below always build against the
# default -- their vectors are committed, not regenerated per test run.
GEN_DIR ?= ../main/gen

# The wake front-end is upstream TFLite-Micro/kissfft C++ (.cc), vendored
# verbatim so it stays bit-identical to what pymicro-features compiles — hence
# its own $(CXX) rule and no -Wextra -Werror (we do not police third-party code).
MF := ../main/microfrontend
MF_SRCS := $(wildcard $(MF)/tensorflow/lite/experimental/microfrontend/lib/*.cc) \
           $(MF)/kissfft/kiss_fft.cc $(MF)/kissfft/tools/kiss_fftr.cc
CXXFLAGS ?= -O2 -Wall -I../main -I$(MF) -I$(MF)/kissfft

all: $(TESTS)
	@for t in $(TESTS); do ./$$t || exit 1; done
	@echo "host tests OK"

# mfcc.c stays C (and keeps -Wextra -Werror); its FFT shim and kissfft are C++,
# so those compile separately and the link is $(CXX).
MFCC_CXX_SRCS := ../main/mfcc_fft.cc $(MF)/kissfft/kiss_fft.cc $(MF)/kissfft/tools/kiss_fftr.cc

test_mfcc: test_mfcc.c ../main/mfcc.c $(MFCC_CXX_SRCS)
	$(CC) $(CFLAGS) -c test_mfcc.c -o test_mfcc.o
	$(CC) $(CFLAGS) -c ../main/mfcc.c -o mfcc.o
	$(CXX) $(CXXFLAGS) -o $@ test_mfcc.o mfcc.o $(MFCC_CXX_SRCS) -lm
test_stream: test_stream.c ../main/stream.c
	$(CC) $(CFLAGS) -o $@ $^ -lm
test_wav: test_wav.c ../main/wav.c
	$(CC) $(CFLAGS) -o $@ $^
test_prompts: test_prompts.c ../main/prompts.c
	$(CC) $(CFLAGS) -o $@ $^
test_vad: test_vad.c ../main/vad.c
	$(CC) $(CFLAGS) -o $@ $^ -lm
test_assist_gate: test_assist_gate.c ../main/assist_gate.c
	$(CC) $(CFLAGS) -o $@ $^
test_wakefront: test_wakefront.c ../main/wakefront.c $(MF_SRCS)
	$(CXX) $(CXXFLAGS) -o $@ -x c++ $^ -lm

# esp-nn, pinned to the exact commit the device builds against. The last link
# in the bit-exactness chain is "esp-nn's ANSI kernels and its S3 kernels agree
# bit for bit", so both sides of it have to be the same release: firmware/main/
# idf_component.yml pins espressif/esp-nn to ==1.3.1, and this is the upstream
# commit that release was cut from (verified byte-identical against
# firmware/managed_components/espressif__esp-nn). A commit rather than a tag
# because upstream published 1.3.1 to the component registry without cutting a
# v1.3.1 git tag — and because a tag can move, which is why every `uses:` in
# .github/workflows is SHA-pinned too. Fetched into an ignored directory rather
# than vendored, so it cannot drift silently.
# NEVER define SKIP_NUDGE / CONFIG_NN_SKIP_NUDGE here: that macro selects
# esp-nn's faster, NON-bit-exact requantisation and would silently break parity.
# test_infer_parity.c #errors if either is defined, so the build cannot lie.
# test_infer_parity/test_wake_parity/test_command_parity stay out of $(TESTS):
# each needs sources only tests/test_codegen_parity.py produces (conv_probe.c,
# a KWS_DATA_ROOT model's *_infer.c, real-clip vectors), so `make -C
# firmware/test` must stay green without any of that. The *_smoke targets are
# different -- their inputs (the committed *_infer.c and *_smoke_vectors.h)
# need no model or data, only esp-nn itself, so they *are* in $(TESTS) and
# pull the $(ESP_NN) network clone into the default target.
ESP_NN_REF ?= 52d6de8a6cffd7afb31ed3c850687ca60087fc0b   # esp-nn 1.3.1
ESP_NN ?= .esp-nn
ESP_NN_SRCS := $(ESP_NN)/src/convolution/esp_nn_conv_ansi.c \
               $(ESP_NN)/src/convolution/esp_nn_depthwise_conv_ansi.c \
               $(ESP_NN)/src/fully_connected/esp_nn_fully_connected_ansi.c \
               $(ESP_NN)/src/softmax/esp_nn_softmax_ansi.c \
               $(ESP_NN)/src/common/esp_nn_mean_ansi.c
# Third-party sources: -Wall only, as with the vendored microfrontend above
# (esp-nn's unused-parameter stubs are not ours to police).
ESP_NN_CFLAGS := -std=c11 -O2 -Wall -I$(ESP_NN)/include -I$(ESP_NN)/src/common

# Not --depth 1 --branch: $(ESP_NN_REF) is a commit, and a shallow clone cannot
# check one out unless it happens to be the tip.
$(ESP_NN):
	git clone --quiet https://github.com/espressif/esp-nn $(ESP_NN)
	git -C $(ESP_NN) checkout --quiet --detach $(ESP_NN_REF)

# Phony on purpose: the probe source is rewritten per layer by the parity test,
# and on a 2-second-resolution filesystem make would call a stale binary
# "up to date" and silently compare the previous layer.
test_infer_parity: test_infer_parity.c $(GEN_DIR)/conv_probe.c | $(ESP_NN)
	$(CC) $(CFLAGS) -I$(GEN_DIR) -I$(ESP_NN)/include -c test_infer_parity.c -o test_infer_parity.o
	$(CC) $(CFLAGS) -I$(ESP_NN)/include -c $(GEN_DIR)/conv_probe.c -o conv_probe.o
	$(CC) $(ESP_NN_CFLAGS) -o $@ test_infer_parity.o conv_probe.o $(ESP_NN_SRCS) -lm

# Whole-model parity against real clips, same deal: generated by
# tests/test_codegen_parity.py into GEN_DIR (a scratch tmp_path when that test
# runs -- see the GEN_DIR comment up top), so not in $(TESTS) and phony for the
# same stale-binary reason as above. One target per model because each test
# regenerates only its own gen/ sources — sharing a binary would make the
# conv-probe tests depend on a generated wake model and vice versa.
test_wake_parity: test_wake_parity.c $(GEN_DIR)/wake_infer.c | $(ESP_NN)
	$(CC) $(CFLAGS) -I$(GEN_DIR) -I$(ESP_NN)/include -c test_wake_parity.c -o test_wake_parity.o
	$(CC) $(CFLAGS) -I$(ESP_NN)/include -c $(GEN_DIR)/wake_infer.c -o wake_infer.o
	$(CC) $(ESP_NN_CFLAGS) -o $@ test_wake_parity.o wake_infer.o $(ESP_NN_SRCS) -lm

test_command_parity: test_command_parity.c $(GEN_DIR)/command_infer.c | $(ESP_NN)
	$(CC) $(CFLAGS) -I$(GEN_DIR) -I$(ESP_NN)/include -c test_command_parity.c -o test_command_parity.o
	$(CC) $(CFLAGS) -I$(ESP_NN)/include -c $(GEN_DIR)/command_infer.c -o command_infer.o
	$(CC) $(ESP_NN_CFLAGS) -o $@ test_command_parity.o command_infer.o $(ESP_NN_SRCS) -lm

# Model-free, data-free smoke parity: the committed wake_infer.c and
# wake_smoke_vectors.h, both tracked in git, need no regeneration -- so unlike
# the three targets above this one is a plain (non-phony) rule and belongs in
# $(TESTS). Deliberately always builds against the default $(GEN_DIR), never
# an override: the whole point is proving the *committed* C is still correct.
test_wake_smoke: test_wake_smoke.c $(GEN_DIR)/wake_infer.c $(GEN_DIR)/wake_smoke_vectors.h | $(ESP_NN)
	$(CC) $(CFLAGS) -I$(GEN_DIR) -I$(ESP_NN)/include -c test_wake_smoke.c -o test_wake_smoke.o
	$(CC) $(CFLAGS) -I$(ESP_NN)/include -c $(GEN_DIR)/wake_infer.c -o wake_infer_smoke.o
	$(CC) $(ESP_NN_CFLAGS) -o $@ test_wake_smoke.o wake_infer_smoke.o $(ESP_NN_SRCS) -lm

test_command_smoke: test_command_smoke.c $(GEN_DIR)/command_infer.c $(GEN_DIR)/command_smoke_vectors.h | $(ESP_NN)
	$(CC) $(CFLAGS) -I$(GEN_DIR) -I$(ESP_NN)/include -c test_command_smoke.c -o test_command_smoke.o
	$(CC) $(CFLAGS) -I$(ESP_NN)/include -c $(GEN_DIR)/command_infer.c -o command_infer_smoke.o
	$(CC) $(ESP_NN_CFLAGS) -o $@ test_command_smoke.o command_infer_smoke.o $(ESP_NN_SRCS) -lm

# Also the generated vector headers: they are produced from whatever model the
# generator was last pointed at, and a stale one silently fails a hand-run
# parity target against perfectly good committed C.
clean:
	rm -f $(TESTS) test_infer_parity test_wake_parity test_command_parity *.o
	rm -f $(GEN_DIR)/*_infer_vectors.h $(GEN_DIR)/conv_probe.c $(GEN_DIR)/conv_probe_vectors.h
.PHONY: all clean test_infer_parity test_wake_parity test_command_parity
