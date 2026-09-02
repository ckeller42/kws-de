CC ?= cc
CFLAGS ?= -std=c11 -O2 -Wall -Wextra -Werror -I../main
TESTS := test_mfcc test_stream test_wav test_prompts test_vad test_wakefront

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

test_mfcc: test_mfcc.c ../main/mfcc.c
	$(CC) $(CFLAGS) -o $@ $^ -lm
test_stream: test_stream.c ../main/stream.c
	$(CC) $(CFLAGS) -o $@ $^ -lm
test_wav: test_wav.c ../main/wav.c
	$(CC) $(CFLAGS) -o $@ $^
test_prompts: test_prompts.c ../main/prompts.c
	$(CC) $(CFLAGS) -o $@ $^
test_vad: test_vad.c ../main/vad.c
	$(CC) $(CFLAGS) -o $@ $^ -lm
test_wakefront: test_wakefront.c ../main/wakefront.c $(MF_SRCS)
	$(CXX) $(CXXFLAGS) -o $@ -x c++ $^ -lm

clean:
	rm -f $(TESTS)
.PHONY: all clean
