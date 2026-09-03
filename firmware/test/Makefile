CC ?= cc
CFLAGS ?= -std=c11 -O2 -Wall -Wextra -Werror -I../main
TESTS := test_mfcc test_stream test_wav test_prompts test_vad test_wakefront test_assist_gate

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

clean:
	rm -f $(TESTS) *.o
.PHONY: all clean
