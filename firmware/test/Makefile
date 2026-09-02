CC ?= cc
CFLAGS ?= -std=c11 -O2 -Wall -Wextra -Werror -I../main
TESTS := test_mfcc test_stream test_wav test_prompts

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

clean:
	rm -f $(TESTS)
.PHONY: all clean
