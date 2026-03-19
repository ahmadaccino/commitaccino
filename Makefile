CC = cc
CFLAGS = -O2 -Wall -Wextra
LDFLAGS = -lcurl
TARGET = commitaccino
PREFIX = /usr/local/bin

all: $(TARGET)

$(TARGET): commitaccino.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

install: $(TARGET)
	cp $(TARGET) $(PREFIX)/$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all install clean
