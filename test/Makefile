CPP = g++
EXTRA_CXXFLAGS =
CXXFLAGS = -Wall -g --std=c++20 -I.. $(EXTRA_CXXFLAGS)

TARGETS = test_storage test_context test_engine 

SRCS = test_storage.cpp test_context.cpp test_engine.cpp
OBJS = $(SRCS:.c=.o)

all: $(TARGETS)

$(TARGET): $(OBJS)
	$(CPP) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CPP) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(TARGETS) *.dSYM
