CPP = g++
EXTRA_CXXFLAGS =
CXXFLAGS = -Wall -g --std=c++20 -I.. $(EXTRA_CXXFLAGS)
LDFLAGS = -lopenblas

TARGETS = test_storage test_context test_engine 

SRCS = test_storage.cpp test_context.cpp test_engine.cpp
OBJS = $(SRCS:.cpp=.o)

all: $(TARGETS)

$(TARGETS): %: %.o
	$(CPP) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CPP) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(TARGETS) *.o *.dSYM *.svg *.png *.dot