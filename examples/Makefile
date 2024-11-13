CPP = g++
EXTRA_CXXFLAGS =
CXXFLAGS = -Wall -O3 --std=c++20 -I.. $(EXTRA_CXXFLAGS)
LDFLAGS = -lopenblas

TARGETS = one_layer mlp

SRCS = one_layer.cpp mlp.cpp
OBJS = $(SRCS:.cpp=.o)

all: $(TARGETS)

$(TARGETS): %: %.o
	$(CPP) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CPP) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(TARGETS) *.o *.dSYM *.svg *.png *.dot