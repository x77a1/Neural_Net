CXXFLAGS =	-O2 -g -Wall -fmessage-length=0

OBJS =		GNN.o

LIBS =

TARGET =	GNN

$(TARGET):	$(OBJS)
	$(CXX) -o $(TARGET) $(OBJS) $(LIBS)

all:	$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)
