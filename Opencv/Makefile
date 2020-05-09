CXX=nvcc
LDLIBS=`pkg-config --libs opencv`

blur: blur.cu
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

clean:
	rm blur
