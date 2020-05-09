CXX=nvcc
LDLIBS=`pkg-config --libs opencv`

main_seq: main.cu
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

clean:
	rm main
