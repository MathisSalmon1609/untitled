CXX=nvcc
LDLIBS=`pkg-config --libs opencv`

flou: flou.cu
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

detectionSommets: detectionSommets.cu
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

affinage: affinage.cu
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

clean:
	rm blur
