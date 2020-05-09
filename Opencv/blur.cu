#include <vector>
#include <opencv2/opencv.hpp>


using namespace std;


void blur(unsigned char * rgb_in, unsigned char * rgb_out, int rows, int cols) {
    for (int row = 1; row < rows - 1; ++row) {
        for (int col = 1; col < cols - 1; ++col) {
            for (int i = 0; i < 3; ++i)
            {
                unsigned char hg = rgb_in[3 * (row * cols + col - 3) + i];
                unsigned char h = rgb_in[3 * (row * cols + col) + i];
                unsigned char hd = rgb_in[3 * (row * cols + col + 3) + i];
                unsigned char g = rgb_in[3 * (row * cols + col - 3) + i];
                unsigned char c = rgb_in[3 * (row * cols + col) + i];
                unsigned char d = rgb_in[3 * (row * cols + col + 3) + i];
                unsigned char bg = rgb_in[3 * (row * cols + col - 3) + i];
                unsigned char b = rgb_in[3 * (row * cols + col) + i];
                unsigned char bd = rgb_in[3 * (row * cols + col + 3) + i];
                rgb_out[3 * (row * cols + col) + i] = (hg + h + hd + g + c + d + bg + b + bd) / 9;
            }
        }
    }
}


int main()
{
    //Declarations
    cv::Mat m_in = cv::imread("in.jpg", cv::IMREAD_UNCHANGED);
    auto rgb = m_in.data;
    auto rows = m_in.rows;
    auto cols = m_in.cols;

    size_t taille_rgb = rows * 3 * cols;
    std::vector< unsigned char > g( taille_rgb );
    cv::Mat m_out( rows, cols, CV_8UC3, g.data() );

    unsigned char * rgb_in;
    unsigned char * rgb_out;

    //Init donnes kernel
    cudaMallocHost( &rgb_in, taille_rgb);
    cudaMallocHost( &rgb_out, taille_rgb);
    cudaMemcpy( rgb_in, rgb, taille_rgb, cudaMemcpyHostToDevice );

    //Debut de chrono
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    blur(rgb_in, rgb_out, rows, cols);

    //Fin de chrono
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cout << cudaGetErrorString(cudaGetLastError()) << endl;
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout << elapsedTime << endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    //Recup donnees kernel
    cudaMemcpy(g.data(), rgb_out, taille_rgb, cudaMemcpyDeviceToHost );
    cv::imwrite( "out_blur.jpg", m_out );
    cudaFree(rgb_in);
    cudaFree(rgb_out);
    return 0;
}