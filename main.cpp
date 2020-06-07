#include <torch/torch.h>
#include <iostream>

/*

rm -rf build;mkdir build;cd build;cmake \
-DCMAKE_C_COMPILER=clang \
-DCMAKE_CXX_COMPILER=clang++ \
-DCMAKE_PREFIX_PATH="$PWD/libtorch" ..;make VERBOSE=1;cd ..

./build/app

*/


at::Tensor distanceCosine(at::Tensor tensor1, at::Tensor tensor2)
{
    at::Tensor mult = tensor1 * tensor2;
    at::Tensor uv = mult.mean();

    at::Tensor arr1TensorSquare = tensor1.square();
    at::Tensor uu = arr1TensorSquare.mean();

    at::Tensor arr2TensorSquare = tensor2.square();
    at::Tensor vv = arr2TensorSquare.mean();

    at::Tensor uuMultVv = uu * vv;
    at::Tensor uuvvSqrt = uuMultVv.sqrt();

    at::Tensor dist = 1.0 - uv / uuvvSqrt;

    return dist;
}

int main() 
{

    auto options = torch::TensorOptions().dtype(torch::kFloat64);

    std::cout << "Play Start------------------------------------------------: " << std::endl;
    double arrAvg[4] = { 1.0, 2.0, 3.0, 4.0 }; 
    at::Tensor arrAvgTensor = torch::from_blob(arrAvg, {1, 4}, options); 
    std::cout << "arrAvgTensor: " << arrAvgTensor << '\n';


    std::cout << '\n';
    std::cout << "Mean : " << arrAvgTensor.mean() << '\n'; // 2.5
    std::cout << "Square : " << arrAvgTensor.square() << '\n'; // [ 1  4  9 16]
    std::cout << "Sqrt : " << arrAvgTensor.sqrt() << '\n'; // [1. 1.41421356 1.73205081 2.   ]

    std::cout << "\nPlay End------------------------------------------------: " << std::endl;


    /////////////////////////////////////////////////////////////////////////
    std::cout << "\n\nDistance Start------------------------------------------------: " << std::endl;

    //   double  embedding[512];
    double arr1[3] = { 1.0, 1.0, 0.0 }; 
    double arr2[3] = { 0.0, 1.0, 0.0 }; 

    at::Tensor arr1Tensor = torch::from_blob(arr1, {1, 3}, options); 
    at::Tensor arr2Tensor = torch::from_blob(arr2, {1, 3}, options); 
    std::cout << "Tensor 1: " << arr1Tensor << '\n';
    std::cout << "Tensor 2: " << arr2Tensor << '\n';

    /*
        ###### Based on `scipy`
        uv = np.average(embeddings1 * embeddings2)
        uu = np.average(np.square(embeddings1))
        vv = np.average(np.square(embeddings2))
        dist = 1.0 - uv / np.sqrt(uu * vv)

    */

    std::cout << "\n\nDistance End------------------------------------------------: " << std::endl;
    
    auto dist1 = distanceCosine(arr1Tensor, arr2Tensor);
    std::cout << "Distance1 : " << dist1 << '\n';

}

