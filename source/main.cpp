#include "MicroBit.h"

#include "tflite_codal.h"
#include "model.h"

// The Micro:bit control object
MicroBit uBit;

// Out main function, run at startup
int main() {
    uBit.init();

    DMESGF("hello world");
    TfLiteCodal * tf = new TfLiteCodal();
    DMESGF("initialising model...");
    tf->initialise(model_tflite, 6000);

    DMESGF("done initialising");

    // test good input
    float goodInput [] = { 0, 335, 1094, 1014, 437, -1, -1, -1, -1, -1 };
    float result = *(float *) tf->inferArray(goodInput, tf->TensorType::TT_FLOAT, 10);
    DMESGF("result g1: %d", static_cast<int>(result));

    // test bad input
    float badInput [] = { 0,403,392,403,426,392, -1, -1, -1, -1 };
    result = *(float *) tf->inferArray(badInput, tf->TensorType::TT_FLOAT, 10);
    DMESGF("result b: %d", static_cast<int>(result));

    // test another good input
    float goodInput2 [] = { 0,391,991,1129,392, -1, -1, -1, -1, -1 };
    result = *(float *) tf->inferArray(goodInput2, tf->TensorType::TT_FLOAT, 10);
    DMESGF("result g2: %d", static_cast<int>(result));

}