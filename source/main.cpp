#include "MicroBit.h"

#include "tflite_codal.h"
#include "model.h"

// The Micro:bit control object
MicroBit uBit;

TfLiteCodal * tf;

float clapDeltas[10] = { 0 };
int clapDeltasIndex = 0;
uint64_t prevTime = 0;

void reset() {
    clapDeltasIndex = 0;
    prevTime = 0;
    for (int i = 0; i < 10; i++) clapDeltas[i] = 0;
}

void onClick(MicroBitEvent) {   // click button a to reset
    DMESGF("reset");
    reset();
}

void onClap(MicroBitEvent) {    // on clap adds to clapDeltas list
    uint64_t time = uBit.timer.getTime();
    if (time-prevTime > 3000) reset();
    if (clapDeltasIndex == 0) prevTime = time;
    clapDeltas[clapDeltasIndex] = time - prevTime;
    clapDeltasIndex++;
    prevTime = time;
    if (clapDeltasIndex == 10) {
        DMESGF("more than 10 so reset");
        reset();
    }
    DMESGF("clap deltas: %d %d %d %d %d %d %d %d %d %d", (int) clapDeltas[0], (int) clapDeltas[1], (int) clapDeltas[2], (int) clapDeltas[3], (int) clapDeltas[4], (int) clapDeltas[5], (int) clapDeltas[6], (int) clapDeltas[7], (int) clapDeltas[8], (int) clapDeltas[9]);
}

void checkValid(MicroBitEvent) {    // click button b to check if match model sequence
    // pad out with -1s
    for (int i = clapDeltasIndex; i < 10; i++) clapDeltas[i] = -1;
    float result = *(float *) tf->inferArray(clapDeltas, tf->TensorType::TT_FLOAT, 10);
    DMESGF("isValid: %d", (int) result);
    reset();
}

// Out main function, run at startup
int main() {
    uBit.init();

    DMESGF("hello world");

    // init ML
    tf = new TfLiteCodal();
    tf->initialise(model_tflite, 6000);

    // init microphone (and clap detector)
    uBit.audio.activateMic();
    uBit.audio.levelSPL->getValue();    // level SPL has lazy initialisation - this wakes it up!

    // event listeners
    uBit.messageBus.listen(DEVICE_ID_MICROPHONE, LEVEL_DETECTOR_SPL_CLAP, onClap);
    uBit.messageBus.listen(DEVICE_ID_BUTTON_A, DEVICE_BUTTON_EVT_CLICK, onClick);
    uBit.messageBus.listen(DEVICE_ID_BUTTON_B, DEVICE_BUTTON_EVT_CLICK, checkValid);

    release_fiber();
}