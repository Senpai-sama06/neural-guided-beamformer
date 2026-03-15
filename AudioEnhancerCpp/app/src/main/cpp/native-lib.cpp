#include <jni.h>
#include <string>
#include <vector>
#include "AudioEngine.h"

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_example_audioenhancercpp_MainActivity_processAudioJNI(
        JNIEnv* env,
        jobject /* this */,
        jstring modelPath,
        jfloatArray inputAudio) {

    // 1. Convert jstring to std::string
    const char *pathChars = env->GetStringUTFChars(modelPath, 0);
    std::string pathStr(pathChars);
    env->ReleaseStringUTFChars(modelPath, pathChars);

    // 2. Convert jfloatArray to std::vector
    jsize len = env->GetArrayLength(inputAudio);
    jfloat *body = env->GetFloatArrayElements(inputAudio, 0);
    std::vector<float> inputVec(body, body + len);
    env->ReleaseFloatArrayElements(inputAudio, body, 0);

    // 3. Run Audio Engine
    AudioEngine engine(pathStr);
    std::vector<float> outputVec = engine.processAudio(inputVec);

    // 4. Convert Result back to jfloatArray
    jfloatArray result = env->NewFloatArray(outputVec.size());
    env->SetFloatArrayRegion(result, 0, outputVec.size(), outputVec.data());

    return result;
}