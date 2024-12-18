#include <EloquentTinyML.h>
#include "australian_credit_model_esp32.h"

#define NUMBER_OF_INPUTS 14
#define NUMBER_OF_OUTPUTS 1
#define TENSOR_ARENA_SIZE 5*1024

// Define min-max values for each feature (from dataset analysis)
const float feature_mins[] = {0, 13.75, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};  // Minimum values
const float feature_maxs[] = {1, 80.25, 28.0, 3, 14, 9, 28.5, 1, 1, 67, 1, 3, 2000, 100000};  // Maximum values

Eloquent::TinyML::TfLite<NUMBER_OF_INPUTS, NUMBER_OF_OUTPUTS, TENSOR_ARENA_SIZE> ml;

// Function to normalize a single value
float normalize(float value, float min_val, float max_val) {
    return (value - min_val) / (max_val - min_val);
}

// Function to normalize an entire input array
void normalizeInput(const float* raw_input, float* normalized_input) {
    for(int i = 0; i < NUMBER_OF_INPUTS; i++) {
        normalized_input[i] = normalize(raw_input[i], feature_mins[i], feature_maxs[i]);
    }
}

void setup() {
    Serial.begin(115200);
    ml.begin(australian_credit_model_esp32);
    Serial.println("Australian Credit Classification Model Loaded");
}

float fResult[NUMBER_OF_OUTPUTS] = {0};
float fRes = NULL;

void loop() {
    // Original raw input values from dataset
    float raw_input1[] = {
        0, 29.58, 1.75, 1, 4, 4, 1.25, 0, 0, 0, 1, 2, 280, 1
    }; // Expected output: 0 (Denied)
    
    float raw_input2[] = {
        0, 21.67, 11.5, 1, 5, 3, 0, 1, 1, 11, 1, 2, 0, 1
    }; // Expected output: 1 (Approved)
    
    float raw_input3[] = {
        1, 20.17, 8.17, 2, 6, 4, 1.96, 1, 1, 14, 0, 2, 60, 159
    }; // Expected output: 1 (Approved)

    float raw_input4[] = {
        1, 17.42, 6.5, 2, 3, 4, 0.125, 0, 0, 0, 0, 2, 60, 101
    }; // Expected output: 0 (Denied). //1 17.42 6.5 2 3 4 0.125 0 0 0 0 2 60 101 0



    // Normalized input arrays
    float normalized_input1[NUMBER_OF_INPUTS];
    float normalized_input2[NUMBER_OF_INPUTS];
    float normalized_input3[NUMBER_OF_INPUTS];
    float normalized_input4[NUMBER_OF_INPUTS];

    // Normalize all inputs
    normalizeInput(raw_input1, normalized_input1);
    normalizeInput(raw_input2, normalized_input2);
    normalizeInput(raw_input3, normalized_input3);
    normalizeInput(raw_input4, normalized_input4);

    // Process test case 1
    Serial.println("\n--- Test Case 1 ---");
    Serial.println("Raw input values:");
    displayRawInput(raw_input1);
    Serial.println("Normalized values:");
    displayRawInput(normalized_input1);
    
    initfResult(fResult);
    fRes = ml.predict(normalized_input1, fResult);
    Serial.print("Prediction: ");
    displayOutput(fResult);
    interpretResult(fResult[0]);

    // Process test case 2
    Serial.println("\n--- Test Case 2 ---");
    Serial.println("Raw input values:");
    displayRawInput(raw_input2);
    Serial.println("Normalized values:");
    displayRawInput(normalized_input2);
    
    initfResult(fResult);
    fRes = ml.predict(normalized_input2, fResult);
    Serial.print("Prediction: ");
    displayOutput(fResult);
    interpretResult(fResult[0]);

    // Process test case 3
    Serial.println("\n--- Test Case 3 ---");
    Serial.println("Raw input values:");
    displayRawInput(raw_input3);
    Serial.println("Normalized values:");
    displayRawInput(normalized_input3);
    
    initfResult(fResult);
    fRes = ml.predict(normalized_input3, fResult);
    Serial.print("Prediction: ");
    displayOutput(fResult);
    interpretResult(fResult[0]);

 // Process test case 4
    Serial.println("\n--- Test Case 4 ---");
    Serial.println("Raw input values:");
    displayRawInput(raw_input4);
    Serial.println("Normalized values:");
    displayRawInput(normalized_input4);
    
    initfResult(fResult);
    fRes = ml.predict(normalized_input4, fResult);
    Serial.print("Prediction: ");
    displayOutput(fResult);
    interpretResult(fResult[0]);


    delay(5000); // 5 second delay between prediction sets
}

void initfResult(float *fResult) {
    for(int i = 0; i < NUMBER_OF_OUTPUTS; i++) {
        fResult[i] = 0.0f;
    }
}

void displayRawInput(float *input) {
    for(int i = 0; i < NUMBER_OF_INPUTS; i++) {
        Serial.print(input[i], 4);
        Serial.print(" ");
    }
    Serial.println();
}

void displayOutput(float *fResult) {
    for(int i = 0; i < NUMBER_OF_OUTPUTS; i++) {
        Serial.print(fResult[i], 6);
        Serial.print(" ");
    }
    Serial.println();
}

void interpretResult(float result) {
    Serial.print("Credit Application Status: ");
    if (result > 0.5) {
        Serial.println("Approved (1)");
    } else {
        Serial.println("Denied (0)");
    }
    Serial.println("Expected output matches prediction: ");
    Serial.println(result > 0.5 ? "Yes" : "No");
}