/* Compression.c */
#include "Compression.h"
#include <math.h>

/**
 * Computes the number of bytes needed to encode a document ID using variable-byte encoding
 *
 * Variable-byte encoding uses 7 bits per byte for data and 1 bit as continuation flag.
 * Each byte except the last has its highest bit set to 1, indicating more bytes follow.
 * The last byte has its highest bit set to 0, indicating the end of the encoded number.
 *
 * @param docId The document ID to be encoded
 * @return Number of bytes needed for encoding
 */
int computeVarByteLength(int docId) {
    int byteCount = 0;
    // Keep shifting right by 7 bits until we've processed all significant bits
    while (docId >= 128) {  // 128 = 2^7, need another byte if value >= 128
        docId >>= 7;
        byteCount++;
    }
    return byteCount + 1;   // Add 1 for the last byte
}

/**
 * Compresses an integer using variable-byte encoding
 *
 * Uses 7 bits per byte for data and 1 bit as continuation flag.
 * Each byte except the last has its highest bit set to 1.
 * The last byte has its highest bit set to 0.
 *
 * @param docId Document ID to compress
 * @param byteBuffer Buffer to store compressed bytes
 * @return Number of bytes written to buffer
 *
 * Example:
 *   docId = 130 (10000010 in binary)
 *   Encoded as: [10000010, 00000001]
 *   Where:
 *     First byte: 1|0000010 (continuation bit|7 data bits)
 *     Second byte: 0|0000001 (end bit|7 data bits)
 *   When decoded: 0000001|0000010 = 130
 */
size_t varByteCompressInt(uint32_t docId, uint8_t *byteBuffer) {
    size_t byteCount = 0;
    // Process 7 bits at a time, setting continuation bit
    while (docId >= 128) {
        byteBuffer[byteCount] = (docId & 127) | 128;    // Keep 7 bits and set high bit to 1
        docId >>= 7;    // Move to next 7 bits
        byteCount++;
    }
    // Write final byte (high bit is 0)
    byteBuffer[byteCount] = docId;
    return byteCount + 1;
}

/**
 * Compresses a double value using linear compression
 *
 * Assumes input score ranges from 0 to 30 and linearly maps it to the range 0-255.
 * This provides a simple and efficient compression for bounded scores.
 *
 * @param impactScore Score value to compress (must be in the range 0-30)
 * @return Compressed value as a single byte (0-255)
 *
 * Compression process:
 * 1. Ensure the input is within the valid range [0, 30]
 * 2. Linearly scale the value to fit in the byte range [0, 255]
 * 3. Cast to uint8_t
 *
 * Example ranges:
 * - Input 0.0 -> Output 0
 * - Input 15.0 -> Output ~128
 * - Input 30.0 -> Output 255
 */
uint8_t linearCompressDouble(double impactScore) {
    if (impactScore <= 0.0) {
        return 0;   // Clamp to 0 for scores below 0
    }
    if (impactScore >= 30.0) {
        return 255; // Clamp to 255 for scores above 30
    }
    // Scale the score linearly from [0, 30] to [0, 255]
    uint8_t compressed = (uint8_t)((impactScore / 30.0) * 255);
    return compressed;
}