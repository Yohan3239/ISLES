#pragma once
#include <iomanip>
#include "resource.h"

inline std::string removeChar(std::string str, char charToRemove) {
    str.erase(std::remove(str.begin(), str.end(), charToRemove), str.end());
    return str;
}
inline void writeToLog(const std::string& message) {

    std::ofstream logFile("C:\\Users\\yohan\\source\\repos\\ISLES\\log.txt", std::ios::app);
    if (logFile.is_open()) {
        logFile << removeChar(message, '0') << std::endl;
        logFile.flush();
    }
}

inline void writeToLogNoLine(const std::string& message) {

    std::ofstream logFile("C:\\Users\\yohan\\source\\repos\\ISLES\\log.txt", std::ios::app);
    if (logFile.is_open()) {
        logFile << removeChar(message, '0');
        logFile.flush();
    }
}

inline void endLine() {

    std::ofstream logFile("C:\\Users\\yohan\\source\\repos\\ISLES\\log.txt", std::ios::app);
    if (logFile.is_open()) {
        logFile << std::endl;
        logFile.flush();
    }
}
inline void handleException(const std::exception& e) {
    std::string errorMessage = "Runtime error: " + std::string(e.what());
    writeToLog(errorMessage);  // Log to file
}