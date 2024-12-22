#pragma once
#include <iomanip>
#include "resource.h"

// Unused but might use later idk
inline std::string removeChar(std::string str, char charToRemove) {
    str.erase(std::remove(str.begin(), str.end(), charToRemove), str.end());
    return str;
}

// Writes to log.txt
inline void writeToLog(const std::string& message) {

    std::ofstream logFile("C:\\Users\\yohan\\source\\repos\\ISLES\\log.txt", std::ios::app);
    if (logFile.is_open()) {
        logFile << message << std::endl;
        logFile.flush(); 
    }
}

// Sub functions
inline void writeToLogNoLine(const std::string& message) {

    std::ofstream logFile("C:\\Users\\yohan\\source\\repos\\ISLES\\log.txt", std::ios::app);
    if (logFile.is_open()) {
        logFile << message;
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
