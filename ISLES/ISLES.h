#pragma once
#include <iomanip>
#include <omp.h>
#include <unordered_map>
#include "CNN.h"
#include "resource.h"

// Writes to log.txt
inline void writeToLog(const std::string& message, std::unordered_map<std::string, std::string> config) {

    
    std::ofstream logFile(config["LOG_PATH"], std::ios::app);
    if (logFile.is_open()) {
        logFile << message << std::endl;
        logFile.flush(); 
    }
}
inline void writeToEpochs(const std::string& message, std::unordered_map<std::string, std::string> config) {

    std::ofstream epochFile(config["EPOCH_LOG_PATH"], std::ios::app);
    if (epochFile.is_open()) {
        epochFile << message << std::endl;
        epochFile.flush();
    }
}
// Sub functions
inline void writeToLogNoLine(const std::string& message, std::unordered_map<std::string, std::string> config) {

    std::ofstream logFile(config["LOG_PATH"], std::ios::app);
    if (logFile.is_open()) {
        logFile << message;
        logFile.flush();
    }
}

inline void endLine(std::unordered_map<std::string, std::string> config) {

    std::ofstream logFile(config["LOG_PATH"], std::ios::app);
    if (logFile.is_open()) {
        logFile << std::endl;
        logFile.flush();
    }
}

