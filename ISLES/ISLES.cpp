// ISLES.cpp : Defines the entry point for the application.

#include <iostream>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <chrono>
#include "ISLES.h"
#include "CNN.h"

#include <cstdio>
#include <windows.h>
#include <commdlg.h>
#include <vector>
#include <string>
#include <unordered_map>


using namespace C;
using namespace std;
using filterCache = vector<vector<vector<vector<vector<float>>>>>;
using filterContainer = vector<filterCache>;

// Print a specific slice of the 3D volumes
void static printImages(const vector<vector<vector<vector<float>>>>& grids) {
    int slice = grids[0][0][0].size()/2;
    
    for (int k = 0; k < grids.size(); ++k) {
        cout << "\n";
        for (int i = grids[k].size() - 1; i >= 0; --i) {
            for (int j = 0; j < grids[k][0].size(); ++j) {
                if (grids[k][i][j][slice] > 0.9) {
                    cout << "@";
                }
                else if (grids[k][i][j][slice] > 0.8) {
                    cout << "%";
                }
                else if (grids[k][i][j][slice] > 0.7) {
                    cout << "#";
                }
                else if (grids[k][i][j][slice] > 0.6) {
                    cout << "*";
                }
                else if (grids[k][i][j][slice] > 0.5) {
                    cout << "+";
                }
                else if (grids[k][i][j][slice] > 0.4) {
                    cout << "=";
                }
                else if (grids[k][i][j][slice] > 0.3) {
                    cout << "~";
                }
                else if (grids[k][i][j][slice] > 0.2) {
                    cout << "-";
                }
                else if (grids[k][i][j][slice] > 0.1) {
                    cout << ",";
                }
                else if (grids[k][i][j][slice] > 0) {
                    cout << ".";
                }
                else if (grids[k][i][j][slice] == 0) {
                    cout << " ";
                }
            }
            cout << "\n";
        }
    }
    cout << "////////////////////////////////////////////////////////////////////";
    cout << "\n\n\n\n\n\n\n\n\n\n";
    cout << "////////////////////////////////////////////////////////////////////";
}

// Print the mean of the output in the forward pass
bool static printMeanActivation(const vector<vector<vector<vector<float>>>>& grids, string layerName) {
    float sum = 0.f;
    int count = 0;
    for (const auto& grid : grids) {
        for (const auto& slice : grid) {
            for (const auto& row : slice) {
                for (float val : row) {
                    sum += val;
                    count++;
                }
            }
        }
    }
    float mean = sum / count;
    // Check for NaN
    if (isnan(mean)) return false;

    cout << "\nFrontpropagation: Mean of " << layerName << ": " << mean << endl;
    return true;
}

// Print the mean of the output in the backward pass
void static printMeanGradSum(const vector<vector<vector<vector<float>>>>& grad, string layerName) {
    float gradSum = 0.f;
    int count = 0;
    for (auto& grid : grad) {
        for (auto& slice : grid) {
            for (auto& row : slice) {
                for (auto& val : row) {
                    gradSum += val;
                    count++;
                }
            }
        }
    }
    cout << "\nBackpropagation: Mean of : " << layerName << ": " << to_string(gradSum / count);
}

// Train the network
static void train(string fileNum, CNN& CNNetwork, float learningRate, unordered_map<string, string>& config) {
    //auto start = chrono::high_resolution_clock::now();
    // Use configuration file to set file paths
    string dataPath = config["EXTRACTED_DATA_PATH"];
    string filename_FLAIR = dataPath + fileNum + "_ses-0001_FLAIR.nii";
    string filename_ADC = dataPath + fileNum + "_ses-0001_adc.nii";
    string filename_DWI = dataPath + fileNum + "_ses-0001_dwi.nii";
    string filename_MSK = dataPath + fileNum + "_ses-0001_msk.nii";
        
    writeToLog(fileNum + " Input Preprocessing.", config);
    // Read ADC file
    if (!CNNetwork.readNifti(filename_ADC, false)) return;
    CNNetwork.clear();
    // Read DWI file
    if (!CNNetwork.readNifti(filename_DWI, false)) return;
    // Read FLAIR file
    if (!CNNetwork.readNifti(filename_FLAIR, true)) return;
    // Read Ground truth mask file
    if (!CNNetwork.readNifti(filename_MSK, false)) return;
    // Check for NaN
    if (!printMeanActivation({ CNNetwork.groundTruthGrid }, "truth")) {
        cerr << "Error: Invalid input!";
        return;
    }


    writeToLog(fileNum + " Forward Pass.", config);
    // 1st Convolution
    if (!CNNetwork.convolve(CNNetwork.gridChannels, CNNetwork.filterChannels1, CNNetwork.convolvedChannels1, 1, 1, CNNetwork.bias1)) return;
    writeToLog("Convolution 1 complete.", config);

    CNNetwork.activateReLUOverChannels(CNNetwork.convolvedChannels1);

    // 2nd Convolution
    if (!CNNetwork.convolve(CNNetwork.convolvedChannels1, CNNetwork.filterChannels2, CNNetwork.convolvedChannels2, 1, 1, CNNetwork.bias2)) return;
    writeToLog("Convolution 2 complete.", config);

    CNNetwork.activateReLUOverChannels(CNNetwork.convolvedChannels2);
    
    // 3rd Convolution
    if (!CNNetwork.convolve(CNNetwork.convolvedChannels2, CNNetwork.filterChannels3, CNNetwork.convolvedChannels3, 1, 2, CNNetwork.bias3)) return;
    writeToLog("Convolution 3 complete.", config);
    
    CNNetwork.activateReLUOverChannels(CNNetwork.convolvedChannels3);

    CNNetwork.pool(CNNetwork.convolvedChannels3, CNNetwork.pooledChannels, 2, 2, 2, 2);

    // Final Convolution
    if (!CNNetwork.convolve(CNNetwork.pooledChannels, CNNetwork.outputFilterChannels, CNNetwork.outputChannel, 1, 0, CNNetwork.finalBias)) return;
    writeToLog("Final convolution complete.", config);
    printMeanActivation(CNNetwork.outputChannel, "Final CONV");
    CNNetwork.activateSigmoidOverChannels(CNNetwork.outputChannel);

    CNNetwork.upsample(CNNetwork.outputChannel[0], CNNetwork.finalUpsampledGrid);
    

    // Print mean of activations (output)
    printMeanActivation(CNNetwork.convolvedChannels1, "Conv1");
    printMeanActivation(CNNetwork.convolvedChannels2, "Conv2");
    printMeanActivation(CNNetwork.convolvedChannels3, "Conv3");
    printMeanActivation(CNNetwork.pooledChannels, "Pooling");
    printMeanActivation(CNNetwork.outputChannel, "output after sigmoid");
    printMeanActivation({CNNetwork.finalUpsampledGrid}, "after upsample");

    writeToEpochs("Sample " + fileNum + ". LOSS: " + to_string(CNNetwork.compLoss(CNNetwork.finalUpsampledGrid, CNNetwork.groundTruthGrid, 0.01f, CNNetwork.gradientOfLoss)), config);
    
    // Backpropagation //
    writeToLog(fileNum + " Backward Pass.", config);
    
    CNNetwork.backwardUpsample(CNNetwork.gradientOfLoss, CNNetwork.tlCaches, CNNetwork.upsampleInputGrad);
    CNNetwork.backwardSigmoid(CNNetwork.upsampleInputGrad, CNNetwork.finalSigmoidInputGrad);
    if (!CNNetwork.backwardConvolve({ CNNetwork.finalSigmoidInputGrad }, CNNetwork.outputFilterChannels, CNNetwork.finalConvolveInputGrad, CNNetwork.finalBias, CNNetwork.pooledChannels, learningRate, 0)) return;
    CNNetwork.backwardPool(CNNetwork.finalConvolveInputGrad, CNNetwork.convolvedChannels3, CNNetwork.pooledChannels, CNNetwork.finalPoolInputGrad, 2, 2, 2, 2);
    CNNetwork.backwardReLU(CNNetwork.finalPoolInputGrad, CNNetwork.convolvedChannels3, CNNetwork.thirdReLUInputGrad);
    if (!CNNetwork.backwardConvolve(CNNetwork.thirdReLUInputGrad, CNNetwork.filterChannels3, CNNetwork.thirdConvolveInputGrad, CNNetwork.bias3, CNNetwork.convolvedChannels2, learningRate, 2)) return;
    CNNetwork.backwardReLU(CNNetwork.thirdConvolveInputGrad, CNNetwork.convolvedChannels2, CNNetwork.secondReLUInputGrad);
    if (!CNNetwork.backwardConvolve(CNNetwork.secondReLUInputGrad, CNNetwork.filterChannels2, CNNetwork.secondConvolveInputGrad, CNNetwork.bias2, CNNetwork.convolvedChannels1, learningRate, 1)) return;
    CNNetwork.backwardReLU(CNNetwork.secondConvolveInputGrad, CNNetwork.convolvedChannels1, CNNetwork.firstReLUInputGrad);
    if (!CNNetwork.backwardConvolve(CNNetwork.firstReLUInputGrad, CNNetwork.filterChannels1, CNNetwork.firstConvolveInputGrad, CNNetwork.bias1, CNNetwork.gridChannels, learningRate, 1)) return;
    writeToLog("Backward pass complete.", config);

    // Print mean of gradients
    printMeanGradSum({ CNNetwork.gradientOfLoss }, "Final output");
    printMeanGradSum({ CNNetwork.upsampleInputGrad }, "Upsample");
    printMeanGradSum({ CNNetwork.finalSigmoidInputGrad }, "Sigmoid");
    printMeanGradSum(CNNetwork.finalConvolveInputGrad, "Final Convolution");
    printMeanGradSum({ CNNetwork.finalPoolInputGrad }, "Pooling");
    printMeanGradSum(CNNetwork.thirdConvolveInputGrad, "Conv3");
    printMeanGradSum(CNNetwork.secondConvolveInputGrad, "Conv2");
    printMeanGradSum(CNNetwork.firstConvolveInputGrad, "Conv1");
    
    // Print mean of filters
    printMeanGradSum(CNNetwork.outputFilterChannels[0], "Final Convolution filters after update");
    printMeanGradSum(CNNetwork.filterChannels1[0], "Conv1 filters after update");
    printMeanGradSum(CNNetwork.filterChannels2[0], "Conv2 filters after update");
    printMeanGradSum(CNNetwork.filterChannels3[0], "Conv3 filters after update");
    //auto end = chrono::high_resolution_clock::now();
    //cout << "Time elapsed: " << (end-start).count();
}

static void test(CNN& CNNetwork, string flair, string adc, string dwi, unordered_map<string,string>& config) {
   // Open ADC file
    if (!CNNetwork.readNifti(adc, false)) {
        cerr << "Error: Failed to read ADC. ";
        return;
    }
    CNNetwork.clear();
    // Open DWI file
    if (!CNNetwork.readNifti(dwi, false)) {
        cerr << "Error: Failed to read DWI. ";
        return;
    }

    //Open FLAIR file
    if (!CNNetwork.readNifti(flair, true)) {
        cerr << "Error: Failed to read FLAIR. ";
        return;
    }
  
    printImages(CNNetwork.gridChannels);
    
    // 1st Convolution
    if (!CNNetwork.convolve(CNNetwork.gridChannels, CNNetwork.filterChannels1, CNNetwork.convolvedChannels1, 1, 1, CNNetwork.bias1)) return;
    CNNetwork.activateReLUOverChannels(CNNetwork.convolvedChannels1);

    printImages(CNNetwork.convolvedChannels1);

    // 2nd Convolution
    if (!CNNetwork.convolve(CNNetwork.convolvedChannels1, CNNetwork.filterChannels2, CNNetwork.convolvedChannels2, 1, 1, CNNetwork.bias2)) return;
    CNNetwork.activateReLUOverChannels(CNNetwork.convolvedChannels2);
    
    printImages(CNNetwork.convolvedChannels2);

    // 3rd Convolution
    if (!CNNetwork.convolve(CNNetwork.convolvedChannels2, CNNetwork.filterChannels3, CNNetwork.convolvedChannels3, 1, 2, CNNetwork.bias3)) return;
    CNNetwork.activateReLUOverChannels(CNNetwork.convolvedChannels3);

    printImages(CNNetwork.convolvedChannels3);

    CNNetwork.pool(CNNetwork.convolvedChannels3, CNNetwork.pooledChannels, 2, 2, 2, 2);

    printImages(CNNetwork.pooledChannels);

    if (!CNNetwork.convolve(CNNetwork.pooledChannels, CNNetwork.outputFilterChannels, CNNetwork.outputChannel, 1, 0, CNNetwork.finalBias)) return;
    CNNetwork.activateSigmoidOverChannels(CNNetwork.outputChannel);

    printImages(CNNetwork.outputChannel);



    CNNetwork.upsample(CNNetwork.outputChannel[0], CNNetwork.finalUpsampledGrid);
    printImages({ CNNetwork.finalUpsampledGrid });

    // Binary segmentation
    CNNetwork.binarySegmentation(CNNetwork.finalUpsampledGrid, CNNetwork.finalBinaryGrid);

    // Last boolean signify if file should be binary or not
    // tensor to nifti file
    CNNetwork.gridToNifti(CNNetwork.gridChannels[0], config["ADC_OUTPUT_PATH"], false);
    CNNetwork.gridToNifti(CNNetwork.gridChannels[1], config["DWI_OUTPUT_PATH"], false);
    CNNetwork.gridToNifti(CNNetwork.gridChannels[2], config["FLAIR_OUTPUT_PATH"], false);
    CNNetwork.gridToNifti(CNNetwork.finalUpsampledGrid, config["LOGIT_OUTPUT_PATH"], false);
    CNNetwork.gridToNifti(CNNetwork.finalBinaryGrid, config["BINARY_OUTPUT_PATH"], true);
}

static void writeSave(const filterContainer& fc, const vector<vector<float>>& bc, const int iterator, unordered_map<string, string>& config) {
    ofstream FilterSaveFile(config["FILTER_SAVE_PATH"]);
    if (!FilterSaveFile) {
        cerr << "Error: Couldn't open filter save file." << endl;
        return;
    }

    int num = fc.size();

    FilterSaveFile << num << "\n"; // save dimension for reading

    for (int n = 0; n < num; n++) {
        int oc = fc[n].size();
        FilterSaveFile << oc << "\n";
        for (int a = 0; a < oc; a++) {
            int ic = fc[n][a].size();
            FilterSaveFile << ic << "\n";

            for (int b = 0; b < ic; b++) {
                int z = fc[n][a][b].size();
                FilterSaveFile << z << "\n";
                for (int c = 0; c < z; c++) {
                    int y = fc[n][a][b][c].size();
                    FilterSaveFile << y << "\n";
                    for (int d = 0; d < y; d++) {
                        int x = fc[n][a][b][c][d].size();
                        FilterSaveFile << x << "\n";
                        for (int e = 0; e < x; e++) {
                            FilterSaveFile << fc[n][a][b][c][d][e];

                            if (e < x - 1) FilterSaveFile << " "; // save values and separate with spaces
                        }
                        FilterSaveFile << "\n";
                    }
                }
            }
        }
    }
    FilterSaveFile.close();

    ofstream BiasSaveFile(config["BIAS_SAVE_PATH"]);
    if (!BiasSaveFile) {
        cerr << "Error: Couldn't open bias save file." << endl;
        return;
    }

    // Similar to filter writing
    for (int i = 0; i < num; ++i) {
        int oc = bc[i].size();
        BiasSaveFile << oc << "\n";
        for (int j = 0; j < oc; ++j) {
            BiasSaveFile << bc[i][j];
            if (j < oc - 1) BiasSaveFile << " ";

        }
        BiasSaveFile << "\n";
    }
    BiasSaveFile << iterator; // also save iterator on bias file because waste of file if I make a new one just for this right
    BiasSaveFile.close();

}

static void readSave(filterContainer& fc, vector<vector<float>>& bc, int& iterator, unordered_map<string, string>& config) {
    ifstream FilterSaveFile(config["FILTER_SAVE_PATH"]);
    if (!FilterSaveFile) {
        cerr << "Error: Couldn't open filter save file." << endl;
        return;
    }
    int num;
    // Read values
    FilterSaveFile >> num;
    fc.resize(num);
    for (int n = 0; n < num; n++) {
        int oc;
        FilterSaveFile >> oc;
        fc[n].resize(oc);
        for (int a = 0; a < oc; a++) {
            int ic;
            FilterSaveFile >> ic;
            fc[n][a].resize(ic);
            for (int b = 0; b < ic; b++) {
                int z;
                FilterSaveFile >> z;
                fc[n][a][b].resize(z);
                for (int c = 0; c < z; c++) {
                    int y;
                    FilterSaveFile >> y;
                    fc[n][a][b][c].resize(y);
                    for (int d = 0; d < y; d++) {
                        int x;
                        FilterSaveFile >> x;
                        fc[n][a][b][c][d].resize(x);
                        for (int e = 0; e < x; e++) {
                            FilterSaveFile >> fc[n][a][b][c][d][e];
                        }
                    }
                }
            }
        }
    }
    FilterSaveFile.close();

    ifstream BiasSaveFile(config["BIAS_SAVE_PATH"]);
    if (!BiasSaveFile) {
        cerr << "Error: Couldn't open bias save file." << endl;
        return;
    }

    bc.resize(num);

    for (int i = 0; i < num; ++i) {
        int oc;
        BiasSaveFile >> oc;
        bc[i].resize(oc);
        for (int j = 0; j < oc; ++j) {
            BiasSaveFile >> bc[i][j]; // read from txt file and insert in bias
        }

    }
    BiasSaveFile >> iterator; // iterator of samples also read at end of file
    BiasSaveFile.close();

}

int main() {
    // Read config text file
    unordered_map<string, string> config;
    ifstream file("config.txt");
    string key, value;
    if (!file) {
        cerr << "Error: Could not open config file.\n";
    }

    // Read key then value for all configs
    while (file >> key >> value) {
        config[key] = value;
    }

    string dataPath = config["EXTRACTED_DATA_PATH"];
    string logPath = config["LOG_PATH"];
    string epochPath = config["EPOCH_LOG_PATH"];
    string filterPath = config["FILTER_SAVE_PATH"];
    string biasPath = config["BIAS_SAVE_PATH"];

    cout << "==================IschemiaNet by Yohan Lee==================\n";

    cout << "TIP In training, lower loss and greater intersection is better.\n";
    
    int iterator = 0;

    unique_ptr<CNN> CNNetwork = make_unique<CNN>();
    string answer;
    int channelNum1 = 16;
    int channelNum2 = 24;
    int channelNum3 = 32;

    float finalBias = -1.5f;
    


    while (true) {
        cout << "Load saved network? [y][n] ";
        cin >> answer;
        if (answer.empty()) {
            continue;  // Re-prompt the user
        }


        if (answer == "y") {
            cout << "Loading...";
            filterContainer fc;
            vector<vector<float>> bc;
            readSave(fc, bc, iterator, config); // read from txt files

            cout << "\nLast load saved at Sample " << iterator;
            CNNetwork->filterChannels1 = fc[0];
            CNNetwork->filterChannels2 = fc[1];
            CNNetwork->filterChannels3 = fc[2];
            CNNetwork->outputFilterChannels = fc[3];

            CNNetwork->bias1 = bc[0];
            CNNetwork->bias2 = bc[1];
            CNNetwork->bias3 = bc[2];
            CNNetwork->finalBias = bc[3];

            cout << "\nLoad Complete.";
            break;
        }
        else if (answer == "n") {

            cout << "\n===================NETWORK INITIALISATION===================\n";
                       
            cout << "\nWARNING! If testing with a new network, the test will not be accurate!\n";

            cout << "Number of Channels for 1st convolutional layer: \n";
            cin >> channelNum1;
           
            cout << "Number of Channels for 2nd convolutional layer: \n";
            cin >> channelNum2;
            cout << "Number of Channels for 3rd convolutional layer: \n";
            cin >> channelNum3;
            cout << "Final Bias: ";
            cin >> finalBias;

            if (channelNum1 <= 0 || channelNum2 <= 0 || channelNum3 <= 0) {
                cerr << "Error: Number of channels must be positive integers!\n";
            }
            
            // Replace paths
            ofstream file(epochPath);
            ofstream file2(logPath);
            ofstream file3(biasPath);
            ofstream file4(filterPath); // clear

            
            CNNetwork->initialiseFilters(CNNetwork->filterChannels1, channelNum1, 3, 3, 3, 3, true);
            CNNetwork->initialiseFilters(CNNetwork->filterChannels2, channelNum2, channelNum1, 3, 3, 3, true);
            CNNetwork->initialiseFilters(CNNetwork->filterChannels3, channelNum3, channelNum2, 5, 5, 5, true);
            // 1x1x1 to convert 8 diff probabilities to a single prediction
            CNNetwork->initialiseFilters(CNNetwork->outputFilterChannels, 1, channelNum3, 1, 1, 1, false); 


            CNNetwork->bias1.resize(channelNum1, 0.01f);
            CNNetwork->bias2.resize(channelNum2, 0.01f);
            CNNetwork->bias3.resize(channelNum3, 0.01f);
            CNNetwork->finalBias.resize(1, finalBias); 
            break;
        }
        else {
            cout << "\nInvalid request\n";
        }
    }

    while (true) {
        cout << "\n=====================TESTING & TRAINING=====================\n";
                   
        cout << "Test or Train using network? [test][train] ";
        cin >> answer;

        if (answer == "test") {
            string testFileFlairPath;
            string testFileADCPath;
            string testFileDWIPath;

            cout << "Input test FLAIR file path: ";
            cin >> testFileFlairPath;
            cout << "Input test ADC file path: ";
            cin >> testFileADCPath;
            cout << "Input test DWI file path: ";
            cin >> testFileDWIPath;
            cout << "Testing...";
            test(*CNNetwork, testFileFlairPath, testFileADCPath, testFileDWIPath, config);
            cout << "\nTesting Complete!\nReturning to menu...\n";
        }
        else if (answer == "train") {
            cout << "Learning rate: ";
            float learningRate = 0.000001f;
            cin >> learningRate;
            cout << "Number of Epochs: ";
            int num = 5;
            cin >> num;
            
            iterator++;
            for (int epoch = 0; epoch < num; ++epoch) {
                for (iterator; iterator <= 5; ++iterator) {
                    cout << "Training on sample " << iterator << "...\n";
                    if (iterator >= 100) {
                        train("0" + to_string(iterator), *CNNetwork, learningRate, config);
                    }
                    else if (iterator >= 10) {
                        train("00" + to_string(iterator), *CNNetwork, learningRate, config);
                    }
                    else {
                        train("000" + to_string(iterator), *CNNetwork, learningRate, config);
                    }
                    // Save filters
                    filterCache fs1 = CNNetwork->filterChannels1; 
                    filterCache fs2 = CNNetwork->filterChannels2;
                    filterCache fs3 = CNNetwork->filterChannels3;
                    filterCache fs4 = CNNetwork->outputFilterChannels;

                    filterContainer fc = { fs1, fs2, fs3, fs4 };
                    // Save biases
                    vector<float> b1 = CNNetwork->bias1;
                    vector<float> b2 = CNNetwork->bias2;
                    vector<float> b3 = CNNetwork->bias3;
                    vector<float> b4 = CNNetwork->finalBias;

                    vector<vector<float>> bc = { b1, b2, b3, b4 }; 

                    writeSave(fc, bc, iterator, config);// save filters and biases into txt files
                    // Create new network to reset everything
                    CNNetwork = make_unique<CNN>();

                    // Import saved filters and biases
                    CNNetwork->filterChannels1 = fs1;
                    CNNetwork->filterChannels2 = fs2;
                    CNNetwork->filterChannels3 = fs3;
                    CNNetwork->outputFilterChannels = fs4;

                    CNNetwork->bias1 = b1;
                    CNNetwork->bias2 = b2;
                    CNNetwork->bias3 = b3;
                    CNNetwork->finalBias = b4;
                    
                    cout << "\nFinished training on Sample " << iterator << ".\n";
                }
                cout << "Epoch " << epoch+1 << " complete.\n";
                iterator = 1;
            }
            cout << "\nTraining Complete!\nReturning to menu...\n";
            iterator = 0;
        }
        else {
            cout << "\nInvalid request";
        }
    }
}