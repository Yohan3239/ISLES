// ISLES.cpp : Defines the entry point for the application.

#include <iostream>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include "framework.h"
#include "ISLES.h"
#include "CNN.h"
#include <cstdio>
#include <windows.h>
#include <commdlg.h>
#include <vector>
#include <string>



#define MAX_LOADSTRING 100


using namespace C;
using namespace std;


// Global Variables:
HINSTANCE hInst;                                // current instance
WCHAR szTitle[MAX_LOADSTRING];                  // The title bar text
WCHAR szWindowClass[MAX_LOADSTRING];            // the main window class name

// Forward declarations of functions included in this code module:
ATOM                MyRegisterClass(HINSTANCE hInstance);
BOOL                InitInstance(HINSTANCE, int);
LRESULT CALLBACK    WndProc(HWND, UINT, WPARAM, LPARAM);
INT_PTR CALLBACK    About(HWND, UINT, WPARAM, LPARAM);

static void train(string fileNum, CNN& CNNetwork, float learningRate) {
    // file num has 0 in front remember
    string filename_FLAIR = "C:\\Users\\yohan\\Documents\\ISLES-2022\\Extracted\\sub-strokecase" + fileNum + "_ses-0001_FLAIR.nii";
    string filename_ADC = "C:\\Users\\yohan\\Documents\\ISLES-2022\\Extracted\\sub-strokecase" + fileNum + "_ses-0001_adc.nii";
    string filename_DWI = "C:\\Users\\yohan\\Documents\\ISLES-2022\\Extracted\\sub-strokecase" + fileNum + "_ses-0001_dwi.nii";
    string filename_MSK = "C:\\Users\\yohan\\Documents\\ISLES-2022\\Extracted\\sub-strokecase" + fileNum + "_ses-0001_msk.nii";


    // Initialise CNNetwork object from CNN class

    // WILL ADD FUNCTIONALITY TO ITERATE OVER ALL 200 TEST FILES i.e. 800 total, inc. mask
    // Open ADC file


    writeToLog(fileNum + " Input Preprocessing.");
    //writeToLog("Trying to open ADC file at: " + filename_ADC);
    CNNetwork.readNifti(filename_ADC, false);
    //writeToLog("Completed reading ADC.");
     // Clear to just set DWI file as resample target as ADC/DWI are same anyways

    // Open DWI file
    //writeToLog("Trying to open DWI file at: " + filename_DWI);
    CNNetwork.readNifti(filename_DWI, false);
    //writeToLog("Completed reading DWI.");

    //Open FLAIR file
    //writeToLog("Trying to open FLAIR file at: " + filename_FLAIR);
    CNNetwork.readNifti(filename_FLAIR, true);
    //writeToLog("Completed reading FLAIR.");
    //writeToLog("All channels ready for convolution. Initialising filter.");
    CNNetwork.readNifti(filename_MSK, false);
    // Initialising filters, 8 outputs, 3 inputs (ADC/DWI/FLAIR), 3x3x3 filter


    writeToLog(fileNum + " Forward Pass.");
    // conv 1
    CNNetwork.convolve(CNNetwork.gridChannels, CNNetwork.filterChannels1, CNNetwork.convolvedChannels1, 1, 1, CNNetwork.bias1);
    writeToLog("Convolution 1 complete.");
    //writeToLog("Applying ReLU function");
    CNNetwork.activateReLUOverChannels(CNNetwork.convolvedChannels1);
    //writeToLog("Applied ReLU function.");


    // conv 2
    CNNetwork.convolve(CNNetwork.convolvedChannels1, CNNetwork.filterChannels2, CNNetwork.convolvedChannels2, 1, 1, CNNetwork.bias2);
    writeToLog("Convolution 2 complete.");
    //writeToLog("Applying ReLU function");
    CNNetwork.activateReLUOverChannels(CNNetwork.convolvedChannels2);
    //writeToLog("Applied ReLU function.");
    // conv 3
    CNNetwork.convolve(CNNetwork.convolvedChannels2, CNNetwork.filterChannels3, CNNetwork.convolvedChannels3, 1, 1, CNNetwork.bias3);
    writeToLog("Convolution 3 complete.");
    //writeToLog("Applying ReLU function");
    CNNetwork.activateReLUOverChannels(CNNetwork.convolvedChannels3);
    //writeToLog("Applied ReLU function.");
    //writeToLog("Pooling.");
    CNNetwork.pool(CNNetwork.convolvedChannels3, CNNetwork.pooledChannels, 2, 2, 2, 2);
    //writeToLog("Pooling complete.");

    // only need 1 output channel
    //writeToLog("Initialising output filter.");
    
    //writeToLog("Initialised output filter.");
    //writeToLog("Final convolution.");

    CNNetwork.convolve(CNNetwork.pooledChannels, CNNetwork.outputFilterChannels, CNNetwork.outputChannel, 1, 1, CNNetwork.finalBias);
    //writeToLog("Applying sigmoid function.");
    CNNetwork.activateSigmoidOverChannels(CNNetwork.outputChannel);
    //writeToLog("Applied sigmoid function.");
    writeToLog("Final convolution complete.");

    CNNetwork.upsample(CNNetwork.outputChannel[0], CNNetwork.finalUpsampledGrid);
    //writeToLog("Upsampled");




    CNNetwork.binarySegmentation(CNNetwork.outputChannel[0], CNNetwork.finalBinaryGrid);

  
    
    
    
    //writeToLog("Completed binary segmentation");
    //writeToLog(to_string(CNNetwork.finalUpsampledGrid.size()));
    //writeToLog(to_string(CNNetwork.finalUpsampledGrid[0].size()));
    //writeToLog(to_string(CNNetwork.finalUpsampledGrid[0][0].size()));
    //writeToLog(to_string(CNNetwork.groundTruthGrid.size()));
    //writeToLog(to_string(CNNetwork.groundTruthGrid[0].size()));
    //writeToLog(to_string(CNNetwork.groundTruthGrid[0][0].size()));
    writeToEpochs("Sample " + fileNum + ". LOSS: " + to_string(CNNetwork.crossEntropyLoss(CNNetwork.finalUpsampledGrid, CNNetwork.groundTruthGrid)));
    // backprop
    writeToLog(fileNum + " Backward Pass.");
    //writeToLog("Backward Upsampling...");
    CNNetwork.backwardUpsample(CNNetwork.gradientOfLoss, CNNetwork.tlCaches, CNNetwork.upsampleInputGrad);
    //writeToLog("Backward Sigmoiding...");
    CNNetwork.backwardSigmoid(CNNetwork.upsampleInputGrad, CNNetwork.finalSigmoidInputGrad);
   writeToLog("Backward Convolving...");

    CNNetwork.backwardConvolve({ CNNetwork.finalSigmoidInputGrad }, CNNetwork.outputFilterChannels, CNNetwork.finalConvolveInputGrad, CNNetwork.finalBias, CNNetwork.pooledChannels, learningRate);
    //writeToLog("Backward Pooling...");
    CNNetwork.backwardPool(CNNetwork.finalConvolveInputGrad, CNNetwork.convolvedChannels3, CNNetwork.pooledChannels, CNNetwork.finalPoolInputGrad, 2, 2, 2, 2);

    //writeToLog("Backward ReLUing...");
    CNNetwork.backwardReLU(CNNetwork.finalPoolInputGrad, CNNetwork.convolvedChannels3, CNNetwork.thirdReLUInputGrad);


    writeToLog("Backward Convolving...");
    CNNetwork.backwardConvolve(CNNetwork.thirdReLUInputGrad, CNNetwork.filterChannels3, CNNetwork.thirdConvolveInputGrad, CNNetwork.bias3, CNNetwork.convolvedChannels2, learningRate);

    //writeToLog("Backward ReLUing...");
    CNNetwork.backwardReLU(CNNetwork.thirdConvolveInputGrad, CNNetwork.convolvedChannels2, CNNetwork.secondReLUInputGrad);
    writeToLog("Backward Convolving...");

    CNNetwork.backwardConvolve(CNNetwork.secondReLUInputGrad, CNNetwork.filterChannels2, CNNetwork.secondConvolveInputGrad, CNNetwork.bias2, CNNetwork.convolvedChannels1, learningRate);
    //writeToLog("Backward ReLUing...");
    CNNetwork.backwardReLU(CNNetwork.secondConvolveInputGrad, CNNetwork.convolvedChannels1, CNNetwork.firstReLUInputGrad);
    writeToLog("Backward Convolving...");
    CNNetwork.backwardConvolve(CNNetwork.firstReLUInputGrad, CNNetwork.filterChannels1, CNNetwork.firstConvolveInputGrad, CNNetwork.bias1, CNNetwork.gridChannels, learningRate);
    
    CNNetwork.gradSum({ CNNetwork.gradientOfLoss });
    CNNetwork.gradSum({ CNNetwork.upsampleInputGrad });
    CNNetwork.gradSum({ CNNetwork.finalSigmoidInputGrad });
    CNNetwork.gradSum({ CNNetwork.finalPoolInputGrad });
    

    CNNetwork.gradSum(CNNetwork.finalConvolveInputGrad);
    CNNetwork.gradSum(CNNetwork.thirdConvolveInputGrad);
    CNNetwork.gradSum(CNNetwork.secondConvolveInputGrad);
    CNNetwork.gradSum(CNNetwork.firstConvolveInputGrad);
    
    


}

int APIENTRY wWinMain(_In_ HINSTANCE hInstance,






    _In_opt_ HINSTANCE hPrevInstance,
    _In_ LPWSTR lpCmdLine,
    _In_ int nCmdShow)
{
    UNREFERENCED_PARAMETER(hPrevInstance);
    UNREFERENCED_PARAMETER(lpCmdLine);
    // CODE //


    ofstream logFile("C:\\Users\\yohan\\source\\repos\\ISLES\\log.txt", std::ios::trunc);
    ofstream epochFile("C:\\Users\\yohan\\source\\repos\\ISLES\\Epochs.txt", std::ios::trunc);

    std::unique_ptr<CNN> CNNetwork = std::make_unique<CNN>();

    float learningRate = 0.000001f;

    CNNetwork->initialiseFilters(CNNetwork->filterChannels1, 8, 3, 3, 3, 3, true);
    CNNetwork->initialiseFilters(CNNetwork->filterChannels2, 16, 8, 3, 3, 3, true);
    CNNetwork->initialiseFilters(CNNetwork->filterChannels3, 32, 16, 3, 3, 3, true);
    CNNetwork->initialiseFilters(CNNetwork->outputFilterChannels, 1, 32, 1, 1, 1, false);

    CNNetwork->bias1.resize(8, 0.0f);
    CNNetwork->bias2.resize(16, 0.0f);
    CNNetwork->bias3.resize(32, 0.0f);
    CNNetwork->finalBias.resize(1, 0.0f);

    for (int i = 1; i <= 100; ++i) {
        if (i != 46 && i != 61 && i != 9 && i != 13 && i != 26 && i != 38) {
            if (i >= 100) {
                train("0" + to_string(i), *CNNetwork, learningRate);
            }
            else if (i >= 10) {
                train("00" + to_string(i), *CNNetwork, learningRate);
            }
            else {
                train("000" + to_string(i), *CNNetwork, learningRate);
            }
            
            
            
            vector<vector<vector<vector<vector<float>>>>> fs1 = CNNetwork->filterChannels1;
            vector<vector<vector<vector<vector<float>>>>> fs2 = CNNetwork->filterChannels2;
            vector<vector<vector<vector<vector<float>>>>> fs3 = CNNetwork->filterChannels3;
            vector<vector<vector<vector<vector<float>>>>> fs4 = CNNetwork->outputFilterChannels;

            vector<float> b1 = CNNetwork->bias1;
            vector<float> b2 = CNNetwork->bias2;
            vector<float> b3 = CNNetwork->bias3;
            vector<float> b4 = CNNetwork->finalBias;

            CNNetwork = std::make_unique<CNN>();

            CNNetwork->filterChannels1 = fs1;
            CNNetwork->filterChannels2 = fs2;
            CNNetwork->filterChannels3 = fs3;
            CNNetwork->outputFilterChannels = fs4;

            CNNetwork->bias1 = b1;
            CNNetwork->bias2 = b2;
            CNNetwork->bias3 = b3;
            CNNetwork->finalBias = b4;
        }    
    }

    //CNN CNN2;
    //CNN2.filterChannels1 = CNNetwork.filterChannels1;
    //CNN2.filterChannels2 = CNNetwork.filterChannels2;
    //CNN2.filterChannels3 = CNNetwork.filterChannels3;
    //CNN2.outputFilterChannels = CNNetwork.outputFilterChannels;

    //CNN2.bias1 = CNNetwork.bias1;
    //CNN2.bias2 = CNNetwork.bias2;
    //CNN2.bias3 = CNNetwork.bias3;
    //CNN2.finalBias = CNNetwork.finalBias;

    //train("0002", CNN2, learningRate);


   
    //CODE END//
    
    // Initialize global strings
    LoadStringW(hInstance, IDS_APP_TITLE, szTitle, MAX_LOADSTRING);
    LoadStringW(hInstance, IDC_ISLES, szWindowClass, MAX_LOADSTRING);
    MyRegisterClass(hInstance);

    // Perform application initialization:
    if (!InitInstance(hInstance, nCmdShow)) {
        return FALSE;
    }

    HACCEL hAccelTable = LoadAccelerators(hInstance, MAKEINTRESOURCE(IDC_ISLES));

    MSG msg;

    // Main message loop
    while (GetMessage(&msg, nullptr, 0, 0)) {
        if (!TranslateAccelerator(msg.hwnd, hAccelTable, &msg)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
    }

    return (int)msg.wParam;
}




//
//  FUNCTION: MyRegisterClass()
//
//  PURPOSE: Registers the window class.
//
ATOM MyRegisterClass(HINSTANCE hInstance)
{
    WNDCLASSEXW wcex;

    wcex.cbSize = sizeof(WNDCLASSEX);

    wcex.style          = CS_HREDRAW | CS_VREDRAW;
    wcex.lpfnWndProc    = WndProc;
    wcex.cbClsExtra     = 0;
    wcex.cbWndExtra     = 0;
    wcex.hInstance      = hInstance;
    wcex.hIcon          = LoadIcon(hInstance, MAKEINTRESOURCE(IDI_ISLES));
    wcex.hCursor        = LoadCursor(nullptr, IDC_ARROW);
    wcex.hbrBackground  = (HBRUSH)(COLOR_WINDOW+1);
    wcex.lpszMenuName   = MAKEINTRESOURCEW(IDC_ISLES);
    wcex.lpszClassName  = szWindowClass;
    wcex.hIconSm        = LoadIcon(wcex.hInstance, MAKEINTRESOURCE(IDI_SMALL));

    return RegisterClassExW(&wcex);
}

//
//   FUNCTION: InitInstance(HINSTANCE, int)
//
//   PURPOSE: Saves instance handle and creates main window
//
//   COMMENTS:
//
//        In this function, we save the instance handle in a global variable and
//        create and display the main program window.
//
BOOL InitInstance(HINSTANCE hInstance, int nCmdShow)
{
   hInst = hInstance; // Store instance handle in our global variable

   HWND hWnd = CreateWindowW(szWindowClass, szTitle, WS_OVERLAPPEDWINDOW,
      CW_USEDEFAULT, 0, CW_USEDEFAULT, 0, nullptr, nullptr, hInstance, nullptr);

   if (!hWnd)
   {
      return FALSE;
   }

   ShowWindow(hWnd, nCmdShow);
   UpdateWindow(hWnd);

   return TRUE;
}

//
//  FUNCTION: WndProc(HWND, UINT, WPARAM, LPARAM)
//
//  PURPOSE: Processes messages for the main window.
//
//  WM_COMMAND  - process the application menu
//  WM_PAINT    - Paint the main window
//  WM_DESTROY  - post a quit message and return
//
//
LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    switch (message)
    {
    case WM_CREATE:
        {
        CreateWindow(L"BUTTON", L"Upload NIfTI File", WS_TABSTOP | WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON,
            10, 10, 200, 30, hWnd, (HMENU)1, GetModuleHandle(NULL), NULL);
        break;  
        }
    case WM_COMMAND:
        {
            int wmId = LOWORD(wParam);
            // Parse the menu selections:
            switch (wmId)
            {
            case IDM_ABOUT:
                DialogBox(hInst, MAKEINTRESOURCE(IDD_ABOUTBOX), hWnd, About);
                break;
            case IDM_EXIT:
                DestroyWindow(hWnd);
                break;
            default:
                return DefWindowProc(hWnd, message, wParam, lParam);
            }
        }
        break;
    case WM_PAINT:
        {
            PAINTSTRUCT ps;
            HDC hdc = BeginPaint(hWnd, &ps);
            // TODO: Add any drawing code that uses hdc here...
            EndPaint(hWnd, &ps);
        }
        break;
    case WM_DESTROY:
        PostQuitMessage(0);
        break;
    default:
        return DefWindowProc(hWnd, message, wParam, lParam);
    }
    return 0;
}

// Message handler for about box.
INT_PTR CALLBACK About(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam)
{
    UNREFERENCED_PARAMETER(lParam);
    switch (message)
    {
        case WM_INITDIALOG:
            return (INT_PTR)TRUE;

        case WM_COMMAND:
            if (LOWORD(wParam) == IDOK || LOWORD(wParam) == IDCANCEL)
            {
                EndDialog(hDlg, LOWORD(wParam));
                return (INT_PTR)TRUE;
            }
            break;
    }
    return (INT_PTR)FALSE;
}

