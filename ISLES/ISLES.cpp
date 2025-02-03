// ISLES.cpp : Defines the entry point for the application.

#include <iostream>
#include <filesystem>
#include <fstream>
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


int APIENTRY wWinMain(_In_ HINSTANCE hInstance,

    




    _In_opt_ HINSTANCE hPrevInstance,
    _In_ LPWSTR lpCmdLine,
    _In_ int nCmdShow)
{
    UNREFERENCED_PARAMETER(hPrevInstance);
    UNREFERENCED_PARAMETER(lpCmdLine);
    // CODE //


    ofstream logFile("C:\\Users\\yohan\\source\\repos\\ISLES\\log.txt", std::ios::trunc);

    string filename_FLAIR = "C:\\Users\\yohan\\Documents\\ISLES-2022\\ISLES-2022\\sub-strokecase0115\\ses-0001\\anat\\sub-strokecase0115_ses-0001_FLAIR.nii\\sub-strokecase0115_ses-0001_FLAIR.nii";
    string filename_ADC = "C:\\Users\\yohan\\Documents\\ISLES-2022\\ISLES-2022\\sub-strokecase0115\\ses-0001\\dwi\\sub-strokecase0115_ses-0001_adc.nii\\sub-Stroke29_iso_adc_skull_stripped.nii";
    string filename_DWI = "C:\\Users\\yohan\\Documents\\ISLES-2022\\ISLES-2022\\sub-strokecase0115\\ses-0001\\dwi\\sub-strokecase0115_ses-0001_dwi.nii\\sub-Stroke29_iso_dwi_skull_stripped.nii";
    string filename_MSK = "C:\\Users\\yohan\\Documents\\ISLES-2022\\ISLES-2022\\derivatives\\sub-strokecase0115\\ses-0001\\sub-strokecase0115_ses-0001_msk.nii\\sub-strokecase0115_ses-0001_msk.nii";

    
    // Initialise CNNetwork object from CNN class
    CNN CNNetwork;
    // WILL ADD FUNCTIONALITY TO ITERATE OVER ALL 200 TEST FILES i.e. 800 total, inc. mask
    // Open ADC file
    writeToLog("Trying to open ADC file at: " + filename_ADC);
    CNNetwork.readNifti(filename_ADC, false);
    writeToLog("Completed reading ADC.");
    CNNetwork.clear(); // Clear to just set DWI file as resample target as ADC/DWI are same anyways

    // Open DWI file
    writeToLog("Trying to open DWI file at: " + filename_DWI);
    CNNetwork.readNifti(filename_DWI, false);
    writeToLog("Completed reading DWI.");

    //Open FLAIR file
    writeToLog("Trying to open FLAIR file at: " + filename_FLAIR);
    CNNetwork.readNifti(filename_FLAIR, true);
    writeToLog("Completed reading FLAIR.");
    writeToLog("All channels ready for convolution. Initialising filter.");

    // Initialising filters, 8 outputs, 3 inputs (ADC/DWI/FLAIR), 3x3x3 filter
    CNNetwork.initialiseFilters(CNNetwork.filterChannels1, 8, 3, 3, 3, 3, true);
    writeToLog("Filter initialised");
    CNNetwork.initialiseFilters(CNNetwork.filterChannels2, 12, 8, 3, 3, 3, true);
    writeToLog("Filter initialised");
    CNNetwork.initialiseFilters(CNNetwork.filterChannels3, 16, 12, 3, 3, 3, true);
    writeToLog("Filter initialised");
    writeToLog("Convolution");
    
    // conv 1
    CNNetwork.convolve(CNNetwork.gridChannels, CNNetwork.filterChannels1, CNNetwork.convolvedChannels1, 1, 1);
    writeToLog("Convolution 1 complete.");
    writeToLog("Applying ReLU function");
    CNNetwork.activateReLUOverChannels(CNNetwork.convolvedChannels1);
    writeToLog("Applied ReLU function.");


    // conv 2
    CNNetwork.convolve(CNNetwork.convolvedChannels1, CNNetwork.filterChannels2, CNNetwork.convolvedChannels2, 1, 1);
    writeToLog("Convolution 2 complete.");
    writeToLog("Applying ReLU function");
    CNNetwork.activateReLUOverChannels(CNNetwork.convolvedChannels2);
    writeToLog("Applied ReLU function.");
    // conv 3
    CNNetwork.convolve(CNNetwork.convolvedChannels2, CNNetwork.filterChannels3, CNNetwork.convolvedChannels3, 1, 1);
    writeToLog("Convolution 3 complete.");
    writeToLog("Applying ReLU function");
    CNNetwork.activateReLUOverChannels(CNNetwork.convolvedChannels3);
    writeToLog("Applied ReLU function.");
    writeToLog("Pooling.");
    CNNetwork.pool(CNNetwork.convolvedChannels3, CNNetwork.pooledChannels, 2, 2, 2, 2);
    writeToLog("Pooling complete.");

    // only need 1 output channel
    writeToLog("Initialising output filter.");
    CNNetwork.initialiseFilters(CNNetwork.outputFilterChannels, 1, 16, 1, 1, 1, false);
    writeToLog("Initialised output filter.");
    writeToLog("Final convolution.");
    writeToLog(to_string(CNNetwork.pooledChannels.size()) + to_string(CNNetwork.pooledChannels[0].size()) + to_string(CNNetwork.pooledChannels[0][0].size()) + to_string(CNNetwork.pooledChannels[0][0][0].size()));
    writeToLog(to_string(CNNetwork.outputFilterChannels.size()) + to_string(CNNetwork.outputFilterChannels[0].size()) + to_string(CNNetwork.outputFilterChannels[0][0].size()) + to_string(CNNetwork.outputFilterChannels[0][0][0].size()) + to_string(CNNetwork.outputFilterChannels[0][0][0][0].size()));
    
    
    CNNetwork.convolve(CNNetwork.pooledChannels, CNNetwork.outputFilterChannels, CNNetwork.outputChannel, 1, 0);
    writeToLog("Applying sigmoid function.");
    CNNetwork.activateSigmoidOverChannels(CNNetwork.outputChannel);
    writeToLog("Applied sigmoid function.");
    writeToLog("Final convolution complete.");  
    CNNetwork.upsample(CNNetwork.outputChannel[0], CNNetwork.finalUpsampledGrid);

    CNNetwork.binarySegmentation(CNNetwork.outputChannel[0], CNNetwork.finalBinaryGrid);
    
    CNNetwork.readNifti(filename_MSK, false);
    writeToLog("Completed reading ground truth mask.");

    writeToLog(to_string(CNNetwork.crossEntropyLoss(CNNetwork.finalUpsampledGrid, CNNetwork.groundTruthGrid)));
    // backprop
    


    //////////////////////////////
    // TO BE REWRITTEN FOR GUI!!//
    //////////////////////////////
    writeToLog("X for slice where X axis is const.");
    writeToLog("Y");
    writeToLog("Z");

    writeToLog("Enter slice");
    int slice = 28;
    
    for (int k = 0; k < CNNetwork.gridChannels.size(); ++k) {
        writeToLog("Grid" + to_string(k) + ": ");
        endLine();
        for (int i = CNNetwork.gridChannels[k].size() - 1; i >= 0; --i) {
            for (int j = 0; j < CNNetwork.gridChannels[k][0].size(); ++j) {
                if (CNNetwork.gridChannels[k][i][j][slice] > 0.9) {
                    writeToLogNoLine("@");
                }
                else if (CNNetwork.gridChannels[k][i][j][slice] > 0.8) {
                    writeToLogNoLine("%");
                }
                else if (CNNetwork.gridChannels[k][i][j][slice] > 0.7) {
                    writeToLogNoLine("#");
                }
                else if (CNNetwork.gridChannels[k][i][j][slice] > 0.6) {
                    writeToLogNoLine("*");
                }
                else if (CNNetwork.gridChannels[k][i][j][slice] > 0.5) {
                    writeToLogNoLine("+");
                }
                else if (CNNetwork.gridChannels[k][i][j][slice] > 0.4) {
                    writeToLogNoLine("=");
                }
                else if (CNNetwork.gridChannels[k][i][j][slice] > 0.3) {
                    writeToLogNoLine("~");
                }
                else if (CNNetwork.gridChannels[k][i][j][slice] > 0.2) {
                    writeToLogNoLine("-");
                }
                else if (CNNetwork.gridChannels[k][i][j][slice] > 0.1) {
                    writeToLogNoLine(",");
                }
                else if (CNNetwork.gridChannels[k][i][j][slice] > 0) {
                    writeToLogNoLine(".");
                }
                else if (CNNetwork.gridChannels[k][i][j][slice] == 0) {
                    writeToLogNoLine(" ");
                }
            }
            endLine();
        }
    }
    

    for (int k = 0; k < CNNetwork.gridChannels.size(); ++k) {
        writeToLog("Grid" + to_string(k) + ": ");
        endLine();
        for (int i = CNNetwork.gridChannels[k].size() - 1; i >= 0; --i) {
            for (int j = 0; j < CNNetwork.gridChannels[k][0][0].size(); ++j) {
                if (CNNetwork.gridChannels[k][i][slice][j] > 0.9) {
                    writeToLogNoLine("@");
                }
                else if (CNNetwork.gridChannels[k][i][slice][j] > 0.8) {
                    writeToLogNoLine("%");
                }
                else if (CNNetwork.gridChannels[k][i][slice][j] > 0.7) {
                    writeToLogNoLine("#");
                }
                else if (CNNetwork.gridChannels[k][i][slice][j] > 0.6) {
                    writeToLogNoLine("*");
                }
                else if (CNNetwork.gridChannels[k][i][slice][j] > 0.5) {
                    writeToLogNoLine("+");
                }
                else if (CNNetwork.gridChannels[k][i][slice][j] > 0.4) {
                    writeToLogNoLine("=");
                }
                else if (CNNetwork.gridChannels[k][i][slice][j] > 0.3) {
                    writeToLogNoLine("~");
                }
                else if (CNNetwork.gridChannels[k][i][slice][j] > 0.2) {
                    writeToLogNoLine("-");
                }
                else if (CNNetwork.gridChannels[k][i][slice][j] > 0.1) {
                    writeToLogNoLine(",");
                }
                else if (CNNetwork.gridChannels[k][i][slice][j] > 0) {
                    writeToLogNoLine(".");
                }
                else if (CNNetwork.gridChannels[k][i][slice][j] == 0) {
                    writeToLogNoLine(" ");
                }
            }
            endLine();
        }
    }
    

    for (int k = 0; k < CNNetwork.gridChannels.size(); ++k) {
        writeToLog("Grid" + to_string(k) + ": ");
        endLine();
        for (int i = CNNetwork.gridChannels[k][0].size() - 1; i >= 0; --i) {
            for (int j = 0; j < CNNetwork.gridChannels[k][0][0].size(); ++j) {
                if (CNNetwork.gridChannels[k][slice][i][j] > 0.9) {
                    writeToLogNoLine("@");
                }
                else if (CNNetwork.gridChannels[k][slice][i][j] > 0.8) {
                    writeToLogNoLine("%");
                }
                else if (CNNetwork.gridChannels[k][slice][i][j] > 0.7) {
                    writeToLogNoLine("#");
                }
                else if (CNNetwork.gridChannels[k][slice][i][j] > 0.6) {
                    writeToLogNoLine("*");
                }
                else if (CNNetwork.gridChannels[k][slice][i][j] > 0.5) {
                    writeToLogNoLine("+");
                }
                else if (CNNetwork.gridChannels[k][slice][i][j] > 0.4) {
                    writeToLogNoLine("=");
                }
                else if (CNNetwork.gridChannels[k][slice][i][j] > 0.3) {
                    writeToLogNoLine("~");
                }
                else if (CNNetwork.gridChannels[k][slice][i][j] > 0.2) {
                    writeToLogNoLine("-");
                }
                else if (CNNetwork.gridChannels[k][slice][i][j] > 0.1) {
                    writeToLogNoLine(",");
                }
                else if (CNNetwork.gridChannels[k][slice][i][j] > 0) {
                    writeToLogNoLine(".");
                }
                else if (CNNetwork.gridChannels[k][slice][i][j] == 0) {
                    writeToLogNoLine(" ");
                }
            }
            endLine();
        }
    }
    

    

    for (int k = 0; k < CNNetwork.pooledChannels.size(); ++k) {
        writeToLog("Grid" + to_string(k) + ": ");
        endLine();
        for (int i = CNNetwork.pooledChannels[k].size() - 1; i >= 0; --i) {
            for (int j = 0; j < CNNetwork.pooledChannels[k][0].size(); ++j) {
                if (CNNetwork.pooledChannels[k][i][j][slice] > 0.9) {
                    writeToLogNoLine("@");
                }
                else if (CNNetwork.pooledChannels[k][i][j][slice] > 0.8) {
                    writeToLogNoLine("%");
                }
                else if (CNNetwork.pooledChannels[k][i][j][slice] > 0.7) {
                    writeToLogNoLine("#");
                }
                else if (CNNetwork.pooledChannels[k][i][j][slice] > 0.6) {
                    writeToLogNoLine("*");
                }
                else if (CNNetwork.pooledChannels[k][i][j][slice] > 0.5) {
                    writeToLogNoLine("+");
                }
                else if (CNNetwork.pooledChannels[k][i][j][slice] > 0.4) {
                    writeToLogNoLine("=");
                }
                else if (CNNetwork.pooledChannels[k][i][j][slice] > 0.3) {
                    writeToLogNoLine("~");
                }
                else if (CNNetwork.pooledChannels[k][i][j][slice] > 0.2) {
                    writeToLogNoLine("-");
                }
                else if (CNNetwork.pooledChannels[k][i][j][slice] > 0.1) {
                    writeToLogNoLine(",");
                }
                else if (CNNetwork.pooledChannels[k][i][j][slice] > 0) {
                    writeToLogNoLine(".");
                }
                else if (CNNetwork.pooledChannels[k][i][j][slice] == 0) {
                    writeToLogNoLine(" ");
                }
            }
            endLine();
        }
    }

    

    for (int k = 0; k < CNNetwork.outputChannel.size(); ++k) {
        writeToLog("Grid" + to_string(k) + ": ");
        endLine();
        for (int i = CNNetwork.outputChannel[k].size() - 1; i >= 0; --i) {
            for (int j = 0; j < CNNetwork.outputChannel[k][0].size(); ++j) {
                if (CNNetwork.outputChannel[k][i][j][slice] > 0.9) {
                    writeToLogNoLine("@");
                }
                else if (CNNetwork.outputChannel[k][i][j][slice] > 0.8) {
                    writeToLogNoLine("%");
                }
                else if (CNNetwork.outputChannel[k][i][j][slice] > 0.7) {
                    writeToLogNoLine("#");
                }
                else if (CNNetwork.outputChannel[k][i][j][slice] > 0.6) {
                    writeToLogNoLine("*");
                }
                else if (CNNetwork.outputChannel[k][i][j][slice] > 0.5) {
                    writeToLogNoLine("+");
                }
                else if (CNNetwork.outputChannel[k][i][j][slice] > 0.4) {
                    writeToLogNoLine("=");
                }
                else if (CNNetwork.outputChannel[k][i][j][slice] > 0.3) {
                    writeToLogNoLine("~");
                }
                else if (CNNetwork.outputChannel[k][i][j][slice] > 0.2) {
                    writeToLogNoLine("-");
                }
                else if (CNNetwork.outputChannel[k][i][j][slice] > 0.1) {
                    writeToLogNoLine(",");
                }
                else if (CNNetwork.outputChannel[k][i][j][slice] > 0) {
                    writeToLogNoLine(".");
                }
                else if (CNNetwork.outputChannel[k][i][j][slice] == 0) {
                    writeToLogNoLine(" ");
                }
            }
            endLine();
        }
    }


    for (int i = CNNetwork.finalBinaryGrid.size() - 1; i >= 0; --i) {
        for (int j = 0; j < CNNetwork.finalBinaryGrid[0].size(); ++j) {
            if (CNNetwork.finalBinaryGrid[i][j][slice] > 0.9) {
                writeToLogNoLine("@");
            }
            else if (CNNetwork.finalBinaryGrid[i][j][slice] > 0.8) {
                writeToLogNoLine("%");
            }
            else if (CNNetwork.finalBinaryGrid[i][j][slice] > 0.7) {
                writeToLogNoLine("#");
            }
            else if (CNNetwork.finalBinaryGrid[i][j][slice] > 0.6) {
                writeToLogNoLine("*");
            }
            else if (CNNetwork.finalBinaryGrid[i][j][slice] > 0.5) {
                writeToLogNoLine("+");
            }
            else if (CNNetwork.finalBinaryGrid[i][j][slice] > 0.4) {
                writeToLogNoLine("=");
            }
            else if (CNNetwork.finalBinaryGrid[i][j][slice] > 0.3) {
                writeToLogNoLine("~");
            }
            else if (CNNetwork.finalBinaryGrid[i][j][slice] > 0.2) {
                writeToLogNoLine("-");
            }
            else if (CNNetwork.finalBinaryGrid[i][j][slice] > 0.1) {
                writeToLogNoLine(",");
            }
            else if (CNNetwork.finalBinaryGrid[i][j][slice] > 0) {
                writeToLogNoLine(".");
            }
            else if (CNNetwork.finalBinaryGrid[i][j][slice] == 0) {
                writeToLogNoLine(" ");
            }
        }
        endLine();
    }
    slice = 56;

    writeToLog(to_string(CNNetwork.finalUpsampledGrid.size()) + to_string(CNNetwork.finalUpsampledGrid[0].size()) + to_string(CNNetwork.finalUpsampledGrid[0][0].size()));
    for (int i = CNNetwork.finalUpsampledGrid.size() - 1; i >= 0; --i) {
        for (int j = 0; j < CNNetwork.finalUpsampledGrid[0].size(); ++j) {
            if (CNNetwork.finalUpsampledGrid[i][j][slice] > 0.9) {
                writeToLogNoLine("@");
            }
            else if (CNNetwork.finalUpsampledGrid[i][j][slice] > 0.8) {
                writeToLogNoLine("%");
            }
            else if (CNNetwork.finalUpsampledGrid[i][j][slice] > 0.7) {
                writeToLogNoLine("#");
            }
            else if (CNNetwork.finalUpsampledGrid[i][j][slice] > 0.6) {
                writeToLogNoLine("*");
            }
            else if (CNNetwork.finalUpsampledGrid[i][j][slice] > 0.5) {
                writeToLogNoLine("+");
            }
            else if (CNNetwork.finalUpsampledGrid[i][j][slice] > 0.4) {
                writeToLogNoLine("=");
            }
            else if (CNNetwork.finalUpsampledGrid[i][j][slice] > 0.3) {
                writeToLogNoLine("~");
            }
            else if (CNNetwork.finalUpsampledGrid[i][j][slice] > 0.2) {
                writeToLogNoLine("-");
            }
            else if (CNNetwork.finalUpsampledGrid[i][j][slice] > 0.1) {
                writeToLogNoLine(",");
            }
            else if (CNNetwork.finalUpsampledGrid[i][j][slice] > 0) {
                writeToLogNoLine(".");
            }
            else if (CNNetwork.finalUpsampledGrid[i][j][slice] <= 0) {
                writeToLogNoLine("/");
            }
        }
        endLine();
    }

    writeToLog(to_string(CNNetwork.groundTruthGrid.size()) + to_string(CNNetwork.groundTruthGrid[0].size()) + to_string(CNNetwork.groundTruthGrid[0][0].size()));
    for (int i = CNNetwork.groundTruthGrid.size() - 1; i >= 0; --i) {
        for (int j = 0; j < CNNetwork.groundTruthGrid[0].size(); ++j) {
            if (CNNetwork.groundTruthGrid[i][j][slice] > 0.9) {
                writeToLogNoLine("@");
            }
            else if (CNNetwork.groundTruthGrid[i][j][slice] > 0.8) {
                writeToLogNoLine("%");
            }
            else if (CNNetwork.groundTruthGrid[i][j][slice] > 0.7) {
                writeToLogNoLine("#");
            }
            else if (CNNetwork.groundTruthGrid[i][j][slice] > 0.6) {
                writeToLogNoLine("*");
            }
            else if (CNNetwork.groundTruthGrid[i][j][slice] > 0.5) {
                writeToLogNoLine("+");
            }
            else if (CNNetwork.groundTruthGrid[i][j][slice] > 0.4) {
                writeToLogNoLine("=");
            }
            else if (CNNetwork.groundTruthGrid[i][j][slice] > 0.3) {
                writeToLogNoLine("~");
            }
            else if (CNNetwork.groundTruthGrid[i][j][slice] > 0.2) {
                writeToLogNoLine("-");
            }
            else if (CNNetwork.groundTruthGrid[i][j][slice] > 0.1) {
                writeToLogNoLine(",");
            }
            else if (CNNetwork.groundTruthGrid[i][j][slice] > 0) {
                writeToLogNoLine(".");
            }
            else if (CNNetwork.groundTruthGrid[i][j][slice] <= 0) {
                writeToLogNoLine(" ");
            }
        }
        endLine();
    }

    vector<vector<vector<float>>> _;
    vector<vector<vector<float>>> d = {{{0.f}}};

   
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

