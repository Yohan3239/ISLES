// ISLES.cpp : Defines the entry point for the application.
//

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

namespace fs = std::filesystem;
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

    std::ofstream logFile("C:\\Users\\yohan\\source\\repos\\ISLES\\log.txt", std::ios::trunc);

    std::string filename_FLAIR = "C:\\Users\\yohan\\Documents\\ISLES-2022\\ISLES-2022\\sub-strokecase0001\\ses-0001\\anat\\sub-strokecase0001_ses-0001_FLAIR.nii\\sub-strokecase0001_ses-0001_FLAIR.nii";
    std::string filename_ADC = "C:\\Users\\yohan\\Documents\\ISLES-2022\\ISLES-2022\\sub-strokecase0001\\ses-0001\\dwi\\sub-strokecase0001_ses-0001_adc.nii\\sub-strokeperf0041_ses-20180528_ornt-2iso_skull-stripped_sequ-306_adc.nii";
    std::string filename_DWI = "C:\\Users\\yohan\\Documents\\ISLES-2022\\ISLES-2022\\sub-strokecase0001\\ses-0001\\dwi\\sub-strokecase0001_ses-0001_dwi.nii\\sub-strokeperf0041_ses-20180528_ornt-2iso_skull-stripped_sequ-307_dwi.nii";


    

    CNN CNNetwork;

    writeToLog("Trying to open file at: " + filename_ADC);
    CNNetwork.readNiftiHeader(filename_ADC, false);
    writeToLog("Completed reading ADC.");

    writeToLog("Trying to open file at: " + filename_DWI);
    CNNetwork.readNiftiHeader(filename_DWI, false);
    writeToLog("Completed reading DWI.");

    writeToLog("Trying to open file at: " + filename_FLAIR);
    CNNetwork.readNiftiHeader(filename_FLAIR, true);
    //CNNetwork.resample(); To be implemented
    writeToLog("Completed reading FLAIR.");

    CNNetwork.initialiseFilter(CNNetwork.filter, 1, 3, 3, 3);
    //CNNetwork.convolve(CNNetwork.voxelsGrid, CNNetwork.filter, CNNetwork.convolveGrid, 1); To be implemented
    

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

