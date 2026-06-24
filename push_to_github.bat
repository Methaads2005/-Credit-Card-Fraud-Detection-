@echo off
title Push to GitHub - Credit Card Fraud Detection
echo ============================================================
echo           Push Project to GitHub Repository
echo ============================================================
echo.

:: Check if git is installed
where git >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Git is not installed or not in your system PATH.
    echo Please download and install Git from: https://git-scm.com/downloads
    echo.
    echo After installing, restart your terminal and double-click this file again.
    echo.
    pause
    exit /b
)

:: Ask for the GitHub repository URL
set /p REPO_URL="Enter your GitHub Repository URL (e.g., https://github.com/username/repo.git): "

if "%REPO_URL%"=="" (
    echo [ERROR] Repository URL cannot be empty.
    pause
    exit /b
)

echo.
echo [1/5] Initializing Git repository...
if not exist .git (
    git init
) else (
    echo Git repository already initialized.
)

echo.
echo [2/5] Staging files...
git add .

echo.
echo [3/5] Committing files...
git commit -m "Initial commit: Credit Card Fraud Detection Major Project"

echo.
echo [4/5] Setting branch name to main and linking remote...
git branch -M main
git remote remove origin >nul 2>nul
git remote add origin %REPO_URL%

echo.
echo [5/5] Pushing files to GitHub...
echo (You may be prompted to log in to GitHub in a separate window)
echo.
git push -u origin main

if %errorlevel% equ 0 (
    echo.
    echo ============================================================
    echo [SUCCESS] Project successfully pushed to GitHub!
    echo ============================================================
) else (
    echo.
    echo [ERROR] Failed to push to GitHub. Please check your credentials or URL.
)

echo.
pause
