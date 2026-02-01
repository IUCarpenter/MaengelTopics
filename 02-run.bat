@echo off
call nlpEnv\Scripts\activate.bat

set /p K=K-Wert eingeben: 
set /p COH=Coherence? (J/N): 

if /I "%COH%"=="J" (
  python pipeline.py -k %K% -coh
) else (
  python pipeline.py -k %K%
)

pause
