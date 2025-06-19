@echo off
color 06
echo.
echo  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
echo  ::                                                                        ::
echo  ::  AI MIXER TOURNAMENT - YOUR TRACK, YOUR SOUND                          ::
echo  ::  ==============================================                        ::
echo  ::                                                                        ::
echo  ::  LAUNCHING THE MIXER ARENA...                                          ::
echo  ::  WARMING UP THE BEATS...                                               ::
echo  ::  GETTING YOUR MIX MASTERS READY!                                       ::
echo  ::                                                                        ::
echo  ::  * 5 PRO AI MODELS READY TO ENHANCE YOUR SOUND                         ::
echo  ::  * PERFECT FOR HIP-HOP, POP, ROCK, EDM & MORE                          ::
echo  ::  * YOUR TRACK, YOUR CHOICE, YOUR STYLE                                 ::
echo  ::                                                                        ::
echo  ::  For full details, check TOURNAMENT_GUIDE.md                           ::
echo  ::                                                                        ::
echo  ::  CREATE. COMPETE. PERFECT YOUR SOUND.                                  ::
echo  ::                                                                        ::
echo  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
echo.

cd %~dp0\tournament_webapp\frontend
echo [+] LAUNCHING YOUR MIXER AT: http://localhost:3000
echo [+] GET READY TO CREATE YOUR PERFECT MIX...
echo.
echo [i] Start by entering your artist name and uploading your track.
echo [i] Our AI models will create different versions of your mix.
echo [i] You choose which sounds better - the system learns from your style.
echo [i] With each round, the mix gets closer to your perfect sound!
echo.
echo [i] Works great with: Hip-Hop, Pop, R^&B, Rock, EDM, and more...
echo.

REM Kill any existing process on port 3000
FOR /F "tokens=5" %%P IN ('netstat -ano ^| findstr :3000 ^| findstr LISTENING') DO (
  echo [!] Releasing port 3000...
  taskkill /PID %%P /F >nul 2>&1
)

echo [+] Building latest version...
call npm run build

echo [+] Starting server...
npx serve -s build -l 3000
