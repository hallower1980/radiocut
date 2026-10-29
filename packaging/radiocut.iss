#define MyVersion GetEnv("RC_VERSION")
[Setup]
AppName=RadioCut
AppVersion={#MyVersion}
AppPublisher=hallower1980
DefaultDirName={localappdata}\Programs\RadioCut
PrivilegesRequired=lowest
DefaultGroupName=RadioCut
OutputDir=..\dist
OutputBaseFilename=RadioCut-Setup-Windows-x64
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
[Files]
Source: "..\dist\RadioCut\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion
[Icons]
Name: "{group}\RadioCut"; Filename: "{app}\RadioCut.exe"
Name: "{userdesktop}\RadioCut"; Filename: "{app}\RadioCut.exe"; Tasks: desktopicon
[Tasks]
Name: desktopicon; Description: "Create a desktop shortcut"
[Run]
Filename: "{app}\RadioCut.exe"; Description: "Launch RadioCut"; Flags: nowait postinstall skipifsilent
