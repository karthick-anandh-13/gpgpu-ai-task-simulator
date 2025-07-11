[Setup]
AppName=GPGPU AI Task Simulator
AppVersion=1.0
DefaultDirName={autopf}\GPGPU_AI_Task_Simulator
DefaultGroupName=GPGPU AI Task Simulator
OutputBaseFilename=GPGPU_AI_Installer
Compression=lzma
SolidCompression=yes

[Files]
Source: "dist\gui.exe"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\GPGPU AI Task Simulator"; Filename: "{app}\gui.exe"
Name: "{commondesktop}\GPGPU AI Task Simulator"; Filename: "{app}\gui.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional icons:"