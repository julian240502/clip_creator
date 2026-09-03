' Lance start.bat sans afficher de fenetre de console.
' Utilise pour le demarrage automatique a la connexion Windows.
Set fso = CreateObject("Scripting.FileSystemObject")
scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
batPath = fso.BuildPath(scriptDir, "start.bat")

Set shell = CreateObject("WScript.Shell")
shell.CurrentDirectory = scriptDir
shell.Run """" & batPath & """", 0, False
